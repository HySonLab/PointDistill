"""
Point Transformer - V3 Mode1
Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from addict import Dict
import math
import numpy as np
import torch
import torch.nn as nn
import spconv.pytorch as spconv
from torch_scatter import segment_csr
from timm.models.layers import DropPath
from collections import OrderedDict
# import ripserplusplus as rpp_py
import time
from loguru import logger as custom_logger

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.point_prompt_training import PDNorm
from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
import pointcept.utils.comm as comm
from pointcept.models.modules import PointModule, PointSequential
from spconv.pytorch import SparseConvTensor
from fvcore.nn import FlopCountAnalysis, flop_count_table
import copy


class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        point_dict = Dict(
            feat=segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


@MODELS.register_module("PT-v3-calculate-gflops")
class PointTransformerV3TrainTeacher(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        context_channels=256,
        backbone_out_channels=64,
        num_classes=16
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        act_layer = nn.GELU

        self.conditions = pdnorm_conditions
        self.embedding_table = nn.Embedding(len(pdnorm_conditions), context_channels)
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[sum(enc_depths[:s]):sum(enc_depths[:s + 1])]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[sum(dec_depths[:s]):sum(dec_depths[:s + 1])]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def calculate_flops(self, point):
        """Calculate FLOPs for each part of the model."""
        flops_dict = {
            "Embedding": 0.0,
            "Encoder": {},
            "Decoder": {},
            "Segmentation Head": 0.0,
            "Total": 0.0
        }

        # Helper function to create a deep copy of a Point object
        def copy_point(point):
            new_point = Point()
            for key, value in point.items():
                if isinstance(value, torch.Tensor):
                    new_point[key] = value.clone().detach()
                elif isinstance(value, spconv.SparseConvTensor):
                    new_point[key] = spconv.SparseConvTensor(
                        features=value.features.clone().detach(),
                        indices=value.indices.clone().detach(),
                        spatial_shape=value.spatial_shape,
                        batch_size=value.batch_size
                    )
                elif isinstance(value, (list, tuple)):
                    new_point[key] = [v.clone().detach() if isinstance(v, torch.Tensor) else v for v in value]
                else:
                    new_point[key] = copy.deepcopy(value)
            return new_point

        # Helper function to compute FLOPs for dense operations only
        def compute_dense_flops(module, input_data, name):
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Dropout, DropPath)):
                try:
                    flops = FlopCountAnalysis(module, input_data)
                    total_flops = flops.total() / 1e9  # Convert to GFLOPs
                    # custom_logger.info(f"FLOPs for {name}: {total_flops:.2f} GFLOPs")
                    return total_flops
                except Exception as e:
                    custom_logger.warning(f"Failed to compute FLOPs for {name}: {str(e)}")
                    return 0.0
            elif isinstance(module, spconv.SubMConv3d):
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_size = module.kernel_size[0] ** 3
                num_points = input_data.features.shape[0]
                flops = 2 * in_channels * out_channels * kernel_size * num_points / 1e9
                # custom_logger.info(f"Approximated FLOPs for {name} (sparse conv): {flops:.2f} GFLOPs")
                return flops
            return 0.0

        # Embedding FLOPs
        point_copy = copy_point(point)
        point_copy = self.embedding(point_copy)  # Transform feat to enc_channels[0]
        for name, module in self.embedding.stem.named_modules():
            if name == "conv":
                flops_dict["Embedding"] += compute_dense_flops(module, point.sparse_conv_feat, "Embedding.conv")
            elif name in ["norm", "act"]:
                flops_dict["Embedding"] += compute_dense_flops(module, point_copy.feat, f"Embedding.{name}")
        flops_dict["Total"] += flops_dict["Embedding"]

        # Encoder FLOPs
        point_copy_enc = copy_point(point_copy)  # Start with embedded output
        for stage_name, stage in self.enc.named_children():
            flops_dict["Encoder"][stage_name] = {"Pooling": 0.0, "Blocks": {}}
            point_copy_stage = copy_point(point_copy_enc)
            for module_name, module in stage.named_children():
                if module_name == "down":  # Pooling
                    flops = compute_dense_flops(module.proj, point_copy_stage.feat, f"{stage_name}.Pooling.proj")
                    flops_dict["Encoder"][stage_name]["Pooling"] += flops
                    point_copy_stage = module(point_copy_stage)
                elif module_name.startswith("block"):  # Attention Blocks
                    block_flops = 0.0
                    # CPE
                    cpe_conv_out = module.cpe[0](point_copy_stage.sparse_conv_feat)
                    block_flops += compute_dense_flops(module.cpe[0], point_copy_stage.sparse_conv_feat, f"{stage_name}.{module_name}.cpe.conv")
                    cpe_linear_out = module.cpe[1](cpe_conv_out.features)  # Use features from sparse conv output
                    block_flops += compute_dense_flops(module.cpe[1], cpe_conv_out.features, f"{stage_name}.{module_name}.cpe.linear")
                    block_flops += compute_dense_flops(module.cpe[2], cpe_linear_out, f"{stage_name}.{module_name}.cpe.norm")
                    # Attention
                    block_flops += compute_dense_flops(module.attn.qkv, point_copy_stage.feat, f"{stage_name}.{module_name}.attn.qkv")
                    block_flops += compute_dense_flops(module.attn.proj, point_copy_stage.feat, f"{stage_name}.{module_name}.attn.proj")
                    p = point_copy_stage.feat.shape[0]
                    heads = module.attn.num_heads
                    patch_size = module.attn.patch_size
                    attn_flops = 2 * p * min(patch_size, p) * heads / 3 / 1e9
                    block_flops += attn_flops
                    # MLP
                    mlp_fc1_out = module.mlp[0].fc1(point_copy_stage.feat)
                    block_flops += compute_dense_flops(module.mlp[0].fc1, point_copy_stage.feat, f"{stage_name}.{module_name}.mlp.fc1")
                    block_flops += compute_dense_flops(module.mlp[0].fc2, mlp_fc1_out, f"{stage_name}.{module_name}.mlp.fc2")
                    # Norms and DropPath
                    block_flops += compute_dense_flops(module.norm1[0], point_copy_stage.feat, f"{stage_name}.{module_name}.norm1")
                    block_flops += compute_dense_flops(module.norm2[0], point_copy_stage.feat, f"{stage_name}.{module_name}.norm2")
                    block_flops += compute_dense_flops(module.drop_path[0], point_copy_stage.feat, f"{stage_name}.{module_name}.drop_path")
                    flops_dict["Encoder"][stage_name]["Blocks"][module_name] = block_flops
                    point_copy_stage = module(point_copy_stage)
            stage_total = flops_dict["Encoder"][stage_name]["Pooling"] + sum(flops_dict["Encoder"][stage_name]["Blocks"].values())
            flops_dict["Encoder"][stage_name]["Total"] = stage_total
            flops_dict["Total"] += stage_total
            point_copy_enc = point_copy_stage  # Update for next stage

        # Decoder FLOPs
        if not self.cls_mode:
            point_copy_dec = copy_point(point_copy_enc)  # Start with encoder output
            for stage_name, stage in self.dec.named_children():
                flops_dict["Decoder"][stage_name] = {"Unpooling": 0.0, "Blocks": {}}
                point_copy_stage = copy_point(point_copy_dec)
                for module_name, module in stage.named_children():
                    if module_name == "up":  # Unpooling
                        flops_dict["Decoder"][stage_name]["Unpooling"] += compute_dense_flops(
                            module.proj[0], point_copy_stage.feat, f"{stage_name}.Unpooling.proj"
                        )
                        flops_dict["Decoder"][stage_name]["Unpooling"] += compute_dense_flops(
                            module.proj_skip[0], point_copy_stage.pooling_parent.feat, f"{stage_name}.Unpooling.proj_skip"
                        )
                        point_copy_stage = module(point_copy_stage)
                    elif module_name.startswith("block"):
                        block_flops = 0.0
                        cpe_conv_out = module.cpe[0](point_copy_stage.sparse_conv_feat)
                        block_flops += compute_dense_flops(module.cpe[0], point_copy_stage.sparse_conv_feat, f"{stage_name}.{module_name}.cpe.conv")
                        cpe_linear_out = module.cpe[1](cpe_conv_out.features)
                        block_flops += compute_dense_flops(module.cpe[1], cpe_conv_out.features, f"{stage_name}.{module_name}.cpe.linear")
                        block_flops += compute_dense_flops(module.cpe[2], cpe_linear_out, f"{stage_name}.{module_name}.cpe.norm")
                        block_flops += compute_dense_flops(module.attn.qkv, point_copy_stage.feat, f"{stage_name}.{module_name}.attn.qkv")
                        block_flops += compute_dense_flops(module.attn.proj, point_copy_stage.feat, f"{stage_name}.{module_name}.attn.proj")
                        p = point_copy_stage.feat.shape[0]
                        heads = module.attn.num_heads
                        patch_size = module.attn.patch_size
                        attn_flops = 2 * p * min(patch_size, p) * heads / 3 / 1e9
                        block_flops += attn_flops
                        mlp_fc1_out = module.mlp[0].fc1(point_copy_stage.feat)
                        block_flops += compute_dense_flops(module.mlp[0].fc1, point_copy_stage.feat, f"{stage_name}.{module_name}.mlp.fc1")
                        block_flops += compute_dense_flops(module.mlp[0].fc2, mlp_fc1_out, f"{stage_name}.{module_name}.mlp.fc2")
                        block_flops += compute_dense_flops(module.norm1[0], point_copy_stage.feat, f"{stage_name}.{module_name}.norm1")
                        block_flops += compute_dense_flops(module.norm2[0], point_copy_stage.feat, f"{stage_name}.{module_name}.norm2")
                        block_flops += compute_dense_flops(module.drop_path[0], point_copy_stage.feat, f"{stage_name}.{module_name}.drop_path")
                        flops_dict["Decoder"][stage_name]["Blocks"][module_name] = block_flops
                        point_copy_stage = module(point_copy_stage)
                stage_total = flops_dict["Decoder"][stage_name]["Unpooling"] + sum(flops_dict["Decoder"][stage_name]["Blocks"].values())
                flops_dict["Decoder"][stage_name]["Total"] = stage_total
                flops_dict["Total"] += stage_total
                point_copy_dec = point_copy_stage  # Update for next stage

        # Segmentation Head FLOPs
        flops_dict["Segmentation Head"] = compute_dense_flops(self.seg_head, point_copy_dec.feat if not self.cls_mode else point_copy_enc.feat, "Segmentation Head")
        flops_dict["Total"] += flops_dict["Segmentation Head"]

        # Summarize
        custom_logger.info(">>>>>>>>>>>>>>>> FLOPs Breakdown <<<<<<<<<<<<<<<")
        custom_logger.info(f"Embedding: {flops_dict['Embedding']:.2f} GFLOPs")
        for stage_name, stage_flops in flops_dict["Encoder"].items():
            custom_logger.info(f"Encoder {stage_name}: {stage_flops['Total']:.2f} GFLOPs")
        if not self.cls_mode:
            for stage_name, stage_flops in flops_dict["Decoder"].items():
                custom_logger.info(f"Decoder {stage_name}: {stage_flops['Total']:.2f} GFLOPs")
        custom_logger.info(f"Segmentation Head: {flops_dict['Segmentation Head']:.2f} GFLOPs")
        custom_logger.info(f"Total: {flops_dict['Total']:.2f} GFLOPs")

        return flops_dict

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        # Move to CUDA
        for key, value in point.items():
            if isinstance(value, torch.Tensor):
                point[key] = value.cuda()
            elif isinstance(value, spconv.SparseConvTensor):
                point[key] = spconv.SparseConvTensor(
                    features=value.features.cuda(),
                    indices=value.indices.cuda(),
                    spatial_shape=value.spatial_shape,
                    batch_size=value.batch_size
                )

        # Calculate FLOPs
        flops_dict = self.calculate_flops(point)

        # Forward pass
        point = self.embedding(point)
        point = self.enc(point)
        if not self.cls_mode:
            point = self.dec(point)
        else:
            point.feat = segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
        seg_logits = self.seg_head(point.feat)

        return seg_logits, flops_dict

