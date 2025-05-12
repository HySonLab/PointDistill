"""
Tester

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import time
import numpy as np
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
from loguru import logger as custom_logger
from .defaults import create_ddp_model
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
from pointcept.utils.visualization import to_numpy, get_label_colors, add_legend
from torchprofile import profile_macs
from torch_geometric.nn import fps as torch_fps
from pointcept.utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D


TESTERS = Registry("testers")


class TesterBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.verbose = verbose
        if self.verbose:
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self):
        # How to handle test data:
        # /data/user/home/luutunghai@gmail.com/projects/PTv3-distill/pointcept/datasets/defaults.py in prepare_test_data
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test,
            shuffle=False,
            num_workers=self.cfg.num_worker,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader

    def test(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)


@TESTERS.register_module()
class SemSegTester(TesterBase):
    def test(self):

        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()

        total_time = 0.0  # Track total inference time
        total_samples = 0  # Track the number of processed samples

        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        # create submit folder only on main process
        if (
            self.cfg.data.test.type == "ScanNetDataset"
            or self.cfg.data.test.type == "ScanNet200Dataset"
            or self.cfg.data.test.type == "ScanNetPPDataset"
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif (
            self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json

            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                os.path.join(save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)
        comm.synchronize()
        record = {}
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):

            # Track inference start time
            start_time = time.time()

            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
            # if os.path.isfile(pred_save_path):
            #     logger.info(
            #         "{}/{}: {}, loaded pred and label.".format(
            #             idx + 1, len(self.test_loader), data_name
            #         )
            #     )
            #     pred = np.load(pred_save_path)
            #     if "origin_segment" in data_dict.keys():
            #         segment = data_dict["origin_segment"]
            # else:
            pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
            for i in range(len(fragment_list)):
                fragment_batch_size = 1
                s_i, e_i = i * fragment_batch_size, min(
                    (i + 1) * fragment_batch_size, len(fragment_list)
                )
                input_dict = collate_fn(fragment_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                idx_part = input_dict["index"]
                with torch.no_grad():
                    pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                    pred_part = F.softmax(pred_part, -1)
                    if self.cfg.empty_cache:
                        torch.cuda.empty_cache()
                    bs = 0
                    for be in input_dict["offset"]:
                        pred[idx_part[bs:be], :] += pred_part[bs:be]
                        bs = be

            if self.cfg.data.test.type == "ScanNetPPDataset":
                pred = pred.topk(3, dim=1)[1].data.cpu().numpy()
            else:
                pred = pred.max(1)[1].data.cpu().numpy()
            if "origin_segment" in data_dict.keys():
                assert "inverse" in data_dict.keys()
                pred = pred[data_dict["inverse"]]
                segment = data_dict["origin_segment"]
            np.save(pred_save_path, pred)

            if (
                self.cfg.data.test.type == "ScanNetDataset"
                or self.cfg.data.test.type == "ScanNet200Dataset"
            ):
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
                    fmt="%d",
                )
            elif self.cfg.data.test.type == "ScanNetPPDataset":
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    pred.astype(np.int32),
                    delimiter=",",
                    fmt="%d",
                )
                pred = pred[:, 0]  # for mIoU, TODO: support top3 mIoU
            elif self.cfg.data.test.type == "SemanticKITTIDataset":
                # 00_000000 -> 00, 000000
                sequence_name, frame_name = data_name.split("_")
                os.makedirs(
                    os.path.join(
                        save_path, "submit", "sequences", sequence_name, "predictions"
                    ),
                    exist_ok=True,
                )
                submit = pred.astype(np.uint32)
                submit = np.vectorize(
                    self.test_loader.dataset.learning_map_inv.__getitem__
                )(submit).astype(np.uint32)
                submit.tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "sequences",
                        sequence_name,
                        "predictions",
                        f"{frame_name}.label",
                    )
                )
            elif self.cfg.data.test.type == "NuScenesDataset":
                np.array(pred + 1).astype(np.uint8).tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "lidarseg",
                        "test",
                        "{}_lidarseg.bin".format(data_name),
                    )
                )

            intersection, union, target = intersection_and_union(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            # Compute FPS
            # Track inference end time
            end_time = time.time()
            total_time += (end_time - start_time)
            total_samples += 1

            # fps = 1 / total_time if total_time > 0 else 0
            fps = total_samples / total_time if total_time > 0 else 0  # spf

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f}) "
                "FPS {fps:.4f}".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.shape,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                    fps=fps
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            if self.cfg.data.test.type == "S3DISDataset":
                torch.save(
                    dict(intersection=intersection, union=union, target=target),
                    os.path.join(save_path, f"{self.test_loader.dataset.split}.pth"),
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class ClsTester(TesterBase):
    def test(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        for i, input_dict in enumerate(self.test_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            end = time.time()
            with torch.no_grad():
                output_dict = self.model(input_dict)
            output = output_dict["cls_logits"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred, label, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            intersection_meter.update(intersection), union_meter.update(
                union
            ), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)

            logger.info(
                "Test: [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {accuracy:.4f} ".format(
                    i + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    accuracy=accuracy,
                )
            )

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU, mAcc, allAcc
            )
        )

        for i in range(self.cfg.data.num_classes):
            logger.info(
                "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=accuracy_class[i],
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


@TESTERS.register_module()
class ClsVotingTester(TesterBase):
    def __init__(
        self,
        num_repeat=100,
        metric="allAcc",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_repeat = num_repeat
        self.metric = metric
        self.best_idx = 0
        self.best_record = None
        self.best_metric = 0

    def test(self):
        for i in range(self.num_repeat):
            logger = get_root_logger()
            logger.info(f">>>>>>>>>>>>>>>> Start Evaluation {i + 1} >>>>>>>>>>>>>>>>")
            record = self.test_once()
            if comm.is_main_process():
                if record[self.metric] > self.best_metric:
                    self.best_record = record
                    self.best_idx = i
                    self.best_metric = record[self.metric]
                info = f"Current best record is Evaluation {i + 1}: "
                for m in self.best_record.keys():
                    info += f"{m}: {self.best_record[m]:.4f} "
                logger.info(info)

    def test_once(self):
        logger = get_root_logger()
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        target_meter = AverageMeter()
        record = {}
        self.model.eval()

        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            voting_list = data_dict.pop("voting_list")
            category = data_dict.pop("category")
            data_name = data_dict.pop("name")
            # pred = torch.zeros([1, self.cfg.data.num_classes]).cuda()
            # for i in range(len(voting_list)):
            #     input_dict = voting_list[i]
            #     for key in input_dict.keys():
            #         if isinstance(input_dict[key], torch.Tensor):
            #             input_dict[key] = input_dict[key].cuda(non_blocking=True)
            #     with torch.no_grad():
            #         pred += F.softmax(self.model(input_dict)["cls_logits"], -1)
            input_dict = collate_fn(voting_list)
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                pred = F.softmax(self.model(input_dict)["cls_logits"], -1).sum(
                    0, keepdim=True
                )
            pred = pred.max(1)[1].cpu().numpy()
            intersection, union, target = intersection_and_union(
                pred, category, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            target_meter.update(target)
            record[data_name] = dict(intersection=intersection, target=target)
            acc = sum(intersection) / (sum(target) + 1e-10)
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))
            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) ".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)
            accuracy_class = intersection / (target + 1e-10)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info("Val result: mAcc/allAcc {:.4f}/{:.4f}".format(mAcc, allAcc))
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        accuracy=accuracy_class[i],
                    )
                )
            return dict(mAcc=mAcc, allAcc=allAcc)

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class CustomSemSegTester(TesterBase):

    def test(self):

        assert self.test_loader.batch_size == 1, "Batch size of test data should be 1"
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()

        total_time = 0.0  # Track total inference time
        total_samples = 0  # Track the number of processed samples

        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        # create submit folder only on main process
        if (
                self.cfg.data.test.type == "ScanNetDataset"
                or self.cfg.data.test.type == "ScanNet200Dataset"
                or self.cfg.data.test.type == "ScanNetPPDataset"
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif (
                self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json

            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                    os.path.join(save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)

        comm.synchronize()
        record = {}
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):

            # Track inference start time
            start_time = time.time()

            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            # pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
            # if os.path.isfile(pred_save_path):
            #     # logger.info(
            #     #     "{}/{}: {}, loaded pred and label.".format(
            #     #         idx + 1, len(self.test_loader), data_name
            #     #     )
            #     # )
            #     pred = np.load(pred_save_path)
            # else:
            pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()

            for i in range(len(fragment_list)):
                fragment_batch_size = 1
                s_i, e_i = i * fragment_batch_size, min(
                    (i + 1) * fragment_batch_size, len(fragment_list)
                )
                input_dict = collate_fn(fragment_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                idx_part = input_dict["index"]
                with torch.no_grad():

                    pred_part = self.model(input_dict)  # (n, k)

                    # pred_part = pred_part["seg_logits"]
                    pred_part = F.softmax(pred_part, -1)
                    if self.cfg.empty_cache:
                        torch.cuda.empty_cache()
                    bs = 0
                    for be in input_dict["offset"]:
                        pred[idx_part[bs:be], :] += pred_part[bs:be]
                        bs = be
            # -----------------------------------------------------------------------
            pred = torch.argmax(pred, dim=1)
            segment = torch.from_numpy(segment).to(pred.device)
            intersection, union, target = intersection_and_union_gpu(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )

            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)

            acc = sum(intersection) / (sum(target) + 1e-10)

            iou = torch.mean(iou_class[mask])  # Directly use PyTorch mean
            m_iou = torch.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = torch.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - end)

            # Compute FPS

            # Track inference end time
            end_time = time.time()
            total_time += (end_time - start_time)
            total_samples += 1

            # fps = 1 / total_time if total_time > 0 else 0
            fps = total_samples / total_time if total_time > 0 else 0 # spf
            # fps = 1 / spf
            # if idx % 300 == 0:
            #     logger.info(
            #         "Test: {} [{}/{}]-{} "
            #         "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
            #         "Accuracy {acc:.4f} ({m_acc:.4f}) "
            #         "mIoU {iou:.4f} ({m_iou:.4f}) "
            #         "FPS {fps:.4f}".format(
            #             data_name,
            #             idx + 1,
            #             len(self.test_loader),
            #             segment.shape,
            #             batch_time=batch_time,
            #             acc=acc,
            #             m_acc=m_acc,
            #             iou=iou,
            #             m_iou=m_iou,
            #             fps=fps
            #         )
            #     )

            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f}) "
                "FPS {fps:.4f}".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.shape,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                    fps=fps
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r

        # Convert list of tensors to stacked tensor for sum operation
        intersection = torch.sum(torch.stack([meters["intersection"] for _, meters in record.items()]), dim=0)
        union = torch.sum(torch.stack([meters["union"] for _, meters in record.items()]), dim=0)
        target = torch.sum(torch.stack([meters["target"] for _, meters in record.items()]), dim=0)

        # Compute IoU and accuracy using PyTorch operations
        iou_class = intersection / (union + 1e-10)
        accuracy_class = intersection / (target + 1e-10)

        # Compute mean IoU and mean Accuracy using torch.mean()
        mIoU = torch.mean(iou_class)
        mAcc = torch.mean(accuracy_class)

        # Compute overall Accuracy
        allAcc = torch.sum(intersection) / (torch.sum(target) + 1e-10)

        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                mIoU, mAcc, allAcc
            )
        )

        if not os.path.exists(self.cfg.miou_result_path):
            os.makedirs(self.cfg.miou_result_path)
            print(f"Created directory {self.cfg.miou_result_path}")

        miou_result_file = os.path.join(self.cfg.miou_result_path, "miou_results.txt")
        with open(miou_result_file, "w") as f:
            for i in range(self.cfg.data.num_classes):
                result_line = "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.cfg.data.names[i],
                    iou=iou_class[i].item(),  # Convert tensor to float
                    accuracy=accuracy_class[i].item(),  # Convert tensor to float
                )

                # Print to console (optional)
                logger.info(result_line)

                # Write to file
                f.write(result_line + "\n")

        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    # This function examines the inference speed of each prediction stage
    def test_one_frame(self):

        assert self.test_loader.batch_size == 1, "Batch size of test data should be 1"
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()

        vis_save_path = '/home/luutunghai@gmail.com/projects/PTv3-distill/visualization/'
        # Start total frame timer
        frame_start_time = time.time()
        self.model.eval()

        total_fps = []

        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):

            if idx == 1:
                custom_logger.info(f"{idx} batch finished")
                break

            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")

            pred_start = time.time()
            pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()

            total_time_data_movement = 0

            for i in range(len(fragment_list)):

                # Step 2: Fragment-based processing
                print(f">>>>>>>>>>>>>>>>>>> Start of Frame {i} <<<<<<<<<<<<<<<<<<<<<<<") # custom_logger.info
                data_movement_start = time.time()
                fragment_batch_size = 1
                s_i, e_i = i * fragment_batch_size, min(
                    (i + 1) * fragment_batch_size, len(fragment_list)
                )
                input_dict = collate_fn(fragment_list[s_i:e_i])
                # coord_shape = input_dict['coord'].shape
                # segment_shape = segment.shape
                # feat_shape = input_dict['feat'].shape
                # offset = input_dict['offset']
                # with open("./debug/input_dict_test.txt", "a") as f:
                #     f.write(f">>>>>>>>>>>>>>>>>>> Start of Segment {i} <<<<<<<<<<<<<<<<<<<<<<<\n")
                #     f.write("input_dict:\n")
                #     f.write(str(input_dict))
                #     f.write("\n\nShapes:\n")
                #     f.write(f"coord shape: {coord_shape}\n")
                #     f.write(f"segment shape: {segment_shape}\n")
                #     f.write(f"feat shape: {feat_shape}\n")
                #     f.write(f"offset: {offset}\n")
                #     f.write(f">>>>>>>>>>>>>>>>>>> End of Segment {i} <<<<<<<<<<<<<<<<<<<<<<<\n")
                #     f.write(f" \n")
                # print("Done")
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                idx_part = input_dict["index"]
                data_movement_time = time.time() - data_movement_start
                total_time_data_movement += data_movement_time
                # custom_logger.info(f"GPU data movement (per frame): {data_movement_time:.7f}")

                with torch.no_grad():

                    # Step 2b: Model Inference
                    inference_start = time.time()

                    pred_part = self.model(input_dict)  # (n, k)
                    # pred_part = pred_part["seg_logits"]

                    # tuple because of ptv3 model for calculating gflops (prediction, gflops_dict)
                    if isinstance(pred_part, tuple):

                        pred_part = pred_part[0]

                    pred_part = F.softmax(pred_part, -1)
                    # torch.save(pred_part, f"./debug/pred_part_student_{i}.pt")

                    inference_time = time.time() - inference_start
                    print(f"Inference time of fragment {i} = {inference_time}")
                    print("NOTE: WE USE H100 TO ACHIEVE FPS ~ 27, PASCAL NODE ~ 12")
                    print(f"FPS: {1 / inference_time:.4f}")
                    # custom_logger.info(f"Inference time (per frame): {inference_time:.7f}")
                    total_fps.append(1 / inference_time)
                    # custom_logger.info(f"FPS: {1 / inference_time:.4f}")

                    # Step 2c: Post-Processing (per fragment)
                    post_processing_start = time.time()

                    if self.cfg.empty_cache:
                        torch.cuda.empty_cache()

                    bs = 0
                    for be in input_dict["offset"]:
                        pred[idx_part[bs:be], :] += pred_part[bs:be]
                        bs = be

                    post_processing_time = time.time() - post_processing_start
                    print(f"Post inference time (per frame): {post_processing_time:.7f}")

                    print(f">>>>>>>>>>>>>>>>>>> End of Frame {i} <<<<<<<<<<<<<<<<<<<<<<<")
                    print(" ")

            # -----------------------------------------------------------------------

            # custom_logger.info(f"Total data movement of a batch: {total_time_data_movement:.7f}")

            pred = torch.argmax(pred, dim=1)
            # torch.save(pred, f"./debug/pred_student.pt")
            pred_end = time.time() - pred_start
            # custom_logger.info(f"One complete batch prediction: {pred_end:.7f}")
            print(f"One complete batch prediction: {pred_end:.7f}")

            metric_calculate_start = time.time()
            segment = torch.from_numpy(segment).to(pred.device)
            intersection, union, target = intersection_and_union_gpu(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )

            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)

            mask = union != 0
            iou_class = intersection / (union + 1e-10)

            acc = sum(intersection) / (sum(target) + 1e-10)

            iou = torch.mean(iou_class[mask])  # Directly use PyTorch mean
            m_iou = torch.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = torch.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            metric_calculate_end = time.time() - metric_calculate_start
            # custom_logger.info(f"One batch metric calculation: {metric_calculate_end:.7f}")

            # Total frame time
            total_frame_time = time.time() - frame_start_time
            # custom_logger.info(f"One batch inference: {total_frame_time:.7f}")
            avg_fps = sum(total_fps) / len(total_fps)
            print(f"Avg FPS for {len(fragment_list)} frames: {avg_fps:.2f}")

    def visualize_prediction_and_GT(self, class_names, is_TTA_on=True, names="Nuscenes", scene_ids=[2, 4]):

        """

        :param class_names: class names
        :param is_TTA_on: if test time augmentation is on
        :param names: name of dataset
        :param scene_ids: specify which scene want to get
        :return: saved visualization
        """

        assert self.test_loader.batch_size == 1, "Batch size of test data should be 1"

        vis_save_path = '/home/luutunghai@gmail.com/projects/PTv3-distill/visualization/'
        self.model.eval()

        processed = 0

        # sweep inference
        for idx, data_dict in enumerate(self.test_loader):

            if idx not in scene_ids:
                continue

            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")

            pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
            GT_coord = torch.zeros((segment.size, 3)).cuda()

            for i in range(len(fragment_list)):

                fragment_batch_size = 1
                s_i, e_i = i * fragment_batch_size, min(
                    (i + 1) * fragment_batch_size, len(fragment_list)
                )
                input_dict = collate_fn(fragment_list[s_i:e_i])

                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                idx_part = input_dict["index"]

                with torch.no_grad():
                    pred_part = self.model(input_dict)  # (n, k)

                    # tuple because of ptv3 model for calculating gflops (prediction, gflops_dict)
                    if isinstance(pred_part, tuple):

                        pred_part = pred_part[0]

                    pred_part = F.softmax(pred_part, -1)

                    pred_part_labels = torch.argmax(pred_part, dim=-1)  # Shape: (24150,)
                    coord = input_dict["coord"]  # Shape: [24150, 3] --> example

                    vis_scene_save_path = os.path.join(vis_save_path, names + "_" + data_name)
                    os.makedirs(vis_scene_save_path, exist_ok=True)
                    pred_part_file_path = os.path.join(vis_scene_save_path, f"sweep_{i}_pred.png")

                    self.visualize_pointcloud_prediction(coords=coord, segment_pred=pred_part_labels,
                                                         class_names=class_names,
                                                         title="Predicted Semantic Segmentation "
                                                               "(Top-down view, single sweep)",
                                                         save_path=pred_part_file_path)

                    # torch.save((pred_part_labels, coord), pred_part_file_path)

                    if is_TTA_on:
                        # print(f"Saved (pred_segment, coordinates) of frame {i} with TTA")
                        print(f"Saved visualization prediction of sweep {i} with TTA")
                    else:
                        # print(f"Saved (pred_segment, coordinates) of frame {i}")
                        print(f"Saved visualization prediction of sweep {i}")

                    if self.cfg.empty_cache:
                        torch.cuda.empty_cache()

                    bs = 0
                    for be in input_dict["offset"]:
                        pred[idx_part[bs:be], :] += pred_part[bs:be]
                        GT_coord[idx_part[bs:be]] = coord[bs:be]

                        bs = be

                    print(" ")

            GT_save_path = os.path.join(vis_scene_save_path,
                                        f"GT_aggregate_multiple_sweeps.png")

            pred_multiple_sweep_save_path = os.path.join(vis_scene_save_path,
                                                         f"pred_aggregate_multiple_sweeps.png")
            pred_segment = torch.argmax(pred, dim=1)
            segment = torch.from_numpy(segment).to(pred.device)

            # Save final prediction visualization with multiple sweeps
            self.visualize_pointcloud_prediction(coords=GT_coord, segment_pred=pred_segment,
                                                 class_names=class_names,
                                                 title="Predicted Semantic Segmentation "
                                                       "(Top-down view, multiple sweeps)",
                                                 save_path=pred_multiple_sweep_save_path)
            print("Saved prediction visualization with multiple sweeps")

            # Save final GT with multiple sweeps
            self.visualize_pointcloud_prediction(coords=GT_coord, segment_pred=segment, class_names=class_names,
                                                 title="GT Semantic Segmentation (Top-down view, multiple sweeps)",
                                                 save_path=GT_save_path)
            print("Saved GT visualization with multiple sweeps")
            # torch.save((segment, GT_coord), GT_save_path)
            # print(f"Saved (GT_segment, GT_coordinates) information")

            processed += 1
            if processed >= len(scene_ids):
                break  # Done with all desired scenes

    def evaluate_performance_by_point_cloud_size(self):
        assert self.test_loader.batch_size == 1, "Batch size of test data should be 1"
        logger = get_root_logger()
        custom_logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()

        vis_save_path = '/home/luutunghai@gmail.com/projects/PTv3-distill/visualization/'
        self.model.eval()

        # Base sizes, excluding the last one which will be dynamic
        base_point_cloud_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384]
        results = {size: {"fps": [], "memory": [], "peak_memory": []} for size in base_point_cloud_sizes}

        is_TTA_activated = len(self.cfg.data.test.test_cfg.aug_transform) > 2
        if is_TTA_activated:
            custom_logger.info("TTA augmentation is activated for evaluation")

        for idx, data_dict in enumerate(self.test_loader):

            if idx == 0:
                continue

            if idx == 2:
                break

            data_dict = data_dict[0]  # Batch size 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")

            original_size = fragment_list[0]["coord"].shape[0]
            custom_logger.info(f"Original point cloud size: {original_size}")

            # Dynamically adjust point_cloud_sizes: base sizes + original_size
            point_cloud_sizes = base_point_cloud_sizes.copy()
            if original_size > base_point_cloud_sizes[-1]:  # If original_size > 16384
                point_cloud_sizes.append(original_size)
            elif original_size > base_point_cloud_sizes[0]:  # If between 256 and 16384
                # Replace the largest base size less than original_size with original_size
                for i, size in enumerate(base_point_cloud_sizes):
                    if size >= original_size:
                        point_cloud_sizes[i - 1] = original_size
                        point_cloud_sizes = point_cloud_sizes[:i]  # Truncate after original_size
                        break
            else:  # If original_size <= 256, just use it as the smallest
                point_cloud_sizes = [original_size]

            # Update results dict with dynamic sizes for this batch
            for size in point_cloud_sizes:
                if size not in results:
                    results[size] = {"fps": [], "memory": [], "peak_memory": []}

            for target_size in point_cloud_sizes:
                custom_logger.info(f"Testing point cloud size: {target_size}")
                custom_logger.info(f" ")
                if target_size < original_size:
                    coords = fragment_list[0]["coord"]
                    if isinstance(coords, np.ndarray):
                        coords = torch.from_numpy(coords).float().cuda()
                    subsample_idx = torch_fps(coords, ratio=target_size / original_size, random_start=False)
                    subsample_idx = subsample_idx.cpu().numpy()
                    subsampled_size = len(subsample_idx)

                    subsampled_fragments = []
                    for frag in fragment_list:
                        subsampled_frag = {}
                        for k, v in frag.items():
                            if isinstance(v, (torch.Tensor, np.ndarray)) and k in ["coord", "feat", "index",
                                                                                   "grid_coord"]:
                                subsampled_frag[k] = v[subsample_idx] if isinstance(v, np.ndarray) else v[subsample_idx]
                            elif k == "offset":
                                subsampled_frag[k] = torch.tensor([subsampled_size], dtype=torch.int32)
                            else:
                                subsampled_frag[k] = v
                        subsampled_fragments.append(subsampled_frag)
                else:
                    custom_logger.info(f"Target size matches or exceeds original size: {original_size}, using original")
                    subsampled_fragments = fragment_list
                    subsampled_size = original_size

                pred = torch.zeros((subsampled_size, self.cfg.data.num_classes), dtype=torch.float32).cuda()
                total_fps = []

                torch.cuda.reset_peak_memory_stats()

                custom_logger.info(f"Making prediction on {len(subsampled_fragments)} frames")
                for i in range(len(subsampled_fragments)):
                    s_i, e_i = i, min(i + 1, len(subsampled_fragments))
                    input_dict = collate_fn(subsampled_fragments[s_i:e_i])

                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    idx_part = input_dict["index"]

                    inference_start = time.time()
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        pred_part = self.model(input_dict)
                        if isinstance(pred_part, tuple):
                            pred_part = pred_part[0]
                        pred_part = F.softmax(pred_part, dim=-1)
                    torch.cuda.synchronize()
                    inference_time = time.time() - inference_start

                    fps_value = 1 / inference_time
                    total_fps.append(fps_value)

                    memory_used = torch.cuda.memory_allocated() / 1024 ** 2
                    peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 2

                    # custom_logger.info(
                    #     f"Size: {target_size}, FPS: {fps_value:.4f}, Memory: {memory_used:.2f} MB, Peak Memory: {peak_memory:.2f} MB"
                    # )

                    results[target_size]["fps"].append(fps_value)
                    results[target_size]["memory"].append(memory_used)
                    results[target_size]["peak_memory"].append(peak_memory)

                    # # Aggregate predictions
                    # bs = 0
                    # for be in input_dict["offset"]:
                    #     pred[idx_part[bs:be], :] += pred_part[bs:be]
                    #     bs = be

                    # if self.cfg.empty_cache:
                    #     torch.cuda.empty_cache()

                # avg_fps = sum(total_fps) / len(total_fps)
                # avg_memory = sum(results[target_size]["memory"]) / len(results[target_size]["memory"])
                # avg_flops = sum(results[target_size]["flops"]) / len(results[target_size]["flops"])
                # custom_logger.info(
                #     f"Size: {target_size}, Avg FPS: {avg_fps:.2f}, Avg Memory: {avg_memory:.2f} MB") # , Avg FLOPs: {avg_flops:.2f} GFLOPs

            # pred = torch.argmax(pred, dim=1)
            # segment = torch.from_numpy(segment).to(pred.device)
            # intersection, union, target = intersection_and_union_gpu(pred, segment, self.cfg.data.num_classes,
            #                                                          self.cfg.data.ignore_index)
            # intersection_meter.update(intersection)
            # union_meter.update(union)
            # target_meter.update(target)

        # custom_logger.info(">>>>>>>>>>>>>>>> Summary >>>>>>>>>>>>>>>>")
        # for size in point_cloud_sizes:
        #     if results[size]["fps"]:  # Avoid empty lists
        #         avg_fps = np.mean(results[size]["fps"])
        #         avg_memory = np.mean(results[size]["memory"])
        #         avg_peak_memory = np.mean(results[size]["peak_memory"])
        #         custom_logger.info(
        #             f"Size: {size}, Avg FPS: {avg_fps:.2f}, Avg Memory: {avg_memory:.2f} MB, Avg Peak Memory: {avg_peak_memory:.2f} MB"
        #         )
        #     else:
        #         custom_logger.info(f"Size: {size}, No data collected")
        #
        # custom_logger.info("<<<<<<<<<<<<< End summary >>>>>>>>>>>>>>>>")
        custom_logger.info(" ")

    @staticmethod
    def collate_fn(batch):
        return batch

    def visualize_pointcloud_prediction(self, coords, segment_pred, class_names,
                                        title="Ground Truth (Top-down view)", save_path=None):
        """
        Visualize predicted point cloud segmentation with class name legends.

        Args:
            coords: [N, 3] point coordinates.
            segment_pred: [N,] predicted labels.
            class_names (List[str]): Class names by index.
            save_path (str): Optional path to save figure.
        """

        if not isinstance(coords, np.ndarray):
            coords = coords.cpu().numpy()
        if not isinstance(segment_pred, np.ndarray):
            segment_pred = segment_pred.cpu().numpy()  # Convert tensor to numpy if needed

        colors = get_label_colors(segment_pred)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)

        ax.set_axis_off()
        plt.axis('off')

        add_legend(ax, segment_pred, class_names=class_names)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        else:
            plt.show()






