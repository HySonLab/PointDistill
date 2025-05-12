weight = None
resume = False
evaluate = True
test_only = False
seed = 28024989
save_path = 'exp/nuscenes/semseg-pt-v3m1-0-train-student'
miou_result_path = f"/home/.../projects/PTv3-distill/{save_path}/miou_result"

num_worker = 1
batch_size = 5  # takes 79 Gb RAMs on A100
batch_size_val = 18
batch_size_test = 1
epoch = 50
eval_epoch = 1
sync_bn = False
enable_amp = True
empty_cache = False
find_unused_parameters = False
mix_prob = 0.8
param_dicts = [dict(keyword='block', lr=0.0002)]
hooks = [
    dict(type='CheckpointLoader'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='SemSegEvaluator'),
    dict(type='CheckpointSaver', save_freq=1),
    dict(type='PreciseEvaluator', test_last=False, keywords="module.", replacement="")
]
train = dict(type='DefaultTrainer')
test = dict(type='SemSegTester', verbose=True)
model = dict(
    type='DefaultSegmentorV2',
    num_classes=16,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=4,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(1, 1, 1, 2, 1),  # 20% of the teacher's depths
        enc_channels=(16, 16, 32, 64, 128),  # Reduced channels
        enc_num_head=(1, 1, 2, 4, 8),  # Reduced number of attention heads
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),  # (1024, 1024, 1024, 1024, 1024)
        dec_depths=(1, 1, 1, 1),  # Reduced decoder depths
        dec_channels=(64, 64, 128, 128),
        dec_num_head=(2, 2, 4, 8),  # Reduced decoder attention heads
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.15,
        shuffle_orders=True,
        pre_norm=True,
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
        pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo")),
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1)
    ])
optimizer = dict(type='AdamW', lr=0.002, weight_decay=0.005)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.002, 0.0002],
    pct_start=0.04,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=100.0)
dataset_type = 'NuScenesDataset'
data_root = '/data/user/luutunghai@gmail.com/dataset/sets/nuscenes'
ignore_index = -1
names = [
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
    'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface',
    'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation'
]


turn_on_TTA_testing = False

TTA_augmentations = [
                [{
                    'type': 'RandomScale',
                    'scale': [0.9, 0.9]
                }], [{
                    'type': 'RandomScale',
                    'scale': [0.95, 0.95]
                }], [{
                    'type': 'RandomScale',
                    'scale': [1, 1]
                }], [{
                    'type': 'RandomScale',
                    'scale': [1.05, 1.05]
                }], [{
                    'type': 'RandomScale',
                    'scale': [1.1, 1.1]
                }],
                [{
                    'type': 'RandomScale',
                    'scale': [0.9, 0.9]
                }, {
                    'type': 'RandomFlip',
                    'p': 1
                }],
                [{
                    'type': 'RandomScale',
                    'scale': [0.95, 0.95]
                }, {
                    'type': 'RandomFlip',
                    'p': 1
                }],
                [{
                    'type': 'RandomScale',
                    'scale': [1, 1]
                }, {
                    'type': 'RandomFlip',
                    'p': 1
                }],
                [{
                    'type': 'RandomScale',
                    'scale': [1.05, 1.05]
                }, {
                    'type': 'RandomFlip',
                    'p': 1
                }],
                [{
                    'type': 'RandomScale',
                    'scale': [1.1, 1.1]
                }, {
                    'type': 'RandomFlip',
                    'p': 1
                }]
    ]

simple_augmentation = [
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
            ]


data = dict(
    num_classes=16,
    ignore_index=-1,
    names=[
        'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
        'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
        'vegetation'
    ],
    train=dict(
        type='NuScenesDataset',
        split='train',
        data_root=data_root,
        transform=[
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'strength', 'segment'),
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=('coord', 'strength'))
        ],
        test_mode=False,
        ignore_index=-1,
        loop=1),
    val=dict(
        type='NuScenesDataset',
        split='val',
        data_root=data_root,
        transform=[
            dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'strength', 'segment'),
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=('coord', 'strength'))
        ],
        test_mode=False,
        ignore_index=-1),
    test=dict(
        type='NuScenesDataset',
        split='val',
        data_root=data_root,
        transform=[
            dict(type='Copy', keys_dict=dict(segment='origin_segment')),
            dict(
                type='GridSample',
                grid_size=0.025,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'strength', 'segment'),
                return_inverse=True)
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='test',
                return_grid_coord=True,
                keys=('coord', 'strength')),
            crop=None,
            post_transform=[
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'grid_coord', 'index'),
                    feat_keys=('coord', 'strength'))
            ],
            aug_transform=TTA_augmentations if turn_on_TTA_testing else simple_augmentation),
        ignore_index=-1))

"""
Turn off TTA by:
data = dict(
    train = dict(...),
    val = dict(...),
    test = dict(
        ...,
        test_cfg = dict(
            ...,
            aug_transform = [
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
            ]
        )
    )
)
"""

""" Activate TTA
[{
                'type': 'RandomScale',
                'scale': [0.9, 0.9]
            }], [{
                'type': 'RandomScale',
                'scale': [0.95, 0.95]
            }], [{
                'type': 'RandomScale',
                'scale': [1, 1]
            }], [{
                'type': 'RandomScale',
                'scale': [1.05, 1.05]
            }], [{
                'type': 'RandomScale',
                'scale': [1.1, 1.1]
            }],
                           [{
                               'type': 'RandomScale',
                               'scale': [0.9, 0.9]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [0.95, 0.95]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [1, 1]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [1.05, 1.05]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [1.1, 1.1]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }]
"""