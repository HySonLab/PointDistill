from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import os
import numpy as np
from pointcept.engines.defaults import default_config_parser, default_setup
from gpu_main import load_weights_ptv3_nucscenes_seg
from preprocess_mini_nuscenes import GetMiniNusceneDataset
import matplotlib.pyplot as plt
from pointcept.utils.env import get_random_seed, set_seed
from pointcept.utils.config import Config
import pointcept.utils.comm as comm
import ast
import time
import torch.nn.functional as F
from dataloader import PTv3_Dataloader
from main import get_student_model, load_weights_ptv3_nucscenes_seg, get_teacher_model
from main import get_teacher_model
from pointcept.engines import test
import torch


def visualization():
    PRETRAINED_PATH = './checkpoints/thomas_model_best.pth'
    TEACHER_CONFIG_FILE = "configs/nuscenes/semseg-pt-v3m1-0-train-teacher.py"

    # Load configuration
    cfg = default_config_parser(TEACHER_CONFIG_FILE, None)
    cfg = default_setup(cfg)
    cfg.model.backbone.enable_flash = False
    cfg.turn_on_TTA_testing = False  # todo: set this to False doesnt turn off TTA in config file, fix later
    class_names = cfg.names

    # Load teacher model
    teacher_model = get_teacher_model(cfg)

    # Load pretrained weights
    model = load_weights_ptv3_nucscenes_seg(teacher_model, PRETRAINED_PATH)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    tester = test.CustomSemSegTester(cfg=cfg, model=model)
    tester.visualize_prediction_and_GT(is_TTA_on=cfg.turn_on_TTA_testing, names="Nuscenes_grid_0.05",
                                       class_names=class_names, scene_ids=[10])


def visualize_GT_pred_student_teacher():
    CONFIG_FILE_STUDENT = "configs/nuscenes/semseg-pt-v3m1-0-train-student.py"
    PRETRAINED_PATH_STUDENT = './checkpoints/checkpoint_batch_40001.pth'

    TEACHER_CONFIG_FILE = "configs/nuscenes/semseg-pt-v3m1-0-train-teacher.py"
    PRETRAINED_PATH_TEACHER = './checkpoints/thomas_model_best.pth'

    # Load configuration of teacher
    cfg_teacher = default_config_parser(TEACHER_CONFIG_FILE, None)
    cfg_teacher = default_setup(cfg_teacher)
    cfg_teacher.data.test.transform[1].grid_size = 0.05
    cfg_teacher.model.backbone.enable_flash = False
    cfg_teacher.batch_size = 5

    cfg_student = default_config_parser(CONFIG_FILE_STUDENT, None)
    cfg_student = default_setup(cfg_student)
    cfg_student.data.test.transform[1].grid_size = 0.05
    cfg_student.model.backbone.enable_flash = False
    cfg_student.batch_size = 5
    """
    On test dataset
    grid_size = 0.01 --> Segment shape =  (29844,) | ~ 22 fragments
    grid_size = 0.05 --> Segment shape =  (27815,) | 8 fragments
    grid_size = 0.1 --> Segment shape =  (17659,) | 1 fragment

    On train dataset
    grid_size = 0.01 --> Segment shape = (249670)  | 1 fragment
    grid_size = 0.05 --> Segment shape =  (249670) | 1 fragments
    grid_size = 0.1 --> Segment shape =  (249670) | 1 fragment
    """
    class_names = cfg_teacher.names

    teacher_model = get_teacher_model(cfg=cfg_teacher)
    student_model = get_student_model(cfg_student)

    teacher_model = load_weights_ptv3_nucscenes_seg(teacher_model, PRETRAINED_PATH_TEACHER)
    student_model = load_weights_ptv3_nucscenes_seg(student_model, PRETRAINED_PATH_STUDENT)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)

    # visualization function have separate the model used for prediction, so passing teacher_model here is only
    # for initializing the class tester
    tester = test.CustomSemSegTester(cfg=cfg_student, model=teacher_model)
    # test_loader = tester.build_test_loader()

    # Data loader
    # configs of student and teacher are the same for setting dataset, so either cfg_student or cfg_teacher are okay
    loader = PTv3_Dataloader(cfg_student)
    val_loader = loader.load_validation_data()

    teacher_model.eval()
    student_model.eval()
    for batch_ndx, input_dict in enumerate(val_loader):

        if batch_ndx == 0:
            continue

        if batch_ndx == 50:
            break

        # GT
        coord = input_dict['coord']
        segment = input_dict['segment']

        teacher_seg_pred_label = get_pred_time_inference(teacher_model, input_dict, model_type='teacher')
        student_seg_pred_label = get_pred_time_inference(student_model, input_dict, model_type='student')

        # Save paths
        data_type = 'val_data'
        base_path = f'./visualization/{data_type}'

        # Ensure the directory exists
        os.makedirs(base_path, exist_ok=True)

        # Define paths
        teacher_path = f'{base_path}/teacher_train_pred_batch-{cfg_teacher.batch_size}-{batch_ndx}.png'
        student_path = f'{base_path}/student_val_pred_batch-{cfg_student.batch_size}-{batch_ndx}.png'
        GT_path = f'{base_path}/GT_batch-{cfg_student.batch_size}-{batch_ndx}.png'

        tester.visualize_pointcloud_prediction(coords=coord, segment_pred=teacher_seg_pred_label,
                                               class_names=class_names,
                                               save_path=teacher_path,
                                               title="Teacher Prediction")

        tester.visualize_pointcloud_prediction(coords=coord, segment_pred=student_seg_pred_label,
                                               class_names=class_names,
                                               save_path=student_path,
                                               title="Student Prediction")

        tester.visualize_pointcloud_prediction(coords=coord, segment_pred=segment,
                                               class_names=class_names,
                                               save_path=GT_path,
                                               title="Ground Truth")


def get_pred_time_inference(model, input_dict, model_type='teacher'):

    # pred_start = time.time()
    seg_logits = model(input_dict)  # --> 16Gb VRAM is not enough if not splitting into chunks
    # pred_end_time = time.time() - pred_start

    seg_pred_softmax = F.softmax(seg_logits, -1)

    seg_pred_label = torch.argmax(seg_pred_softmax, dim=-1)  # Shape: (24150,)

    # print(f"Inference {model_type} time = {pred_end_time:.2f} s")

    return seg_pred_label


if __name__ == "__main__":
    visualize_GT_pred_student_teacher()