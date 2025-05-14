import torch
# from torch.utils.tensorboard import SummaryWriter
from dataloader import PTv3_Dataloader
from PTv3_model import PointTransformerV3TrainTeacher, load_weights_ptv3_nucscenes_seg
from pointcept.engines.defaults import default_config_parser, default_setup
from pointcept.models.losses import build_criteria
import os
from gpu_main import compute_miou, compute_iou_all_classes
from pointcept.engines import test
import numpy as np
import subprocess
from loguru import logger as custom_logger
import sys
from fvcore.nn import FlopCountAnalysis
from torch.profiler import profile, record_function, ProfilerActivity
from PTv3_model_gflops import PointTransformerV3TrainTeacher as PTv3_GFLOPs

# Pretrained model path and config file
PRETRAINED_PATH = './checkpoints/model_last.pth'
# PRETRAINED_PATH = './checkpoints/checkpoint_epoch_1.pth'
CONFIG_FILE = "configs/nuscenes/semseg-pt-v3m1-0-train-teacher.py"

# Remove default sink (optional)
custom_logger.remove()


# # Save the logs in loguru INFO
# log_path = "./logs/various-point-size_GFLOPs_log_student.txt"
# os.makedirs(os.path.dirname(log_path), exist_ok=True)
#
# # Add a file sink and a console sink
# custom_logger.add(log_path, mode="w", rotation="10 MB", enqueue=True, encoding="utf-8",
#                   format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}")
# custom_logger.add(sys.stderr, format="{time} | {level} | {message}")


def get_teacher_model(cfg):
    """
    Load the teacher model and load the pretrained weights.

    Returns:
        PointTransformerV3: The teacher model.
    """
    model_config = cfg.model.backbone
    return PointTransformerV3TrainTeacher(
        in_channels=model_config.in_channels,
        pdnorm_conditions=model_config.pdnorm_conditions,
        cls_mode=model_config.cls_mode,
        pdnorm_bn=model_config.pdnorm_bn,
        mlp_ratio=model_config.mlp_ratio,
        qkv_bias=model_config.qkv_bias,
        enable_flash=model_config.enable_flash,
        order=model_config.order,
        stride=model_config.stride,
        enc_depths=model_config.enc_depths,
        enc_channels=model_config.enc_channels,
        enc_num_head=model_config.enc_num_head,
        enc_patch_size=model_config.enc_patch_size,
        dec_depths=model_config.dec_depths,
        dec_channels=model_config.dec_channels,
        dec_num_head=model_config.dec_num_head,
        dec_patch_size=model_config.dec_patch_size,
        qk_scale=model_config.qk_scale,
        attn_drop=model_config.attn_drop,
        proj_drop=model_config.proj_drop,
        drop_path=model_config.drop_path,
        shuffle_orders=model_config.shuffle_orders,
        pre_norm=model_config.pre_norm,
        enable_rpe=model_config.enable_rpe,
        upcast_attention=model_config.upcast_attention,
        upcast_softmax=model_config.upcast_softmax,
        pdnorm_ln=model_config.pdnorm_ln,
        pdnorm_decouple=model_config.pdnorm_decouple,
        pdnorm_adaptive=model_config.pdnorm_adaptive,
        pdnorm_affine=model_config.pdnorm_affine
    )


def get_teacher_model_GFLOPs(cfg):
    """
    Load the teacher model and load the pretrained weights.

    Returns:
        PointTransformerV3: The teacher model.
    """
    model_config = cfg.model.backbone
    return PTv3_GFLOPs(
        in_channels=model_config.in_channels,
        pdnorm_conditions=model_config.pdnorm_conditions,
        cls_mode=model_config.cls_mode,
        pdnorm_bn=model_config.pdnorm_bn,
        mlp_ratio=model_config.mlp_ratio,
        qkv_bias=model_config.qkv_bias,
        enable_flash=model_config.enable_flash,
        order=model_config.order,
        stride=model_config.stride,
        enc_depths=model_config.enc_depths,
        enc_channels=model_config.enc_channels,
        enc_num_head=model_config.enc_num_head,
        enc_patch_size=model_config.enc_patch_size,
        dec_depths=model_config.dec_depths,
        dec_channels=model_config.dec_channels,
        dec_num_head=model_config.dec_num_head,
        dec_patch_size=model_config.dec_patch_size,
        qk_scale=model_config.qk_scale,
        attn_drop=model_config.attn_drop,
        proj_drop=model_config.proj_drop,
        drop_path=model_config.drop_path,
        shuffle_orders=model_config.shuffle_orders,
        pre_norm=model_config.pre_norm,
        enable_rpe=model_config.enable_rpe,
        upcast_attention=model_config.upcast_attention,
        upcast_softmax=model_config.upcast_softmax,
        pdnorm_ln=model_config.pdnorm_ln,
        pdnorm_decouple=model_config.pdnorm_decouple,
        pdnorm_adaptive=model_config.pdnorm_adaptive,
        pdnorm_affine=model_config.pdnorm_affine
    )


def get_student_model(cfg):
    """
    Create a student model with the same architecture as the teacher model.

    Returns:
        PointTransformerV3: The student model.
    """
    model_config = cfg.model.backbone
    return PointTransformerV3TrainTeacher(
        in_channels=model_config.in_channels,
        pdnorm_conditions=model_config.pdnorm_conditions,
        cls_mode=model_config.cls_mode,
        pdnorm_bn=model_config.pdnorm_bn,
        mlp_ratio=model_config.mlp_ratio,
        qkv_bias=model_config.qkv_bias,
        enable_flash=model_config.enable_flash,
        order=model_config.order,
        stride=model_config.stride,
        enc_depths=model_config.enc_depths,  # Reduced depth
        enc_channels=model_config.enc_channels,  # Reduced channels
        enc_num_head=model_config.enc_num_head,  # Reduced heads
        enc_patch_size=model_config.enc_patch_size,  # Smaller patches
        dec_depths=model_config.dec_depths,  # Reduced decoder depth
        dec_channels=model_config.dec_channels,  # Reduced decoder channels
        dec_num_head=model_config.dec_num_head,  # Reduced decoder heads
        dec_patch_size=model_config.dec_patch_size,  # Smaller decoder patches
        qk_scale=model_config.qk_scale,
        attn_drop=model_config.attn_drop,
        proj_drop=model_config.proj_drop,
        drop_path=model_config.drop_path,
        shuffle_orders=model_config.shuffle_orders,
        pre_norm=model_config.pre_norm,
        enable_rpe=model_config.enable_rpe,
        upcast_attention=model_config.upcast_attention,
        upcast_softmax=model_config.upcast_softmax,
        pdnorm_ln=model_config.pdnorm_ln,
        pdnorm_decouple=model_config.pdnorm_decouple,
        pdnorm_adaptive=model_config.pdnorm_adaptive,
        pdnorm_affine=model_config.pdnorm_affine
    )


def get_student_model_GFLOPs(cfg):
    """
    Load the teacher model and load the pretrained weights.

    Returns:
        PointTransformerV3: The teacher model.
    """
    model_config = cfg.model.backbone
    return PTv3_GFLOPs(
        in_channels=model_config.in_channels,
        pdnorm_conditions=model_config.pdnorm_conditions,
        cls_mode=model_config.cls_mode,
        pdnorm_bn=model_config.pdnorm_bn,
        mlp_ratio=model_config.mlp_ratio,
        qkv_bias=model_config.qkv_bias,
        enable_flash=model_config.enable_flash,
        order=model_config.order,
        stride=model_config.stride,
        enc_depths=model_config.enc_depths,
        enc_channels=model_config.enc_channels,
        enc_num_head=model_config.enc_num_head,
        enc_patch_size=model_config.enc_patch_size,
        dec_depths=model_config.dec_depths,
        dec_channels=model_config.dec_channels,
        dec_num_head=model_config.dec_num_head,
        dec_patch_size=model_config.dec_patch_size,
        qk_scale=model_config.qk_scale,
        attn_drop=model_config.attn_drop,
        proj_drop=model_config.proj_drop,
        drop_path=model_config.drop_path,
        shuffle_orders=model_config.shuffle_orders,
        pre_norm=model_config.pre_norm,
        enable_rpe=model_config.enable_rpe,
        upcast_attention=model_config.upcast_attention,
        upcast_softmax=model_config.upcast_softmax,
        pdnorm_ln=model_config.pdnorm_ln,
        pdnorm_decouple=model_config.pdnorm_decouple,
        pdnorm_adaptive=model_config.pdnorm_adaptive,
        pdnorm_affine=model_config.pdnorm_affine
    )


def train_student(log_file="test_one_frame_student.txt"):
    CONFIG_FILE_STUDENT = "configs/nuscenes/semseg-pt-v3m1-0-train-student.py"
    PRETRAINED_PATH_STUDENT = './checkpoints/checkpoint_batch_40001.pth'
    # Load configuration
    cfg = default_config_parser(CONFIG_FILE_STUDENT, None)
    cfg = default_setup(cfg)

    cfg.model.backbone.enable_flash = False
    print("--------> Enable Flash is manually turns off")

    student_model = get_student_model(cfg)
    # student_model = get_student_model_GFLOPs(cfg)

    student_model = load_weights_ptv3_nucscenes_seg(student_model, PRETRAINED_PATH_STUDENT)

    # Data load
    loader = PTv3_Dataloader(cfg)
    train_loader = loader.load_training_data()

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = student_model.to(device)

    count_parameters(model)

    # Loss setup
    criteria = build_criteria(cfg.model.criteria)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    max_lr = cfg.scheduler.max_lr[0]  # Single value for max_lr

    # Scheduler setup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        pct_start=cfg.scheduler.pct_start,
        anneal_strategy=cfg.scheduler.anneal_strategy,
        div_factor=cfg.scheduler.div_factor,
        final_div_factor=cfg.scheduler.final_div_factor,
        total_steps=len(train_loader) * cfg.epoch,
    )

    # Training loop
    num_epochs = cfg.epoch
    os.makedirs('checkpoints', exist_ok=True)  # Ensure checkpoint directory exists

    num_classes = len(cfg.names)  # Assuming cfg.names contains class names

    tester = test.CustomSemSegTester(cfg=cfg, model=model)
    tester.test_one_frame()

    # for epoch in range(num_epochs):
    #     running_loss = 0.0
    #
    #     total_train_iou_scores = torch.zeros(num_classes, device=device)
    #
    #     # train
    #     model.train()
    #
    #     print("-----> Training")
    #     for batch_ndx, input_dict in enumerate(train_loader):
    #
    #         if batch_ndx == 1:
    #             print(">>>>>>>>>> Interrupted by setting stop batch == 1 <<<<<<<<<<<<")
    #             break
    #
    #         if batch_ndx > 40000:
    #             print("-------> Enough training, now evaluation")
    #             break
    #
    #         # Move input data to device
    #         input_dict = {k: v.to(device) for k, v in input_dict.items()}
    #         # coord_shape = input_dict['coord'].shape
    #         # segment_shape = input_dict['segment'].shape
    #         # feat_shape = input_dict['feat'].shape
    #         # with open("./debug/input_dict_train.txt", "w") as f:
    #         #     f.write("input_dict:\n")
    #         #     f.write(str(input_dict))
    #         #     f.write("\n\nShapes:\n")
    #         #     f.write(f"coord shape: {coord_shape}\n")
    #         #     f.write(f"segment shape: {segment_shape}\n")
    #         #     f.write(f"feat shape: {feat_shape}\n")
    #         # print("Done")
    #
    #         # Forward pass
    #         seg_logits = model(input_dict)
    #         # logits_tensor = seg_logits.get("feat", None)
    #         logits_tensor = seg_logits
    #
    #         # -----------------Start Profiling -----------------/
    #
    #         # Time in each layer
    #         activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]
    #         sort_by_keyword = str(device) + "_time_total"
    #
    #         with profile(activities=activities, record_shapes=True) as prof:
    #             with record_function("model_inference"):
    #                 model(input_dict)
    #
    #         print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))
    #
    #         # Memory in each layer
    #         with profile(activities=[ProfilerActivity.CPU],
    #                      profile_memory=True, record_shapes=True) as prof:
    #             model(input_dict)
    #
    #         print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    #         print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    #
    #         # -----------------End Profiling -----------------/
    #
    #         # Compute loss
    #         loss = criteria(logits_tensor, input_dict["segment"])
    #
    #         # Compute IoU
    #         ground_truth = input_dict["segment"]
    #
    #         with torch.no_grad():
    #             teacher_preds = torch.argmax(seg_logits, dim=1)
    #             teacher_iou_scores = compute_iou_all_classes(
    #                 teacher_preds,
    #                 ground_truth,
    #                 num_classes
    #             )
    #             teacher_miou = compute_miou(teacher_iou_scores)
    #             total_train_iou_scores += teacher_iou_scores
    #
    #         # Backward pass
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         # Accumulate loss
    #         running_loss += loss.item()
    #
    #         # Print progress every 10 batches
    #         if batch_ndx % 30 == 0:
    #             print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_ndx}/{len(train_loader)}], "
    #                   f"Loss: {loss.item():.2f}, IoU: {teacher_miou:.2f}")
    #
    #         # save model at batch:
    #         if batch_ndx % 10000 == 0:
    #             checkpoint_path = os.path.join('checkpoints', f'checkpoint_batch_{batch_ndx + 1}.pth')
    #             torch.save(model.state_dict(), checkpoint_path)
    #             print(f"Model checkpoint saved at batch {batch_ndx} in {checkpoint_path}")
    #
    #     # Step the scheduler
    #     scheduler.step()
    #
    #     # Print epoch loss and log to TensorBoard
    #     avg_loss = running_loss / len(train_loader)
    #     train_miou = compute_miou(total_train_iou_scores / len(train_loader))
    #     print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, mIoU: {train_miou:.4f}")
    #
    #     # Save checkpoints
    #     if (epoch + 1) % cfg.eval_epoch == 0 or (epoch + 1) == num_epochs:
    #         checkpoint_path = os.path.join('checkpoints', f'checkpoint_epoch_{epoch + 1}.pth')
    #         torch.save(model.state_dict(), checkpoint_path)
    #         print(f"Model checkpoint saved at {checkpoint_path}")
    #
    #     # ====================== TEST EVALUATION ======================
    #
    #     print("---------> Testing")
    #
    #     tester = test.CustomSemSegTester(cfg=cfg, model=model)
    #     tester.test()


def count_parameters(model):
    """Calculate total and trainable parameters in a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

    return total_params, trainable_params


def main():
    TEACHER_CONFIG_FILE = "configs/nuscenes/semseg-pt-v3m1-0-train-teacher.py"
    TEACHER_PRETRAINED_PATH = './checkpoints/checkpoint_batch_80001_grid-size-0.1.pth'

    # Load configuration
    cfg = default_config_parser(TEACHER_CONFIG_FILE, None)
    cfg = default_setup(cfg)

    # Initialize TensorBoard
    # writer = SummaryWriter(log_dir='logs/ptv3_training')
    grid_size = 0.1
    cfg.data.train.transform[4].grid_size = grid_size
    cfg.model.backbone.enable_flash = False
    cfg.batch_size = 2  # takes 79 Gb RAMs on A100
    cfg.batch_size_val = 2
    cfg.batch_size_test = 1

    # Save the logs in loguru INFO
    log_path = f"./logs/train_teacher_grid-size-{grid_size}.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Add a file sink and a console sink
    custom_logger.add(log_path, mode="a", rotation="10 MB", enqueue=True, encoding="utf-8",
                      format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}")
    custom_logger.add(sys.stderr, format="{time} | {level} | {message}")

    print(f"------> Grid Size = {grid_size} - Enable Flash = {cfg.model.backbone.enable_flash}")
    # Load teacher model
    teacher_model = get_teacher_model(cfg)
    # teacher_model = get_teacher_model_GFLOPs(cfg)

    # Load pretrained weights
    model = load_weights_ptv3_nucscenes_seg(teacher_model, TEACHER_PRETRAINED_PATH)

    # Data load
    loader = PTv3_Dataloader(cfg)
    train_loader = loader.load_training_data()
    # test_loader = loader.load_validation_data()

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    count_parameters(model)

    # Loss setup
    criteria = build_criteria(cfg.model.criteria)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    max_lr = cfg.scheduler.max_lr[0]  # Single value for max_lr

    # Scheduler setup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        pct_start=cfg.scheduler.pct_start,
        anneal_strategy=cfg.scheduler.anneal_strategy,
        div_factor=cfg.scheduler.div_factor,
        final_div_factor=cfg.scheduler.final_div_factor,
        total_steps=len(train_loader) * cfg.epoch,
    )

    # Training loop
    num_epochs = cfg.epoch
    os.makedirs('checkpoints', exist_ok=True)  # Ensure checkpoint directory exists

    num_classes = len(cfg.names)  # Assuming cfg.names contains class names

    # tester = test.CustomSemSegTester(cfg=cfg, model=model)  # , model=model
    # test_loader = tester.evaluate_performance_by_point_cloud_size()

    for epoch in range(num_epochs):
        running_loss = 0.0

        total_train_iou_scores = torch.zeros(num_classes, device=device)

        # train
        model.train()

        print("-----> Training")
        for batch_ndx, input_dict in enumerate(train_loader):

            # if batch_ndx == 1:
            #     print(">>>>>>>>>> Interrupted by setting stop batch == 1 <<<<<<<<<<<<")
            #     break

            # Move input data to device
            input_dict = {k: v.to(device) for k, v in input_dict.items()}

            # Forward pass
            seg_logits = model(input_dict)
            # logits_tensor = seg_logits.get("feat", None)

            # # -----------------Start Profiling -----------------/
            #
            # # Time in each layer
            # activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]
            # sort_by_keyword = str(device) + "_time_total"
            #
            # with profile(activities=activities, record_shapes=True) as prof:
            #     with record_function("model_inference"):
            #         model(input_dict)
            #
            # print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))
            #
            # # Memory in each layer
            # with profile(activities=[ProfilerActivity.CPU],
            #              profile_memory=True, record_shapes=True) as prof:
            #     model(input_dict)
            #
            # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
            # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
            #
            # # -----------------End Profiling -----------------/

            logits_tensor = seg_logits

            # Compute loss
            loss = criteria(logits_tensor, input_dict["segment"])

            # Compute IoU
            ground_truth = input_dict["segment"]

            # print(logits_tensor.shape)
            # print(ground_truth.shape)
            with torch.no_grad():
                teacher_preds = torch.argmax(seg_logits, dim=1)
                teacher_iou_scores = compute_iou_all_classes(
                    teacher_preds,
                    ground_truth,
                    num_classes
                )
                teacher_miou = compute_miou(teacher_iou_scores)
                total_train_iou_scores += teacher_iou_scores

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Log loss to TensorBoard
            # writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_ndx)

            # Print progress every 10 batches
            if batch_ndx % 100 == 0:
                # print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_ndx}/{len(train_loader)}], "
                #       f"Loss: {loss.item():.2f}, IoU: {teacher_miou:.2f}")

                custom_logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_ndx}/{len(train_loader)}], "
                                   f"Loss: {loss.item():.2f}, IoU: {teacher_miou:.2f}")

            if batch_ndx % 5000 == 0 and batch_ndx != 0:
                checkpoint_path = os.path.join('checkpoints',
                                               f'checkpoint_batch_{batch_ndx + 1}_grid-size-{grid_size}.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Model checkpoint saved at {checkpoint_path}")

        # Step the scheduler
        scheduler.step()

        # Print epoch loss and log to TensorBoard
        avg_loss = running_loss / len(train_loader)
        train_miou = compute_miou(total_train_iou_scores / len(train_loader))
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, mIoU: {train_miou:.4f}")
        # writer.add_scalar('Loss/epoch_avg', avg_loss, epoch + 1)

        # Save checkpoints
        if (epoch + 1) % cfg.eval_epoch == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = os.path.join('checkpoints', f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

        # ====================== TEST EVALUATION ======================

        print("---------> Testing")

        tester = test.CustomSemSegTester(cfg=cfg, model=model)
        tester.test()

    # # writer.close()


if __name__ == "__main__":
    # set up the GPU
    # subprocess.run(["bash", "run_set_up_GPU.sh"], check=True)

    main()
    # train_student()  # Val result: mIoU/mAcc/allAcc 0.7608/0.8255/0.9389
    # python main.py

    """
    teacher params: Total Parameters: 46160080
    Trainable Parameters: 46160080
    GPU used: 4.3 Gb
    
    student params: Total Parameters: 2778112
    Trainable Parameters: 2778112
    GPU used: 4.2 Gb
    """
    """
    [2025-03-25 11:13:21,726 INFO test.py line 1047 13120] >>>>>>>>>>>>>>>> Summary >>>>>>>>>>>>>>>>
    [2025-03-25 11:13:21,726 INFO test.py line 1052 13120] Size: 256, Avg FPS: 26.89, Avg Memory: 18.83 MB
    [2025-03-25 11:13:21,726 INFO test.py line 1052 13120] Size: 512, Avg FPS: 30.33, Avg Memory: 18.87 MB
    [2025-03-25 11:13:21,726 INFO test.py line 1052 13120] Size: 1024, Avg FPS: 30.44, Avg Memory: 18.96 MB
    [2025-03-25 11:13:21,726 INFO test.py line 1052 13120] Size: 2048, Avg FPS: 30.70, Avg Memory: 19.15 MB
    [2025-03-25 11:13:21,726 INFO test.py line 1052 13120] Size: 4096, Avg FPS: 29.29, Avg Memory: 19.51 MB
    [2025-03-25 11:13:21,726 INFO test.py line 1052 13120] Size: 8192, Avg FPS: 28.61, Avg Memory: 20.25 MB
    [2025-03-25 11:13:21,726 INFO test.py line 1052 13120] Size: 16384, Avg FPS: 25.59, Avg Memory: 21.72 MB
    [2025-03-25 11:13:21,726 INFO test.py line 1052 13120] Size: 22000, Avg FPS: 28.11, Avg Memory: 23.06 MB
    
    [2025-03-25 11:16:06,935 INFO test.py line 1047 13893] >>>>>>>>>>>>>>>> Summary >>>>>>>>>>>>>>>>
    [2025-03-25 11:16:06,935 INFO test.py line 1052 13893] Size: 256, Avg FPS: 19.51, Avg Memory: 187.82 MB
    [2025-03-25 11:16:06,935 INFO test.py line 1052 13893] Size: 512, Avg FPS: 21.99, Avg Memory: 187.86 MB
    [2025-03-25 11:16:06,935 INFO test.py line 1052 13893] Size: 1024, Avg FPS: 21.90, Avg Memory: 187.96 MB
    [2025-03-25 11:16:06,935 INFO test.py line 1052 13893] Size: 2048, Avg FPS: 21.58, Avg Memory: 188.14 MB
    [2025-03-25 11:16:06,935 INFO test.py line 1052 13893] Size: 4096, Avg FPS: 19.96, Avg Memory: 188.51 MB
    [2025-03-25 11:16:06,935 INFO test.py line 1052 13893] Size: 8192, Avg FPS: 17.71, Avg Memory: 189.24 MB
    [2025-03-25 11:16:06,935 INFO test.py line 1052 13893] Size: 16384, Avg FPS: 16.34, Avg Memory: 190.71 MB
    [2025-03-25 11:16:06,936 INFO test.py line 1052 13893] Size: 22000, Avg FPS: 16.88, Avg Memory: 191.72 MB
    """
