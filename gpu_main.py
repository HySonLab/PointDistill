import torch
# from torch.utils.tensorboard import SummaryWriter
from dataloader import PTv3_Dataloader
from PTv3_model import PointTransformerV3, load_weights_ptv3_nucscenes_seg
from pointcept.engines.defaults import default_config_parser, default_setup
from pointcept.models.losses import build_criteria
import os
from pytorch3d.loss import chamfer_distance
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.sparse import csr_matrix
from gtda.homology import VietorisRipsPersistence
from datetime import datetime
from loguru import logger
from pointcept.engines import test

# Visualize Topological features of point cloud:
# https://giotto-ai.github.io/gtda-docs/latest/notebooks/mapper_quickstart.html#generate-and-visualise-data

# Pretrained model path and config file
PRETRAINED_PATH = './checkpoints/model_best.pth' #'./checkpoints/checkpoint_epoch_50_backup.pth'
CONFIG_FILE = "configs/nuscenes/semseg-pt-v3m1-0-base.py"


def setup_logging(log_dir='logs', file_name='training_gkd_activated_only'):
    """
    Set up Loguru logging to save print information to a file and display in console.
    Args:
        log_dir: Directory to save log files.
        :param file_name: file name of the log file
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{file_name}.txt')

    # Remove default handler and configure Loguru
    logger.remove()  # Remove default console handler
    logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")
    logger.add(lambda msg: print(msg, end=""), format="{message}", level="INFO")  # Console output

    return logger


# def compute_iou_all_classes(preds, labels, num_classes):
#     """
#     Compute IoU for all classes using GPU operations.
#
#     Args:
#         preds (torch.Tensor): Predicted labels of shape [N].
#         labels (torch.Tensor): Ground truth labels of shape [N].
#         num_classes (int): Number of classes.
#
#     Returns:
#         torch.Tensor: IoU scores for each class.
#     """
#     preds = preds.view(-1)
#     labels = labels.view(-1)
#
#     iou_scores = torch.zeros(num_classes, device=preds.device)
#
#     for class_id in range(num_classes):
#         pred_inds = preds == class_id
#         target_inds = labels == class_id
#
#         intersection = (pred_inds & target_inds).sum().float()
#         union = (pred_inds | target_inds).sum().float()
#
#         if union > 0:
#             iou_scores[class_id] = intersection / union
#         else:
#             iou_scores[class_id] = float('nan')  # Use NaN for classes not present
#
#     return iou_scores


def compute_iou_all_classes(preds, labels, num_classes, ignore_index=-1):
    """
    Compute IoU for all classes using GPU operations.

    Args:
        preds (torch.Tensor): Predicted labels of shape [N].
        labels (torch.Tensor): Ground truth labels of shape [N].
        num_classes (int): Number of classes.
        ignore_index (int): Ignore label for invalid pixels.

    Returns:
        torch.Tensor: IoU scores for each class.
    """
    preds = preds.view(-1)
    labels = labels.view(-1)

    # Mask out ignored values
    valid_mask = labels != ignore_index
    preds = preds[valid_mask]
    labels = labels[valid_mask]

    # Compute intersection and union efficiently
    intersection = torch.zeros(num_classes, device=preds.device)
    union = torch.zeros(num_classes, device=preds.device)

    for class_id in range(num_classes):
        pred_inds = preds == class_id
        target_inds = labels == class_id

        intersection[class_id] = (pred_inds & target_inds).sum().float()
        union[class_id] = (pred_inds | target_inds).sum().float()

    # Ensure missing classes get IoU=0 instead of NaN
    iou_scores = torch.where(union > 0, intersection / (union + 1e-10), torch.tensor(0.0, device=preds.device))

    return iou_scores


def compute_miou(iou_scores):
    """
    Compute the mean Intersection over Union (mIoU) from the IoU scores.

    Args:
        iou_scores (torch.Tensor): IoU scores for each class.

    Returns:
        torch.Tensor: The mean IoU.
    """
    # Exclude NaN values which represent classes not present
    valid_iou = iou_scores[~iou_scores.isnan()]
    miou = valid_iou.mean() if valid_iou.numel() > 0 else torch.tensor(0.0, device=iou_scores.device)
    return miou


def get_teacher_model(cfg):
    """
    Load the teacher model and load the pretrained weights.

    Returns:
        PointTransformerV3: The teacher model.
    """
    model_config = cfg.model.teacher_backbone
    return PointTransformerV3(
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
    model_config = cfg.model.student_backbone
    return PointTransformerV3(
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


def compute_persistence_entropy(diagram, dimensions=[0, 1, 2], device='cuda'):
    """
    Compute differentiable persistence entropy features for each homology dimension.
    Args:
        diagram: torch.Tensor of shape (N, 3) with [birth, death, dimension]
        dimensions: List of homology dimensions to compute entropy for
        device: Device for computation
    Returns:
        torch.Tensor: Entropy features of shape (n_dimensions, 1)
    """
    entropy_features = []
    for dim in dimensions:
        mask = diagram[:, 2] == dim
        points = diagram[mask]
        if points.shape[0] == 0:
            entropy_features.append(torch.tensor(0.0, device=device))
            continue

        # Compute persistence (death - birth)
        persistence = points[:, 1] - points[:, 0]  # Shape: (N,)
        persistence = torch.clamp(persistence, min=0.0)  # Handle numerical errors

        # Normalize persistence to sum to 1 (like probabilities)
        total_persistence = torch.sum(persistence)
        if total_persistence > 0:
            p = persistence / (total_persistence + 1e-8)
            # Compute entropy: -sum(p * log(p))
            entropy = -torch.sum(p * torch.log(p + 1e-8))
        else:
            entropy = torch.tensor(0.0, device=device)

        entropy_features.append(entropy)

    return torch.stack(entropy_features).unsqueeze(-1)  # Shape: (n_dimensions, 1)


def normalize_diagram(diagram, max_death_cap=1e6):
    """
    Min-max normalize a persistence diagram's (birth, death) coordinates, handling inf values.
    Args:
        diagram: torch.Tensor of shape (N, 3) with [birth, death, dimension]
        max_death_cap: Finite value to replace inf death times
    Returns:
        torch.Tensor: Normalized tensor with (birth, death) in [0, 1], dimension unchanged
    """
    bd = diagram[:, :2]
    bd = torch.where(torch.isinf(bd), torch.tensor(max_death_cap, device=bd.device), bd)
    finite_mask = torch.isfinite(bd)
    if finite_mask.any():
        min_val = torch.min(bd[finite_mask])
        max_val = torch.max(bd[finite_mask])
        normalized_bd = (bd - min_val) / (max_val - min_val + 1e-8)
    else:
        normalized_bd = bd
    normalized_diagram = torch.cat([normalized_bd, diagram[:, 2:3]], dim=1)
    return normalized_diagram


def compute_chamfer_loss(persistence_diagram_1, persistence_diagram_2, dimensions=[0, 1, 2]):
    """
    Compute differentiable Chamfer loss on persistence entropy features across homology dimensions.
    Args:
        persistence_diagram_1: NumPy array of shape (n_features, 3) with [birth, death, dim]
        persistence_diagram_2: NumPy array of shape (n_features, 3) with [birth, death, dim]
        dimensions: List of homology dimensions to compute loss for
    Returns:
        torch.Tensor: Chamfer loss on entropy features
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pd1 = torch.from_numpy(persistence_diagram_1).float().to(device)
    pd2 = torch.from_numpy(persistence_diagram_2).float().to(device)

    pd1 = normalize_diagram(pd1)
    pd2 = normalize_diagram(pd2)

    # Compute entropy features
    entropy1 = compute_persistence_entropy(pd1, dimensions, device)
    entropy2 = compute_persistence_entropy(pd2, dimensions, device)

    # Treat entropy features as points for Chamfer loss
    points1 = entropy1.unsqueeze(0)  # Shape: [1, n_dimensions, 1]
    points2 = entropy2.unsqueeze(0)  # Shape: [1, n_dimensions, 1]

    if torch.any(torch.isnan(points1)) or torch.any(torch.isinf(points1)) or \
       torch.any(torch.isnan(points2)) or torch.any(torch.isinf(points2)):
        # logger.warning("NaN or inf in entropy features; returning 0 loss")
        return torch.tensor(0.0, device=device, requires_grad=False)

    loss_chamfer, _ = chamfer_distance(points1, points2)
    if torch.isnan(loss_chamfer) or torch.isinf(loss_chamfer):
        # logger.warning("Chamfer loss is NaN or inf; returning 0 loss")
        return torch.tensor(0.0, device=device, requires_grad=False)

    # logger.info(f"Chamfer Loss on entropy features: {loss_chamfer.item():.4f}")
    return loss_chamfer


def compute_chamfer_loss_no_topo(y_pred, y_true):
    loss_chamfer, _ = chamfer_distance(y_pred, y_true)

    return loss_chamfer


def compute_ph_sparse_knn(features, k_neighbors=50, logger=None,
                          vectorRips_metric="precomputed",
                          knn_metric="euclidean"):
    """
    Compute Persistent Homology using a sparse k-NN distance matrix.
    Args:
        :param knn_metric: "euclidean", "cosine", ...
        :param features: torch.Tensor or np.ndarray, shape [n_points, n_features]
        :param k_neighbors: Number of nearest neighbors for sparse approximation
        :param logger: Loguru logger object
        :param vectorRips_metric: "precomputed", "euclidean", "manhattan" or "cosine"
    Returns:
        diagrams: NumPy array of persistence diagrams, shape (n_features, 3)

    """
    if isinstance(features, torch.Tensor):
        if logger:
            logger.info("Detaching tensor for giotto-tda compatibility")
        features = features.cpu().detach().numpy()

    n_points = features.shape[0]
    if n_points < k_neighbors:
        raise ValueError(f"n_points ({n_points}) must be >= k_neighbors ({k_neighbors})")

    nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric=knn_metric).fit(features)
    distances, indices = nbrs.kneighbors(features)

    row = np.repeat(np.arange(n_points), k_neighbors)
    col = indices.flatten()
    data = distances.flatten()
    sparse_dist = csr_matrix((data, (row, col)), shape=(n_points, n_points))
    sparse_dist = sparse_dist.maximum(sparse_dist.T)

    sparse_dist_list = [sparse_dist]
    ph = VietorisRipsPersistence(metric=vectorRips_metric, homology_dimensions=[0, 1, 2])
    diagrams = ph.fit_transform(sparse_dist_list)[0]  # Shape: (n_features, 3)
    if logger:
        logger.info(f"Computed PH diagrams with {diagrams.shape[0]} features")
    return diagrams


def main(use_gradient_guided=False, use_persistent_homology=False, normalize_gkd=True, norm_type='minmax'):
    # Load configuration
    cfg = default_config_parser(CONFIG_FILE, None)
    cfg = default_setup(cfg)

    # Initialize TensorBoard
    # writer = SummaryWriter(log_dir='logs/ptv3_training')
    # Set up logging
    logger = setup_logging(log_dir='./logs')

    # Load teacher model
    teacher_model = get_teacher_model(cfg)
    print("---> Loading teacher model")

    # Load student model
    student_model = get_student_model(cfg)
    print("---> Loading student model")

    # Load pretrained weights
    teacher_model = load_weights_ptv3_nucscenes_seg(teacher_model, PRETRAINED_PATH)

    # Data load
    loader = PTv3_Dataloader(cfg)
    train_loader = loader.load_training_data()

    # Move models to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    # print("load model done")

    # Loss function
    detection_loss_fn = build_criteria(cfg.model.criteria)

    # KD loss
    kld_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')  # KLD loss

    # lambda_param = cfg.model.lambda_param  # Weight for distillation loss
    # print("Loss function done")

    # Optimizer for student model
    student_optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay
    )
    # print("student_optimizer done")
    # It's unclear why there's a teacher_optimizer since the teacher is in eval mode.
    # If you intend to train the teacher, uncomment and modify the following:
    # teacher_optimizer = torch.optim.AdamW(
    #     teacher_model.parameters(),
    #     lr=cfg.optimizer.lr,
    #     weight_decay=cfg.optimizer.weight_decay
    # )

    # Loss scaling parameters
    lambda_param = 10.0  # For GKD loss --> amplify the importance features
    beta_param = 0.3  # For persistent homology (Chamfer) loss
    gamma_param = 0.3  # For KLD loss

    # Training loop
    # teacher_model.eval()  # Freeze the teacher model
    student_model.train()
    print("--------> Training Start")

    num_epochs = cfg.epoch
    os.makedirs('checkpoints', exist_ok=True)

    num_classes = len(cfg.names)  # Assuming cfg.names contains class names

    for epoch in range(num_epochs):
        # print("loop")
        for batch_ndx, input_dict in enumerate(train_loader):
            # print("Loadding")
            # Move input data to device
            input_dict = {k: v.to(device, non_blocking=True) for k, v in input_dict.items()}
            # print(input_dict)
            ground_truth = input_dict["segment"]
            # print("Move input data to device done")

            # Forward pass through teacher model
            # with torch.no_grad(): #todo: torch.no_grad will disable GKD --> need to solve it
            teacher_seg_logits, teacher_latent_feature = teacher_model(input_dict) # teacher_latent_feature (N, 512)
            teacher_preds = torch.argmax(teacher_seg_logits, dim=1)
            teacher_iou_scores = compute_iou_all_classes(
                teacher_preds,
                ground_truth,
                num_classes
            )
            teacher_miou = compute_miou(teacher_iou_scores)
            # print("teacher model move done")

            # # Forward pass: Student model
            student_seg_logits, student_latent_feature = student_model(input_dict) # student_latent_feature (N, 512)
            student_preds = torch.argmax(student_seg_logits, dim=1)
            student_iou_scores = compute_iou_all_classes(student_preds, ground_truth, num_classes)
            student_miou = compute_miou(student_iou_scores)
            # print("teacher model move done")

            # Compute detection loss for the teacher model (optional, since teacher is frozen)
            # If the teacher is frozen, this loss is not used for backpropagation / added regularization term
            # Compute segmentation losses
            teacher_loss = detection_loss_fn(teacher_seg_logits, ground_truth) + torch.mean(teacher_latent_feature ** 2)

            # Compute detection loss for the student model
            student_loss = detection_loss_fn(student_seg_logits, ground_truth)

            # Initialize distillation losses
            gkd_loss = torch.tensor(0.0, device=device)
            chamfer_loss = torch.tensor(0.0, device=device)

            # Gradient-guided loss (GKD)
            if use_gradient_guided:
                M_teacher = gradient_guided_features(teacher_loss, teacher_latent_feature, normalize=normalize_gkd,
                                                     norm_type=norm_type)
                M_teacher = M_teacher.detach()
                M_student = gradient_guided_features(student_loss, student_latent_feature, normalize=normalize_gkd,
                                                     norm_type=norm_type)
                gkd_loss = F.smooth_l1_loss(M_student, M_teacher, beta=1.0)
                # print(f"Correlation: {torch.corrcoef(torch.stack([M_teacher, M_student]))[0, 1].item()}")
                # print(f"Mean abs diff: {torch.mean(torch.abs(M_teacher - M_student)).item()}")

            # Persistent homology loss (Chamfer on TDA features)
            # Chamfer loss between teacher and student latent features
            # || given persistent diagram has been computed in Ptv3-model --> was used Ripser++ --> difficult to install
            # chamfer_loss = compute_chamfer_loss(teacher_latent_feature, student_latent_feature)

            # Now changed to gtda package
            # Use Knn to select representative points
            if use_persistent_homology:
                TDA_features_teacher = compute_ph_sparse_knn(teacher_latent_feature.feat, k_neighbors=50,
                                                             vectorRips_metric="precomputed",
                                                             knn_metric="euclidean")
                TDA_features_student = compute_ph_sparse_knn(student_latent_feature.feat, k_neighbors=50,
                                                             vectorRips_metric="precomputed",
                                                             knn_metric="euclidean")

                chamfer_loss = compute_chamfer_loss(TDA_features_teacher, TDA_features_student, dimensions=[0, 1, 2])

            # KLD loss
            kld_loss = kld_loss_fn(
                F.log_softmax(student_seg_logits, dim=1),
                F.softmax(teacher_seg_logits, dim=1)
            )

            # Combine losses
            total_student_loss = student_loss + gamma_param * kld_loss
            if use_gradient_guided:
                total_student_loss += lambda_param * gkd_loss
            if use_persistent_homology:
                total_student_loss += beta_param * chamfer_loss

            # Backpropagation
            student_optimizer.zero_grad()
            total_student_loss.backward()
            student_optimizer.step()

            # # Log training loss
            # global_step = epoch * len(train_loader) + batch_ndx
            # writer.add_scalar('Loss/student', student_loss.item(), global_step)
            # writer.add_scalar('Loss/chamfer', chamfer_loss.item(), global_step)
            # writer.add_scalar('Loss/teacher', teacher_loss.item(), global_step)
            # writer.add_scalar('Loss/KLD', kld_loss.item(), global_step)
            #
            # if use_gradient_guided:
            #     writer.add_scalar('Loss/gradient_guided_loss', distillation_loss.item(), global_step)
            #
            # writer.add_scalar('Metrics/student_mIoU', student_miou.item(), global_step)
            # writer.add_scalar('Metrics/teacher_mIoU', teacher_miou.item(), global_step)

            # Print progress
            # Print progress
            if batch_ndx % 50 == 0:
                # print(
                #     f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_ndx}/{len(train_loader)}], "
                #     f"Student Loss: {student_loss.item():.4f}, "
                #     f"Student mIoU: {student_miou.item():.4f}, "
                #     f"Teacher Loss: {teacher_loss.item():.4f}, "
                #     f"Teacher mIoU: {teacher_miou.item():.4f}, "
                #     f"KLD Loss: {kld_loss.item():.4f}"
                # )

                log_message = (
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_ndx}/{len(train_loader)}], "
                    f"Student Loss: {student_loss.item():.4f}, "
                    f"Student mIoU: {student_miou.item():.4f}, "
                    f"Teacher Loss: {teacher_loss.item():.4f}, "
                    f"Teacher mIoU: {teacher_miou.item():.4f}, "
                    f"KLD Loss: {kld_loss.item():.4f}"
                )

                logger.info(log_message)

                if use_gradient_guided:
                    # print(f"GKD Loss: {gkd_loss.item():.4f}")
                    # print(f"Correlation: {torch.corrcoef(torch.stack([M_teacher, M_student]))[0, 1].item()}")
                    # print(f"Mean abs diff: {torch.mean(torch.abs(M_teacher - M_student)).item()}")

                    logger.info(f"GKD Loss: {gkd_loss.item():.4f}")
                    logger.info(f"Correlation: {torch.corrcoef(torch.stack([M_teacher, M_student]))[0, 1].item()}")
                    logger.info(f"Mean abs diff: {torch.mean(torch.abs(M_teacher - M_student)).item()}")
                if use_persistent_homology:
                    # print(f"Chamfer Loss: {chamfer_loss.item():.4f}")
                    logger.info(f"Chamfer Loss: {chamfer_loss.item():.4f}")

                # print("-------------------------------------------")
                logger.info("-------------------------------------------")
        # Save checkpoints
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = os.path.join('checkpoints', f'student_checkpoint_epoch_{epoch + 1}.pth')
            torch.save(student_model.state_dict(), checkpoint_path)
            print(f"Student model checkpoint saved at {checkpoint_path}")

        # ====================== TEST EVALUATION ======================

        # print("---------> Testing")
        logger.info("---------> Testing")
        tester = test.CustomSemSegTester(cfg=cfg, model=student_model)
        tester.test()

    # writer.close()


def gradient_guided_features(output_loss, enc_feature, normalize=True, norm_type='minmax'):
    """
    Compute gradient-guided features with optional normalization.
    Args:
        output_loss: Scalar loss tensor.
        enc_feature: Feature object with .feat tensor of shape (N, C).
        normalize: Whether to normalize the output M.
        norm_type: 'minmax' or 'zscore' for normalization type.
    Returns:
        M: Tensor of shape (N,) with per-point importance scores.
    """

    if not enc_feature.feat.requires_grad:
        raise ValueError("enc_feature.feat must have requires_grad=True")
    if enc_feature.feat.grad_fn is None:
        raise ValueError("enc_feature.feat has no grad_fn; it is detached")

    grads = torch.autograd.grad(
        outputs=output_loss,
        inputs=enc_feature.feat,
        grad_outputs=torch.ones_like(output_loss),
        retain_graph=True,
        create_graph=False
    )[0]  # Shape: (N, C)

    # Debug gradient norm
    # print(f"Gradient norm: {grads.norm().item()}")

    weights = grads.abs().mean(dim=0)  # Shape: (C,)
    # print(f"Weights min/max/std: {weights.min().item()}/{weights.max().item()}/{weights.std().item()}")

    weighted_feat = enc_feature.feat * weights  # Shape: (N, C)
    M = weighted_feat.sum(dim=1)  # Shape: (N,)
    # print(f"M range before norm: {(M.max() - M.min()).item()}")

    if normalize:
        if norm_type == 'minmax':
            M = (M - M.min()) / (M.max() - M.min() + 1e-6)
        elif norm_type == 'zscore':
            M = (M - M.mean()) / (M.std() + 1e-6)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")
    else:
        print("No normalization applied to M")

    return M


if __name__ == "__main__":

    # Activate both GKD and persistent homology
    # main(use_gradient_guided=True, use_persistent_homology=True, normalize_gkd=True, norm_type='minmax')

    # Activate only GKD
    main(use_gradient_guided=True, use_persistent_homology=False, normalize_gkd=True, norm_type='minmax')

    # Activate only persistent homology
    # main(use_gradient_guided=False, use_persistent_homology=True, normalize_gkd=True, norm_type='minmax')

    # Disable both (only segmentation and KLD losses)
    # main(use_gradient_guided=False, use_persistent_homology=False, normalize_gkd=True, norm_type='minmax')

    # No normalization for GKD
    # main(use_gradient_guided=True, use_persistent_homology=True, normalize_gkd=False)

