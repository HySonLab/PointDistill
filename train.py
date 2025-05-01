import torch
# from torch.utils.tensorboard import SummaryWriter
from dataloader import PTv3_Dataloader
from PTv3_model import PointTransformerV3TrainTeacher, load_weights_ptv3_nucscenes_seg
from pointcept.engines.defaults import default_config_parser, default_setup
from pointcept.models.losses import build_criteria
import os
from gpu_main import compute_miou, compute_iou_all_classes
import torch.nn as nn
from gpu_main import get_student_model, get_teacher_model, gradient_guided_features, setup_logging
import torch.nn.functional as F
from custom_vectorips import DifferentiableVietorisRipsPersistence
from pytorch3d.loss import chamfer_distance
from pointcept.engines import test
from tqdm import tqdm


class TopoLoss(nn.Module):
    def __init__(self, loss_type='chamfer', input_dim=3, hidden_dim=64, output_dim=1, reduction='mean'):
        """
        Topological loss module to compare persistence diagrams.

        Args:
            loss_type (str): Type of loss ('chamfer' or 'neural'). Default: 'chamfer'.
            input_dim (int): Input dimension of persistence diagrams (typically 3: birth, death, dim). Default: 3.
            hidden_dim (int): Hidden dimension for neural aggregator. Default: 64.
            output_dim (int): Output dimension for neural aggregator. Default: 1.
            reduction (str): Reduction method for batch loss ('mean', 'sum', or 'none'). Default: 'mean'.
        """
        super().__init__()
        self.loss_type = loss_type.lower()
        self.reduction = reduction.lower()

        if self.loss_type not in ['chamfer', 'neural']:
            raise ValueError("loss_type must be 'chamfer' or 'neural'")

        # Initialize neural aggregator for 'neural' loss
        if self.loss_type == 'neural':
            self.aggregator = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.ReLU()
            )
            self.loss_fn = nn.MSELoss(reduction='none')  # Per-diagram loss, reduced later

        # Validate reduction
        if self.reduction not in ['mean', 'sum', 'none']:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")

    def _aggregate_diagram(self, diagram):
        """Aggregate a single persistence diagram using the neural network."""
        if diagram.shape[0] == 0:
            return torch.tensor(0.0, device=diagram.device, requires_grad=True)
        return self.aggregator(diagram).sum()

    def _chamfer_distance(self, student_diag, teacher_diag):
        """Compute Chamfer distance between two persistence diagrams."""
        if student_diag.shape[0] == 0 or teacher_diag.shape[0] == 0:
            return torch.tensor(0.0, device=student_diag.device, requires_grad=True)
        loss, _ = chamfer_distance(student_diag[:, :2].unsqueeze(0), teacher_diag[:, :2].unsqueeze(0))
        return loss

    def forward(self, student_diagrams, teacher_diagrams):
        """
        Compute topological loss between student and teacher persistence diagrams.

        Args:
            student_diagrams (list of torch.Tensor): List of student persistence diagrams,
                each with shape [n_features, 3] (birth, death, dim).
            teacher_diagrams (list of torch.Tensor): List of teacher persistence diagrams,
                each with shape [n_features, 3].

        Returns:
            torch.Tensor: Scalar loss (if reduction='mean' or 'sum') or per-batch losses (if reduction='none').
        """
        if len(student_diagrams) != len(teacher_diagrams):
            raise ValueError("Student and teacher diagrams must have the same batch size")

        batch_losses = []

        if self.loss_type == 'neural':
            # Compute features for each diagram and compare with MSE
            for student_diag, teacher_diag in zip(student_diagrams, teacher_diagrams):
                student_feature = self._aggregate_diagram(student_diag)
                teacher_feature = self._aggregate_diagram(teacher_diag)
                loss = self.loss_fn(student_feature, teacher_feature)
                batch_losses.append(loss)
        else:  # chamfer
            # Compute Chamfer distance between diagrams
            for student_diag, teacher_diag in zip(student_diagrams, teacher_diagrams):
                loss = self._chamfer_distance(student_diag, teacher_diag)
                batch_losses.append(loss)

        # Stack losses and apply reduction
        batch_losses = torch.stack(batch_losses)
        if self.reduction == 'mean':
            return batch_losses.mean()
        elif self.reduction == 'sum':
            return batch_losses.sum()
        return batch_losses

    def to(self, device):
        """Move the module to the specified device."""
        super().to(device)
        return self


def train_and_evaluate(use_gradient_guided=False, use_persistent_homology=False, normalize_gkd=True,
                       norm_type='minmax', dataset='nuscenes'):
    # Pretrained model path and config file
    PRETRAINED_PATH = './checkpoints/model_best.pth'  # './checkpoints/checkpoint_epoch_50_backup.pth'
    CONFIG_FILE = f"configs/{dataset}/semseg-pt-v3m1-0-distill.py"

    # Load configuration
    cfg = default_config_parser(CONFIG_FILE, None)
    cfg = default_setup(cfg)

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

    # Load data
    loader = PTv3_Dataloader(cfg)
    train_loader = loader.load_training_data()

    # Move models to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)

    # Loss function
    detection_loss_fn = build_criteria(cfg.model.criteria)

    # KD loss
    kld_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')  # KLD loss

    # Optimizer for student model
    student_optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay
    )

    # Loss scaling parameters
    lambda_param = 10.0  # For GKD loss --> amplify the importance features
    beta_param = 0.1  # For persistent homology (Chamfer) loss
    gamma_param = 0.3  # For KLD loss

    vr = DifferentiableVietorisRipsPersistence(homology_dimensions=[0, 1, 2], max_edge_length=2.0,
                                               max_edges=200, use_knn=False, knn_neighbors=20,
                                               auto_config=True,
                                               distance_method="optimized_cdist")  # dist_matrix, cdist

    # Initialize TopoLoss
    topo_loss_fn = TopoLoss(loss_type='chamfer', reduction='mean').to(device)

    # Training loop
    # teacher_model.eval()  # Freeze the teacher model
    student_model.train()
    print("--------> Training Start")

    num_epochs = cfg.epoch

    num_classes = len(cfg.names)  # Assuming cfg.names contains class names

    for epoch in tqdm(range(num_epochs), desc="Epochs", leave=True):

        # Inner progress bar for batches
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for batch_ndx, input_dict in enumerate(train_loader):

            # Move input data to device
            input_dict = {k: v.to(device, non_blocking=True) for k, v in input_dict.items()}

            ground_truth = input_dict["segment"]

            teacher_seg_logits, teacher_latent_feature = teacher_model(input_dict)  # teacher_latent_feature (N, 512)
            teacher_preds = torch.argmax(teacher_seg_logits, dim=1)
            teacher_iou_scores = compute_iou_all_classes(
                teacher_preds,
                ground_truth,
                num_classes
            )
            teacher_miou = compute_miou(teacher_iou_scores)

            # # Forward pass: Student model
            student_seg_logits, student_latent_feature = student_model(input_dict)  # student_latent_feature (N, 512)
            student_preds = torch.argmax(student_seg_logits, dim=1)

            student_iou_scores = compute_iou_all_classes(student_preds, ground_truth, num_classes)
            student_miou = compute_miou(student_iou_scores)

            # Compute loss for the teacher model
            teacher_loss = detection_loss_fn(teacher_seg_logits, ground_truth) #+ torch.mean(teacher_latent_feature ** 2)

            # Compute loss for the student model
            student_loss = detection_loss_fn(student_seg_logits, ground_truth)

            # Initialize distillation losses
            gkd_loss = torch.tensor(0.0, device=device)
            topo_loss = torch.tensor(0.0, device=device)

            # Gradient-guided loss (GKD)
            if use_gradient_guided:
                M_teacher = gradient_guided_features(teacher_loss, teacher_latent_feature, normalize=normalize_gkd,
                                                     norm_type=norm_type)
                M_teacher = M_teacher.detach()
                M_student = gradient_guided_features(student_loss, student_latent_feature, normalize=normalize_gkd,
                                                     norm_type=norm_type)
                gkd_loss = F.smooth_l1_loss(M_student, M_teacher, beta=1.0)

            if use_persistent_homology:

                feat_teacher_shape = teacher_latent_feature.feat.shape
                feat_student_shape = student_latent_feature.feat.shape

                # Convert to shape (batch, number_of_points, features)
                if len(feat_teacher_shape) == 2 and len(feat_student_shape) == 2:

                    student_latent_feature = student_latent_feature.feat.unsqueeze(0)
                    teacher_latent_feature = teacher_latent_feature.feat.unsqueeze(0)
                else:
                    student_latent_feature = student_latent_feature.feat
                    teacher_latent_feature = teacher_latent_feature.feat

                teacher_diagrams = vr.fit_transform(teacher_latent_feature)

                student_diagrams = vr.fit_transform(student_latent_feature)

                # Compute topological loss
                topo_loss = topo_loss_fn(student_diagrams, teacher_diagrams)

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
                total_student_loss += beta_param * topo_loss

            # Backpropagation
            student_optimizer.zero_grad()
            total_student_loss.backward()
            student_optimizer.step()

            # Update progress bar with metrics
            metrics = {
                "student_loss": f"{total_student_loss.item():.4f}",
                "student_miou": f"{student_miou.item():.4f}",
                "teacher_loss": f"{teacher_loss.item():.4f}",
                "teacher_miou": f"{teacher_miou.item():.4f}",
                "kld_loss": f"{kld_loss.item():.4f}",
            }
            if use_gradient_guided:
                metrics["gkd_loss"] = f"{gkd_loss.item():.4f}"
            if use_persistent_homology:
                metrics["topo_loss"] = f"{topo_loss.item():.4f}"

            batch_pbar.set_postfix(metrics)

            # if batch_ndx % 100 == 0:
            #
            #     log_message = (
            #         f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_ndx}/{len(train_loader)}], "
            #         f"Student Loss: {total_student_loss.item():.4f}, "
            #         f"Student mIoU: {student_miou.item():.4f}, "
            #         f"Teacher Loss: {teacher_loss.item():.4f}, "
            #         f"Teacher mIoU: {teacher_miou.item():.4f}, "
            #         f"KLD Loss: {kld_loss.item():.4f}"
            #     )
            #
            #     logger.info(log_message)
            #
            #     if use_gradient_guided:
            #         # print(f"GKD Loss: {gkd_loss.item():.4f}")
            #         # print(f"Correlation: {torch.corrcoef(torch.stack([M_teacher, M_student]))[0, 1].item()}")
            #         # print(f"Mean abs diff: {torch.mean(torch.abs(M_teacher - M_student)).item()}")
            #
            #         logger.info(f"GKD Loss: {gkd_loss.item():.4f}")
            #
            #     if use_persistent_homology:
            #         # print(f"Chamfer Loss: {chamfer_loss.item():.4f}")
            #         logger.info(f"Topo Loss: {topo_loss.item():.4f}")
            #
            #     logger.info("-------------------------------------------")

        # Save checkpoints
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = os.path.join('checkpoints', f'student_checkpoint_epoch_{epoch + 1}.pth')
            torch.save(student_model.state_dict(), checkpoint_path)
            print(f"Student model checkpoint saved at {checkpoint_path}")

        logger.info("---------> Testing")
        tester = test.CustomSemSegTester(cfg=cfg, model=student_model)
        tester.test()


if __name__ == "__main__":

    # Activate both GKD and persistent homology
    train_and_evaluate(use_gradient_guided=True, use_persistent_homology=True,
                       normalize_gkd=True, norm_type='minmax', dataset='nuscenes')

    # Activate only GKD
    # train_and_evaluate(use_gradient_guided=True, use_persistent_homology=False,
    #                    normalize_gkd=True, norm_type='minmax', dataset='nuscenes')

    # Activate only persistent homology
    # train_and_evaluate(use_gradient_guided=False, use_persistent_homology=True,
    #                    normalize_gkd=True, norm_type='minmax', dataset='nuscenes')

    # Disable both (only segmentation and KLD losses)
    # train_and_evaluate(use_gradient_guided=False, use_persistent_homology=False,
    #                    normalize_gkd=True, norm_type='minmax', dataset='nuscenes')

    # No normalization for GKD
    # train_and_evaluate(use_gradient_guided=True, use_persistent_homology=True,
    #                    normalize_gkd=False, dataset='nuscenes')

