"""
PyTorch LightningModule for 3D DINOv2 self-supervised training.
Handles optimizer, scheduler, training/validation steps, and teacher-student updates.
"""

import os
import re
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

import torch
from torch import nn
from torch import Tensor
from torch.optim import AdamW, Optimizer
from pytorch_lightning import LightningModule

from lightly.utils.optim import update_param_groups
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule

from losses.dino import DINOv2Loss
from utils.misc import jacobian_metrics
from utils.metrics import dice_coeff
from utils.pca import pca_lowrank_transform
from utils.convexAdam_3D import  convex_adam_3d_param
from models.meta_arch import DINOv2_3D_Meta_Architecture


class DINOv2_3D_LightningModule(LightningModule):
    """
    PyTorch LightningModule for 3D DINOv2 self-supervised learning.
    Implements training, prediction, optimizer config, and teacher-student logic.
    """

    def __init__(
        self,
        batch_size_per_device: int = 4,
        hidden_size: int = 768,
        ibot_separate_head: bool = True,
        base_lr: float = 0.0005,  # Reduced from 0.004 as per issue #6
        min_lr: float = 1e-6,
        weight_decay: float = 0.04,
        layer_decay: float = 0.9,
        gradient_clip_val: float = 3.0,
        teacher_temp_warmup_epochs: int = 30,
        teacher_temp_min: float = 0.04,
        teacher_temp_max: float = 0.07,
        freeze_last_layer_epochs: int = 1,
        projection_dim: int = 65536,
        ibot_projection_dim: int = 8192,
        mask_ratio_min: float = 0.6,
        mask_ratio_max: float = 0.8,
        sampling = 'random',
        momentum_start_value: float = 0.992,
        momentum_end_value: float = 0.998,
        apply_gin: bool = False,
        ckpt_path: str = '',
        output_dir: str = '',
        backbone: nn.Module = None,
    ) -> None:
        """
        Initialize the DINOv2Trainer3D LightningModule.
        Args: see model config for details.
        """
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.layer_decay = layer_decay
        self.gradient_clip_val = gradient_clip_val
        self.teacher_temp_warmup_epochs = teacher_temp_warmup_epochs
        self.teacher_temp_min = teacher_temp_min
        self.teacher_temp_max = teacher_temp_max
        self.momentum_start_value = momentum_start_value
        self.momentum_end_value = momentum_end_value
        self.freeze_last_layer_epochs = freeze_last_layer_epochs
        self.metrics = {"train": None, "val": None}

        self.save_hyperparameters()

        # Model
        self.model = DINOv2_3D_Meta_Architecture(
            hidden_size=hidden_size,
            norm_last_layer=False,
            ibot_separate_head=ibot_separate_head,
            projection_dim=projection_dim,
            ibot_projection_dim=ibot_projection_dim,
            mask_ratio_min=mask_ratio_min,
            mask_ratio_max=mask_ratio_max,
            sampling=sampling,
            backbone=backbone,
            freeze_last_layer=self.freeze_last_layer_epochs,
            apply_gin=apply_gin
        )

        if ckpt_path:
            ckpt = torch.load(ckpt_path, weights_only=False)
            new_state_dict = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items()}
            self.model.load_state_dict(new_state_dict, strict=False)

        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.df_results = pd.DataFrame()

        self.kidney_left_dice = []
        self.kidney_right_dice = []
        self.liver_dice = []
        self.spleen_dice = []
        self.mean_dice = []

        self.init_kidney_left_dice = []
        self.init_kidney_right_dice = []
        self.init_liver_dice = []
        self.init_spleen_dice = []
        self.init_mean_dice = []

        self.fixed = []
        self.moving = []

        self.sdjlog = []
        self.fold_ratio = []

        # Loss
        self.criterion = DINOv2Loss(
            teacher_temp_min=teacher_temp_min,
            teacher_temp_max=teacher_temp_max,
            teacher_temp_warmup_epochs=teacher_temp_warmup_epochs,
            output_dim=projection_dim,
            ibot_output_dim=ibot_projection_dim,
            ibot_loss_weight=1.0,
            koleo_loss_weight=0.1,
        )

        # self.online_classifier = OnlineLinearClassifier(
        #     feature_dim=768,
        #     num_classes=num_classes,
        # )

    def predict_step(
        self, batch: tuple[list[Tensor], Tensor, list[str]], batch_idx: int
    ) -> Tensor:
        """
        Inference step for prediction mode.
        Args:
            batch: Tuple of (inputs, targets, meta)
            batch_idx: Batch index
        Returns:
            Model outputs
        """
        inputs = batch[0]
        with torch.no_grad():
            outputs = self.model.encode(inputs)
            return outputs

    def training_step(
        self, batch: tuple[list[Tensor], Tensor, list[str]], batch_idx: int
    ) -> Tensor:
        views, targets = batch[0], batch[1]

        # Forward pass
        outputs = self.model(views)

        # Calculate losses with proper scheduling
        loss_dict = self.criterion(
            outputs["pred"], global_step=self.trainer.global_step
        )

        # Log losses
        self.log_dict(
            {
                "train_loss": loss_dict["total_loss"],
                "train_dino_loss": loss_dict["dino_loss"],
                "train_ibot_loss": loss_dict["ibot_loss"]
                if loss_dict["ibot_loss"] is not None
                else 0.0,
                "train_koleo_loss": loss_dict["koleo_loss"]
                if loss_dict["koleo_loss"] is not None
                else 0.0,
                "teacher_temp": loss_dict["teacher_temp"],
                "global_step": float(self.trainer.global_step), # Add for debugging
                "lr": self.trainer.optimizers[0].param_groups[0]['lr'],  # Add for debugging
                "weight_decay": cosine_schedule(
                    step=self.trainer.global_step,
                    max_steps=self.trainer.estimated_stepping_batches,
                    start_value=0.04,
                    end_value=0.4),  # Add for debugging,
                "teacher_momentum": cosine_schedule(
                    step=self.trainer.global_step,
                    max_steps=self.trainer.estimated_stepping_batches,
                    start_value=0.992,
                    end_value=1.0)
            },
            prog_bar=False,
            sync_dist=True,
            batch_size=len(targets),
        )
        return loss_dict["total_loss"]

    def validation_step(
        self, batch: tuple[Tensor, Tensor, list[str]], batch_idx: int
    ) -> Tensor:

        fix_arr = batch[0:1][0:1]
        mov_arr = batch[1:2][0:1]
        fix_lbs = batch[2:3][0:1]
        mov_lbs = batch[3:4][0:1]

        self.fixed.append(os.path.basename(self.trainer.datamodule.val_dataset.dataset.data[batch_idx]["fixed"])[:-7])
        self.moving.append(os.path.basename(self.trainer.datamodule.val_dataset.dataset.data[batch_idx]["moving"])[:-7])

        OFD, OFH, OFW = fix_arr.shape[2], fix_arr.shape[3], fix_arr.shape[4]
        OMD, OMH, OMW = fix_arr.shape[2], fix_arr.shape[3], fix_arr.shape[4]

        FD, FH, FW = fix_arr.shape[2], fix_arr.shape[3], fix_arr.shape[4]
        MD, MH, MW = mov_arr.shape[2], mov_arr.shape[3], mov_arr.shape[4]

        patch_size = self.model.student_backbone.patch_size
        PFD, PFH, PFW = FD // patch_size[0], FH // patch_size[1], FW // patch_size[1]
        PMD, PMH, PMW = MD // patch_size[0], MH // patch_size[1], MW // patch_size[1]

        # calculate initial dice
        dice = dice_coeff(mov_lbs.contiguous(), fix_lbs.contiguous(), 5)  # 5 - nako, 14 - abdomenctct
        print(f"\nInitial dice: {dice.mean().item()}")

        # self.init_kidney_left_dice.append(dice[0].item())
        # self.init_kidney_right_dice.append(dice[1].item())
        # self.init_liver_dice.append(dice[2].item())
        # self.init_spleen_dice.append(dice[3].item())
        # self.init_mean_dice.append(dice.mean().item())

        fix_feature = self.model.student_backbone(fix_arr).squeeze().cpu().numpy()
        mov_feature = self.model.student_backbone(mov_arr).squeeze().cpu().numpy()

        if self.output_dir:
            np.save(os.path.join(self.output_dir, os.path.basename(self.trainer.datamodule.val_dataset.dataset.data[batch_idx]["fixed"])[:-7] + "_" + "fixed_features.npy"), fix_feature)
            np.save(os.path.join(self.output_dir, os.path.basename(self.trainer.datamodule.val_dataset.dataset.data[batch_idx]["moving"])[:-7] + "_" + "moving_features.npy"), mov_feature)

        all_features = np.concatenate((mov_feature[1:, :], fix_feature[1:, :]), axis=0)

        # PCA with 3 channels for visualization
        reduced_patches, eigenvalues = pca_lowrank_transform(all_features, 3)

        mov_pca = reduced_patches[:PMD * PMH * PMW, :]
        fix_pca = reduced_patches[PMD * PMH * PMW:, :]
        mov_pca = mov_pca.reshape([PMD, PMH, PMW, -1])
        fix_pca = fix_pca.reshape([PFD, PFH, PFW, -1])

        fix_pca = fix_pca.unsqueeze(0).permute(0, 4, 1, 2, 3).cuda()
        mov_pca = mov_pca.unsqueeze(0).permute(0, 4, 1, 2, 3).cuda()
        fix_pca_rescaled = nn.functional.interpolate(fix_pca, size=(OFD, OFH, OFW), mode="trilinear", align_corners=False)
        mov_pca_rescaled = nn.functional.interpolate(mov_pca, size=(OMD, OMH, OMW), mode="trilinear", align_corners=False)
        fix_pca_rescaled = fix_pca_rescaled.permute(0, 2, 3, 4, 1).squeeze().cpu()
        mov_pca_rescaled = mov_pca_rescaled.permute(0, 2, 3, 4, 1).squeeze().cpu()

        mov_pca_rescaled = (mov_pca_rescaled - mov_pca_rescaled.min()) / (mov_pca_rescaled.max() - mov_pca_rescaled.min())
        fix_pca_rescaled = (fix_pca_rescaled - fix_pca_rescaled.min()) / (fix_pca_rescaled.max() - fix_pca_rescaled.min())

        # Create matplotlib figure
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].imshow(mov_pca_rescaled[:, :, OMW // 2])
        ax[0].set_title(f"Moving features")
        ax[1].imshow(fix_pca_rescaled[:, :, OFW // 2])
        ax[1].set_title(f"Fixed features")
        plt.tight_layout()

        # Convert to numpy image
        self.logger.experiment.log({f'val/pca_3_channels_{batch_idx}': wandb.Image(fig)})
        plt.close(fig)

        # PCA with 12 channels for registration
        reduced_patches, eigenvalues = pca_lowrank_transform(all_features, 12)

        mov_pca = reduced_patches[:PMD*PMH*PMW, :]
        fix_pca = reduced_patches[PMD*PMH*PMW:, :]
        mov_pca = mov_pca.reshape([PMD, PMH, PMW, -1])
        fix_pca = fix_pca.reshape([PFD, PFH, PFW, -1])

        fix_pca = fix_pca.unsqueeze(0).permute(0, 4, 1, 2, 3).cuda()
        mov_pca = mov_pca.unsqueeze(0).permute(0, 4, 1, 2, 3).cuda()
        fix_pca_rescaled = nn.functional.interpolate(fix_pca, size=(OFD, OFH, OFW), mode="trilinear", align_corners=False)
        mov_pca_rescaled = nn.functional.interpolate(mov_pca, size=(OMD, OMH, OMW), mode="trilinear", align_corners=False)

        fix_pca_rescaled = fix_pca_rescaled.permute(0, 2, 3, 4, 1).squeeze().cpu().numpy()
        mov_pca_rescaled = mov_pca_rescaled.permute(0, 2, 3, 4, 1).squeeze().cpu().numpy()

        if self.output_dir:
            np.save(os.path.join(self.output_dir, os.path.basename(self.trainer.datamodule.val_dataset.dataset.data[batch_idx]["fixed"])[:-7] + "_" + "fixed_features_pca.npy"), fix_pca_rescaled)
            np.save(os.path.join(self.output_dir, os.path.basename(self.trainer.datamodule.val_dataset.dataset.data[batch_idx]["moving"])[:-7] + "_" + "moving_features_pca.npy"), mov_pca_rescaled)

        """ConvexAdam optimization"""
        print('Starting ConvexAdam optimization')

        grid_sp_adam = 4
        smooth_weight = 0.12  # 0.12 - nako, 0.05 - ct
        num_iter = 1000
        lr = 0.2
        iter_smooth_kernel = 3
        iter_smooth_num = 3
        final_upsample = 1

        with torch.enable_grad():
            disp = convex_adam_3d_param(
                fix_pca_rescaled * 0.1,
                mov_pca_rescaled * 0.1,
                fix_lbs,
                mov_lbs,
                disp_hw=1,
                loss_func="SSD",
                grid_sp_adam=grid_sp_adam,
                lambda_weight=smooth_weight,
                selected_niter=num_iter,
                lr=lr,
                disp_init=None,
                iter_smooth_kernel=iter_smooth_kernel,
                iter_smooth_num=iter_smooth_num,
                end_smooth_kernel=1,
                final_upsample=final_upsample
            )

        sdjlog, fold_ratio = jacobian_metrics(torch.tensor(disp).permute(3, 0, 1, 2).unsqueeze(0))
        print(f'SDJLog: {sdjlog}, FoldRatio: {fold_ratio}')
        self.log("val_sdjlog", sdjlog, on_step=False, on_epoch=True)
        self.log("val_foldratio", fold_ratio, on_step=False, on_epoch=True)

        # warp moving image
        disp = np.moveaxis(disp, 3, 0)
        D, H, W = mov_arr.shape[-3], mov_arr.shape[-2], mov_arr.shape[-1]
        identity = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
        warped_lbs = map_coordinates(mov_lbs.cpu().numpy().squeeze(), identity + disp, order=0)

        dice = dice_coeff(torch.Tensor(warped_lbs).unsqueeze(0).unsqueeze(0).contiguous().cuda(), fix_lbs.contiguous(), 5)  # 5 - nako, 14 - abdomenctct
        print(f"Final dice: {dice.mean().item()}\n")
        self.log("val_dice", dice.mean().item(), prog_bar=True, on_step=False, on_epoch=True)

        return dice.mean()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(self):

        # Calculate learning rate based on batch size
        # lr_scale = math.sqrt(
        #     self.batch_size_per_device * self.trainer.world_size / 1024
        # )
        # lr = self.base_lr * lr_scale

        lr = self.base_lr
        print(f'Initial learning rate is {lr:.6f}')

        num_layers = len(self.model.student_backbone.vit.blocks)

        def lr_layer(layer_idx: int) -> float:
            return self.layer_decay ** (num_layers + 1 - layer_idx)

        # Create parameter groups with layer-wise learning rates
        param_groups = []

        # Fix: Only include student parameters that require gradients
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # Skip teacher parameters (they should not require gradients anyway)
            if "teacher" in name:
                continue

            # Skip if not student parameters
            if "student" not in name:
                continue

            group = {
                "name": name,
                "params": [param],
                "lr": lr,
                "weight_decay": self.weight_decay,
            }

            # Update lr based on layer
            if any(
                s in name
                for s in [
                    "pos_embed",
                    "mask_token",
                    "cls_token",
                    "register_tokens",
                ]
            ):
                group["lr"] = lr * lr_layer(0)
            elif "patch_embed" in name:
                group["lr"] = lr * lr_layer(0) * 0.2
            elif "residual" in name:
                group["lr"] = lr
            elif "blocks" in name:
                # Fix: More robust regex matching
                match = re.search(r"blocks\.(\d+)\.", name)
                if match:
                    layer_idx = int(match.group(1))
                    group["lr"] = lr * lr_layer(layer_idx + 1)
            elif "norm" in name:
                # Use default lr for norm layers
                pass
            elif "head" in name or "_dino_head" in name or "_ibot_head" in name:
                # Use default lr for heads
                pass
            else:
                # For any other student parameters, use default lr
                pass

            # Update weight_decay
            if name.endswith(".bias") or ".norm" in name or "gamma" in name:
                group["weight_decay"] = 0.0

            # Include parameter group
            param_groups.append(group)

        # Ensure we have parameters to optimize
        if not param_groups:
            raise ValueError("No student parameters found for optimization!")

        print(f"Found {len(param_groups)} parameter groups for optimization")
        print(
            f"Total parameters: {sum(len(group['params']) for group in param_groups)}"
        )

        optimizer = AdamW(param_groups, lr=lr)

        # Fix: Ensure proper scheduler configuration
        max_steps = max(self.trainer.estimated_stepping_batches, 1)
        warmup_steps = int(max_steps / self.trainer.max_epochs * 10)

        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=warmup_steps,
                max_epochs=max_steps,
                end_value=self.min_lr / lr,
            ),
            "interval": "step",
        }

        self.criterion.max_steps = max_steps
        self.criterion.max_epochs = self.trainer.max_epochs
        return [optimizer], [scheduler]

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        # Gradient clipping as per issue #10
        self.clip_gradients(
            optimizer=optimizer,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm="norm",
        )

    def on_before_optimizer_step(self, optimizer: AdamW, *args) -> None:
        # Cancel last layer gradients during warmup (issue #5)
        self.model.cancel_last_layer_gradients(self.current_epoch)

        # Apply weight decay schedule
        weight_decay = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=max(self.trainer.estimated_stepping_batches, 1),
            start_value=0.04,
            end_value=0.4,
        )
        updates = []
        for group in optimizer.param_groups:
            if group["weight_decay"] != 0.0:
                updates.append({"name": group["name"], "weight_decay": weight_decay})

        update_param_groups(optimizer, updates=updates)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # EMA update of teacher
        max_steps = max(self.trainer.estimated_stepping_batches, 1000)

        # Update teacher - this should work fine in DDP
        self.model.update_teacher(
            global_step=self.trainer.global_step, max_steps=max_steps, start_value=self.momentum_start_value, end_value=self.momentum_end_value
        )

        # Remove manual synchronization - it's causing the hangups
        # DDP will handle this automatically after the backward pass
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def _sync_teacher_parameters(self):
        """Synchronize teacher parameters across all DDP processes."""
        # Remove the problematic conditional synchronization
        # The issue is that all processes need to participate in broadcast

        if self.trainer.world_size <= 1:
            return  # No sync needed for single GPU

        # Use all_reduce instead of broadcast to avoid deadlocks
        with torch.no_grad():
            for param in self.model.teacher_backbone.parameters():
                torch.distributed.all_reduce(
                    param.data, op=torch.distributed.ReduceOp.AVG
                )
            for param in self.model.teacher_dino_head.parameters():
                torch.distributed.all_reduce(
                    param.data, op=torch.distributed.ReduceOp.AVG
                )
            if self.model.ibot_separate_head:
                for param in self.model.teacher_ibot_head.parameters():
                    torch.distributed.all_reduce(
                        param.data, op=torch.distributed.ReduceOp.AVG
                    )
