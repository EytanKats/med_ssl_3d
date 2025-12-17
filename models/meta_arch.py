"""
DINOv2_3D model definition for 3D self-supervised learning.
Includes teacher/student ViT, DINO/iBOT heads, and update/cancel logic.
"""

import copy
import random
import torch
from torch import nn
import torch.nn.functional as F

from lightly.models.modules import DINOProjectionHead
from lightly.utils.scheduler import cosine_schedule
from lightly.models.utils import update_momentum
from transforms.blockmask import RandomBlockMask3D

from torchvision.transforms import Compose
from monai.transforms import (
    RandHistogramShift,
    RandGaussianSmooth,
    RandBiasField,
    RandScaleIntensity,
    RandShiftIntensity,
    RandGaussianNoise,
    ScaleIntensityRange
)

from utils.gin import gin_aug

# References:
# Thanks to the following repositories that provided the structure and necessary components for this implementation:
#     https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers
#     https://github.com/Project-MONAI/VISTA
#     https://github.com/lightly-ai/lightly/blob/master/benchmarks/imagenet/vitb16/dinov2.py


def random_flip_3d(x):
    flips = [random.choice([True, False]) for _ in range(3)]
    if flips[0]:
        x = torch.flip(x, dims=[2])  # depth
    if flips[1]:
        x = torch.flip(x, dims=[3])  # height
    if flips[2]:
        x = torch.flip(x, dims=[4])  # width
    return x, flips


def undo_flip_3d(x, flips):
    if flips[2]:
        x = torch.flip(x, dims=[4])
    if flips[1]:
        x = torch.flip(x, dims=[3])
    if flips[0]:
        x = torch.flip(x, dims=[2])
    return x


def local_entropy_3d(x, kernel_size=9, eps=1e-6):
    """
    Compute local entropy for 3D volumes.

    Args:
        x: (B, 1, H, W, D) tensor — intensity or single-channel feature map
        kernel_size: size of local neighborhood for entropy estimation
    Returns:
        entropy_map: (B, 1, H, W, D) tensor
    """
    B, C, H, W, D = x.shape
    assert C == 1, "Use single-channel (grayscale/intensity) input."

    # Compute local histogram approximation using local mean and variance
    patches = F.unfold(
        x.view(B, 1, H, W * D),  # flatten depth for now
        kernel_size=kernel_size,
        padding=kernel_size // 2
    )  # (B, kernel_size**2, H*W*D)

    # Compute probabilities (soft histogram via softmax)
    p = F.softmax(patches / (patches.std(dim=1, keepdim=True) + eps), dim=1)
    entropy = -(p * torch.log(p + eps)).sum(dim=1)
    entropy = entropy.view(B, 1, H, W, D)

    # Normalize entropy to [0, 1]
    entropy = (entropy - entropy.amin()) / (entropy.amax() - entropy.amin() + eps)
    return entropy


def entropy_to_mask_prob(entropy_map, min_mask=0.3, max_mask=0.8):
    """
    Convert entropy map (0–1) to per-voxel masking probability.
    High-entropy regions → high mask ratio (maks them),
    Low-entropy regions → low mask ratio (keep them).
    """
    mask_prob = min_mask + (max_mask - min_mask) * entropy_map
    return mask_prob


def sample_entropy_mask(entropy_map, min_mask=0.3, max_mask=0.8):
    mask_prob = entropy_to_mask_prob(entropy_map, min_mask, max_mask)
    mask = torch.bernoulli(mask_prob)
    return mask  # shape (B, 1, H, W, D), 1 = masked


def freeze_eval_module(module: nn.Module) -> None:
    """Freeze the parameters of a module."""
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


# Wrappers to ensure the param groups are named differently for the DINO and iBOT heads
class DINOHead(nn.Module):
    """A wrapper for the DINO projection head."""

    def __init__(self, dino_head: nn.Module) -> None:
        super().__init__()
        self._dino_head = dino_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._dino_head(x)

    def cancel_last_layer_gradients(self, current_epoch: int) -> None:
        self._dino_head.cancel_last_layer_gradients(current_epoch)


class IBOTHead(nn.Module):
    """A wrapper for the iBOT projection head."""

    def __init__(self, ibot_head: nn.Module) -> None:
        super().__init__()
        self._ibot_head = ibot_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._ibot_head(x)

    def cancel_last_layer_gradients(self, current_epoch: int) -> None:
        self._ibot_head.cancel_last_layer_gradients(current_epoch)


class DINOv2_3D_Meta_Architecture(nn.Module):
    """
    3D DINOv2 model with teacher/student ViT backbones and DINO/iBOT heads.
    Supports block masking and teacher-student EMA updates.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        norm_last_layer: bool = False,
        ibot_separate_head: bool = True,
        freeze_last_layer: int = -1,
        projection_dim: int = 65536,
        mask_ratio_min: float = 0.6,
        mask_ratio_max: float = 0.8,
        ibot_projection_dim = 65536,
        sampling = 'random',
        apply_gin = False,
        backbone: nn.Module = None,
    ):
        """
        Initialize DINOv2_3D model.
        Args: see model config for details.
        """
        super().__init__()

        self.norm_last_layer = norm_last_layer
        self.ibot_separate_head = ibot_separate_head

        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.sampling = sampling

        self.apply_gin = apply_gin

        self.hidden_size = hidden_size
        self.teacher_backbone = backbone
        # Freeze teacher backbone
        freeze_eval_module(self.teacher_backbone)

        # DINO head
        teacher_dino_head = DINOProjectionHead(
            input_dim=self.hidden_size,
            output_dim=projection_dim,
            norm_last_layer=norm_last_layer,
        )
        self.teacher_dino_head = DINOHead(teacher_dino_head)
        freeze_eval_module(self.teacher_dino_head)

        # Student components
        self.student_backbone = copy.deepcopy(self.teacher_backbone)
        # Unfreeze student
        for param in self.student_backbone.parameters():
            param.requires_grad = True
        self.student_backbone.train()

        student_dino_head = DINOProjectionHead(
            input_dim=self.hidden_size,
            output_dim=projection_dim,
            freeze_last_layer=freeze_last_layer,
            norm_last_layer=norm_last_layer,
        )
        self.student_dino_head = DINOHead(student_dino_head)

        # iBOT heads - separate or shared
        if ibot_separate_head:
            teacher_ibot_head = DINOProjectionHead(
                input_dim=self.hidden_size,
                output_dim=ibot_projection_dim,
                freeze_last_layer=freeze_last_layer,
                norm_last_layer=norm_last_layer,
            )
            self.teacher_ibot_head = IBOTHead(teacher_ibot_head)
            freeze_eval_module(self.teacher_ibot_head)

            student_ibot_head = DINOProjectionHead(
                input_dim=self.hidden_size,
                output_dim=ibot_projection_dim,
                freeze_last_layer=freeze_last_layer,
                norm_last_layer=norm_last_layer,
            )
            self.student_ibot_head = IBOTHead(student_ibot_head)
        else:
            self.teacher_ibot_head = self.teacher_dino_head
            self.student_ibot_head = self.student_dino_head

        self.intensity_aug = Compose(
            [
                RandBiasField(coeff_range=(0.0, 0.5), prob=0.3),
                RandScaleIntensity(factors=(0.8, 1.2), prob=0.5),
                RandShiftIntensity(offsets=(-0.1, 0.1), prob=0.5),
                RandHistogramShift(num_control_points=(5, 15), prob=0.3),
                RandGaussianNoise(mean=0.0, std=0.05, prob=0.3),
                RandGaussianSmooth(sigma_x=(0.5, 1.5), prob=0.3)
            ]
        )


    def forward_teacher(self, x, mask=None):

        # for i in range(x.shape[0]):
        #     low = torch.randint(low=-1000, high=-199, size=(1,)).item()
        #     high = torch.randint(low=200, high=1001, size=(1,)).item()
        #     intensity_scale = ScaleIntensityRange(a_min=low, a_max=high, b_min=0.0, b_max=1.0, clip=True)
        #     x[i] = intensity_scale(x[i])

        features = self.teacher_backbone(x)
        cls_token = features[:, 0]
        features = features if mask is None else features[mask]
        return cls_token, features

    def forward_student(self, x, mask=None):

        for i in range(x.shape[0]):

            # low = torch.randint(low=-1000, high=-199, size=(1,)).item()
            # high = torch.randint(low=200, high=1001, size=(1,)).item()
            # intensity_scale = ScaleIntensityRange(a_min=low, a_max=high, b_min=0.0, b_max=1.0, clip=True)
            # x[i] = intensity_scale(x[i])

            x[i] = self.intensity_aug(x[i])

            if self.apply_gin:
                with torch.autocast(device_type="cuda", enabled=False):  # disables autocast
                    x[i] = gin_aug(x[i].unsqueeze(0).float()).squeeze(0)

        x, flips = random_flip_3d(x)

        features = self.student_backbone(x, mask=mask)

        H, W, D = self.student_backbone.grid_size
        features_3d = features[:, 1:, :].permute(0, 2, 1).view(features.shape[0], self.hidden_size, H, W, D)
        features_3d = undo_flip_3d(features_3d, flips)
        features = features_3d.permute(0, 2, 3, 4, 1).view(features.shape[0], -1, features.shape[-1])

        features = features if mask is None else features[mask[:, 1:]]
        # features = features if mask is None else features[mask]

        features_no_mask = self.student_backbone(x)
        cls_tokens = features_no_mask[:, 0]


        return cls_tokens, features

    def update_teacher(self, global_step: int, max_steps: int, start_value: float , end_value: float) -> None:
        """Update teacher using EMA with cosine momentum schedule."""
        momentum = cosine_schedule(
            step=global_step, max_steps=max_steps, start_value=start_value, end_value=end_value
        )

        # Remove problematic device movement logic
        # In DDP, parameters should already be on correct devices
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_dino_head, self.teacher_dino_head, m=momentum)
        if self.ibot_separate_head:
            update_momentum(self.student_ibot_head, self.teacher_ibot_head, m=momentum)

    def cancel_last_layer_gradients(self, current_epoch: int) -> None:
        """Cancel gradients in the last layer during warmup."""
        self.student_dino_head.cancel_last_layer_gradients(current_epoch)
        if self.ibot_separate_head:
            self.student_ibot_head.cancel_last_layer_gradients(current_epoch)

    def forward(self, views: list[torch.Tensor]):
        """
        Forward pass for DINOv2 3D with block masking and multi-view augmentations.
        Args:
            views: List of augmented 3D tensors (global/local views)
        Returns:
            Dict of teacher/student outputs for loss computation
        """
        device = views[0].device

        # import matplotlib.pyplot as plt
        # for view_0, view_1 in zip(views[0], views[1]):
        #     fig, axes = plt.subplots(1, 2)
        #     axes[0].imshow(view_0[0, :, :, 64].detach().cpu().numpy(), cmap="gray")
        #     axes[1].imshow(view_1[0, :, :, 64].detach().cpu().numpy(), cmap="gray")
        #     plt.show()
        #     plt.close()

        # global_views = torch.cat(views[:2])
        global_views = torch.cat(views[:1])

        if len(views) > 2:
            local_views = torch.cat(views[2:])
        else:
            local_views = None

        # Masking
        B = len(global_views)
        sequence_length = self.student_backbone.sequence_length
        mask = global_views.new_zeros((B, sequence_length), dtype=torch.bool)
        H, W, D = self.student_backbone.grid_size
        assert H == W == D, "Patch size must be cubic for 3D input"
        assert H * W * D == sequence_length - 1, (
            f"Unexpected grid size {H * W * D} ({H}, {W}, {D}) does not match sequence length {sequence_length - 1}"
        )

        if self.sampling == 'random':
            block_masker = RandomBlockMask3D(max_block_size=3, mask_ratio_min=self.mask_ratio_min, mask_ratio_max=self.mask_ratio_max)
            block_mask = block_masker(size=(B, D, H, W), device=device)
            mask[:, 1:] = block_mask.flatten(start_dim=1)
        elif self.sampling == 'entropy':
            with torch.no_grad():
                entropy_map = local_entropy_3d(global_views, kernel_size=9)  # compute texture complexity
                entropy_map = F.interpolate(entropy_map, size=(D, H, W), mode="trilinear", align_corners=False)
                block_mask = sample_entropy_mask(entropy_map, min_mask=self.mask_ratio_min, max_mask=self.mask_ratio_max).squeeze(1)
                mask[:, 1:] = block_mask.flatten(start_dim=1).to(torch.bool)

        # Teacher forward
        with torch.no_grad():
            teacher_cls_token, teacher_patch_tokens = self.forward_teacher(
                global_views, mask=mask
            )
            teacher_cls_token = self.teacher_dino_head(teacher_cls_token)
            teacher_patch_tokens_ibot = self.teacher_ibot_head(teacher_patch_tokens)

        # Student forward
        student_global_cls_token, student_global_patch_tokens = self.forward_student(
            global_views, mask=mask
        )
        student_global_cls_token = self.student_dino_head(student_global_cls_token)
        student_global_patch_tokens_ibot = self.student_ibot_head(
            student_global_patch_tokens
        )

        # Local views
        if local_views is not None:
            student_local_cls_token, _ = self.forward_student(local_views, mask=None)
            student_local_cls_token = self.student_dino_head(student_local_cls_token)
            student_cls_token = torch.cat(
                [student_global_cls_token, student_local_cls_token], dim=0
            )
        else:
            student_cls_token = student_global_cls_token

        out = {
            "teacher_cls_token": teacher_cls_token,
            "student_cls_token": student_cls_token,
            "teacher_patch_tokens_backbone": teacher_patch_tokens,
            "student_patch_tokens_backbone": student_global_patch_tokens,
            "teacher_patch_tokens": teacher_patch_tokens_ibot,
            "student_patch_tokens": student_global_patch_tokens_ibot,
            "student_glob_cls_token": student_global_cls_token,
            "mask": block_mask.to(torch.bool),
            "n_local_views": torch.tensor(len(views) - 2, device=device),
        }

        return {"pred": out}

    def encode(self, x: torch.Tensor):
        """
        Simple encoding method that returns the raw CLS token features.
        Useful for feature extraction or as input to downstream tasks.
        Args:
            x: Input tensor of shape (B, C, D, H, W)
        Returns:
            CLS token features of shape (B, hidden_size)
        """
        backbone_features = self.student_backbone(x, mask=None)
        return backbone_features[:, 0]  # Return CLS token only
