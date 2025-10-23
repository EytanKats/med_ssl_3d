import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from lightly.loss import KoLeoLoss
from lightly.models.modules.center import Center, CENTER_MODE_TO_FUNCTION, center_momentum
from lightly.utils.scheduler import linear_warmup_schedule


# References:
#     https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss
#     https://github.com/lightly-ai/lightly/blob/master/benchmarks/imagenet/vitb16/dinov2.py


class DINOLoss(torch.nn.Module):
    """Implementation of the loss described in 'Emerging Properties in
    Self-Supervised Vision Transformers'. [0]

    This implementation follows the code published by the authors. [1]
    It supports global and local image crops. A linear warmup schedule for the
    teacher temperature is implemented to stabilize training at the beginning.
    Centering is applied to the teacher output to avoid model collapse.

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294
    - [1]: https://github.com/facebookresearch/dino

    Attributes:
        output_dim:
            Dimension of the model output.
        teacher_temp:
            Temperature parameter for the teacher network.
        student_temp:
            Temperature parameter for the student network.
        center:
            Center used for the teacher output. It is updated with a moving average
            during training.
        center_momentum:
            Momentum term for the center calculation.
        warmup_teacher_temp_epochs:
                Number of epochs for the warmup phase of the teacher temperature (for backward compatibility).
        teacher_temp_schedule:
            A linear schedule for the teacher temperature during the warmup phase (for backward compatibility).
    """

    def __init__(
        self,
        output_dim: int = 65536,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 30,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        center_mode: str = "mean",
    ) -> None:

        """Initializes the DINOLoss Module.

        Args:
            center_mode:
                Mode for center calculation. Only 'mean' is supported.
            warmup_teacher_temp:
                Initial temperature for the teacher network (for backward compatibility).
            warmup_teacher_temp_epochs:
                Number of epochs for the warmup phase of the teacher temperature (for backward compatibility).
        """
        super().__init__()

        self.teacher_temp = teacher_temp
        self.student_temp = student_temp

        # TODO(Guarin, 08/24): Refactor this to use the Center module directly once
        # we do a breaking change.
        if center_mode not in CENTER_MODE_TO_FUNCTION:
            raise ValueError(
                f"Unknown mode '{center_mode}'. Valid modes are "
                f"{sorted(CENTER_MODE_TO_FUNCTION.keys())}."
            )
        self._center_fn = CENTER_MODE_TO_FUNCTION[center_mode]
        self.center: torch.nn.Parameter
        self.register_buffer("center", torch.zeros(1, 1, output_dim))
        self.center_momentum = center_momentum

        # comput the warmup teacher temperature internally for backward compatibility
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.teacher_temp_schedule = torch.linspace(
            start=warmup_teacher_temp,
            end=teacher_temp,
            steps=warmup_teacher_temp_epochs,
        )

    def forward(
        self,
        teacher_out: list[Tensor],
        student_out: list[Tensor],
        teacher_temp: float | None = None,
        epoch: int | None = None,
    ) -> Tensor:
        """Cross-entropy between softmax outputs of the teacher and student networks.

        Args:
            teacher_out:
                List of tensors with shape (batch_size, output_dim) containing features
                from the teacher model. Each tensor must represent one view of the
                batch.
            student_out:
                List of tensors with shape (batch_size, output_dim) containing features
                from the student model. Each tensor must represent one view of the
                batch.
            teacher_temp:
                The temperature used for the teacher output. If None, the default
                temperature defined in __init__ is used.
            epoch:
                The current epoch for backward compatibility.

        Returns:
            The average cross-entropy loss.
        """

        # Get teacher temperature
        if teacher_temp is not None:
            teacher_temperature = torch.tensor(teacher_temp)
        elif epoch is not None:  # for backward compatibility
            if epoch < self.warmup_teacher_temp_epochs:
                teacher_temperature = self.teacher_temp_schedule[epoch]
            else:
                teacher_temperature = torch.tensor(self.teacher_temp)
        else:
            teacher_temperature = torch.tensor(self.teacher_temp)

        # Calculate cross-entropy loss.
        teacher_out_stacked = torch.stack(teacher_out)
        t_out: Tensor = F.softmax(
            (teacher_out_stacked - self.center) / teacher_temperature, dim=-1
        )
        student_out_stacked = torch.stack(student_out)
        s_out = F.log_softmax(student_out_stacked / self.student_temp, dim=-1)

        # Calculate feature similarities, ignoring the diagonal
        # b = batch_size, t = n_views_teacher, s = n_views_student, d = output_dim
        loss = -torch.einsum("tbd,sbd->ts", t_out, s_out)
        # loss.fill_diagonal_(0)

        # Number of loss terms, ignoring the diagonal
        # n_terms = loss.numel() - loss.diagonal().numel()
        batch_size = teacher_out_stacked.shape[1]

        loss = loss.sum() / batch_size

        # Update the center used for the teacher output
        self.update_center(teacher_out_stacked)

        return loss

    @torch.no_grad()
    def update_center(self, teacher_out: Tensor) -> None:
        """Moving average update of the center used for the teacher output.

        Args:
            teacher_out:
                Tensor with shape (num_views, batch_size, output_dim) containing
                features from the teacher model.
        """

        # Calculate the batch center using the specified center function
        batch_center = self._center_fn(x=teacher_out, dim=(0, 1))

        # Update the center with a moving average
        self.center.data = center_momentum(
            center=self.center, batch_center=batch_center, momentum=self.center_momentum
        )


class IBOTPatchLoss(torch.nn.Module):
    """Implementation of the iBOT patch loss [0] as used in DINOv2 [1].

    Implementation is based on [2].

    - [0]: iBOT, 2021, https://arxiv.org/abs/2111.07832
    - [1]: DINOv2, 2023, https://arxiv.org/abs/2304.07193
    - [2]: https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/ibot_patch_loss.py

    Attributes:
        output_dim:
            Dimension of the model output.
        teacher_temp:
            Temperature for the teacher output.
        student_temp:
            Temperature for the student output.
        center_mode:
            Mode for center calculation. Only 'mean' is supported.
        center_momentum:
            Momentum term for the center update.
    """

    def __init__(
        self,
        output_dim: int = 65536,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_mode: str = "mean",
        center_momentum: float = 0.9,
    ) -> None:
        """Initializes the iBOTPatchLoss module with the specified parameters."""
        super().__init__()

        self.teacher_temp = teacher_temp
        self.student_temp = student_temp

        self.center = Center(
            size=(1, output_dim),
            mode=center_mode,
            momentum=center_momentum,
        )

    def forward(
        self,
        teacher_out: Tensor,
        student_out: Tensor,
        mask: Tensor,
        teacher_temp: float | None = None,
    ) -> Tensor:
        """Forward pass through the iBOT patch loss.

        Args:
            teacher_out:
                Tensor with shape (batch_size * sequence_length, embed_dim) containing
                the teacher output of the masked tokens.
            student_out:
                Tensor with shape (batch_size * sequence_length, embed_dim) containing
                the student output of the masked tokens.
            mask:
                Boolean tensor with shape (batch_size, height, width) containing the
                token mask. Exactly batch_size * sequence_length entries must be set to
                True in the mask.
            teacher_temp:
                The temperature used for the teacher output. If None, the default
                temperature defined in __init__ is used.

        Returns:
            The loss value.
        """
        # B = batch size, N = sequence length = number of masked tokens, D = embed dim
        # H = height (in tokens), W = width (in tokens)
        # Note that N <= H * W depending on how many tokens are masked.
        teacher_temperature = torch.tensor(
            teacher_temp if teacher_temp is not None else self.teacher_temp
        )

        # Calculate cross-entropy loss.
        teacher_softmax = F.softmax(
            (teacher_out - self.center.value) / teacher_temperature, dim=-1
        )
        student_log_softmax = F.log_softmax(student_out / self.student_temp, dim=-1)

        # (B * N, D) -> (B * N)
        loss = -torch.sum(teacher_softmax * student_log_softmax, dim=-1)

        # Get weights.
        # (B, H, W) -> (B, 1, 1)
        num_masked_per_image = mask.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
        # (B, 1, 1) -> (B, H, W) -> (B * N)
        weight = (1.0 / num_masked_per_image).expand_as(mask)[mask]

        # Apply weighting.
        B = mask.shape[0]
        loss = (loss * weight).sum() / B

        self.center.update(teacher_out)

        return loss


class DINOv2Loss(nn.Module):
    def __init__(
        self,
        student_temp: float = 0.1,
        teacher_temp_min: float = 0.04,
        teacher_temp_max: float = 0.07,
        teacher_temp_warmup_epochs: int = 30,
        center_momentum: float = 0.9,
        output_dim: int = 65536,
        ibot_output_dim: int = 8192,
        ibot_loss_weight: float = 1.0,
        koleo_loss_weight: float = 0.1,
        max_steps: int = 1000,
        max_epochs: int = 500,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp_min = teacher_temp_min
        self.teacher_temp_max = teacher_temp_max
        self.teacher_temp_warmup_epochs = teacher_temp_warmup_epochs
        self.center_momentum = center_momentum
        self.output_dim = output_dim
        self.ibot_output_dim = ibot_output_dim
        self.w_ibot = ibot_loss_weight
        self.w_koleo = koleo_loss_weight
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        # Initialize loss functions with dynamic temperature
        self.dino_loss_fn = DINOLoss(
            output_dim=self.output_dim,
            teacher_temp=self.teacher_temp_min,  # Will be updated dynamically
            student_temp=self.student_temp,
            center_momentum=self.center_momentum,
        )

        self.ibot_loss_fn = (
            IBOTPatchLoss(
                output_dim=self.ibot_output_dim,
                teacher_temp=self.teacher_temp_min,  # Will be updated dynamically
                student_temp=self.student_temp,
                center_momentum=self.center_momentum,
            )
            if self.w_ibot > 0
            else None
        )

        self.koleo_loss_fn = KoLeoLoss() if self.w_koleo > 0 else None

    def get_teacher_temperature(self, global_step: int) -> float:
        """Calculate teacher temperature with linear warmup using PyTorch Lightning trainer info."""

        warmup_steps = int(
            (self.teacher_temp_warmup_epochs / self.max_epochs) * self.max_steps
        )

        return linear_warmup_schedule(
            step=global_step,
            warmup_steps=warmup_steps,
            start_value=self.teacher_temp_min,
            end_value=self.teacher_temp_max,
        )

    def forward(self, input_dict, global_step: int = 0):
        """
        input_dict: {
            "teacher_cls_out": Tensor,
            "student_cls_out": Tensor,
            "teacher_mask_patches": Tensor (optional),
            "student_mask_glob_patches": Tensor (optional),
            "student_glob_cls_token": Tensor (optional),
            "mask": Tensor (optional, for patch loss),
            "n_local_views": int (optional, number of local views)
        }
        Computes the DINOV2 loss for 3D data with dynamic teacher temperature
        """

        teacher_outputs = input_dict.get("teacher_cls_token", None)
        student_outputs = input_dict.get("student_cls_token", None)
        teacher_patches = input_dict.get("teacher_patch_tokens", None)
        student_patches = input_dict.get("student_patch_tokens", None)
        teacher_patches_backbone = input_dict.get("teacher_patch_tokens_backbone", None)
        student_patches_backbone = input_dict.get("student_patch_tokens_backbone", None)
        student_glob_cls_token = input_dict.get("student_glob_cls_token", None)
        mask = input_dict.get("mask", None)
        n_local_views = input_dict.get("n_local_views", 0)

        if student_outputs is None or teacher_outputs is None:
            raise ValueError(
                "Both student_outputs and teacher_outputs must be provided."
            )

        # Calculate dynamic teacher temperature
        teacher_temp = self.get_teacher_temperature(global_step)

        # Update temperature in loss functions
        self.dino_loss_fn.teacher_temp = teacher_temp
        if self.ibot_loss_fn is not None:
            self.ibot_loss_fn.teacher_temp = teacher_temp

        # Convert n_local_views to int if it's a tensor
        if isinstance(n_local_views, torch.Tensor):
            n_local_views = n_local_views.item()

        # DINO loss - proper chunking based on views
        n_views = n_local_views + 2  # 2 global + n local views
        dino_loss = self.dino_loss_fn(
            teacher_out=[teacher_outputs],  # .chunk(2),  # 2 global teacher views
            student_out=[student_outputs],  # .chunk(n_views),  # All student views
            teacher_temp=teacher_temp,
        )

        # Patch-level iBOT loss
        ibot_loss = (
            self.ibot_loss_fn(
                teacher_out=teacher_patches,
                student_out=student_patches,
                mask=mask,
                teacher_temp=teacher_temp,
            )
            if self.ibot_loss_fn is not None
            else 0
        )

        koleo_loss = self.koleo_loss_fn(student_glob_cls_token)

        losses = {
            "dino_loss": dino_loss,
            "ibot_loss": ibot_loss,
            "koleo_loss": koleo_loss,
            "total_loss": dino_loss
            + self.w_ibot * ibot_loss
            + self.w_koleo * koleo_loss,
            "teacher_temp": teacher_temp,
        }

        return losses
