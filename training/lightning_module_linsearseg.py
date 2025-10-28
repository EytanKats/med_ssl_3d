import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from dynamic_network_architectures.building_blocks.patch_encode_decode import LayerNormNd, PatchDecode

from training.lightning_module import DINOv2_3D_LightningModule


# ------------------------------------------------------------------
# Simple segmentation head
# ------------------------------------------------------------------
class SegmentationHead(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_dim=256):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


# ------------------------------------------------------------------
# Lightning Module for Segmentation
# ------------------------------------------------------------------
class SegmentationModule(pl.LightningModule):
    def __init__(
        self,
        ckpt_path: str,
        num_classes: int = 5,
        lr: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained PRIMUS backbone (frozen)
        ckpt = DINOv2_3D_LightningModule.load_from_checkpoint(ckpt_path)
        self.backbone = ckpt.model.student_backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Segmentation head
        self.seg_head = SegmentationHead(in_channels=180, hidden_dim=40, out_channels=num_classes)
        # self.seg_head = PatchDecode(
        #     self.backbone.patch_size,
        #     180,
        #     num_classes,
        #     norm=LayerNormNd,
        #     activation=nn.GELU,
        # )

        # Loss & metric
        self.loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")

        self.lr = lr

    def forward(self, x):

        with torch.no_grad():
            feats = self.backbone(x)

        H, W, D = x.shape[-3] // self.backbone.patch_size[-3], x.shape[-2] // self.backbone.patch_size[-2], x.shape[-1] // self.backbone.patch_size[-1]
        feats = feats[:, 1:, :].permute(0, 2, 1).view(feats.shape[0], 180, H, W, D)

        logits = self.seg_head(feats)
        logits = F.interpolate(logits, size=x.shape[2:], mode="trilinear", align_corners=False)
        return logits

    def training_step(self, batch, batch_idx):

        x, y = batch["image"], batch["label"]

        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch["image"], batch["label"]

        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(F.softmax(logits, dim=1), dim=1, keepdim=True)
        self.dice_metric(preds, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        return {"val_loss": loss}

    def on_validation_epoch_end(self):

        dice_score = self.dice_metric.aggregate().item()
        self.log("val_dice", dice_score, prog_bar=True)
        self.dice_metric.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.seg_head.parameters(), lr=self.lr)
