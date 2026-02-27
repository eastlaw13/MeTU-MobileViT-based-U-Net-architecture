import os
import sys
import wandb
import torch
import numpy as np
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from PIL import Image
from torchinfo import summary
from transformers import SegformerForSemanticSegmentation, logging

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.iou import iou_component, iou_calculation
from utils.Loss import DiceLoss, FocalLoss
from utils.lr_schedule import CosineAnnealingWithWarmupLR


def generate_color_palette(num_classes: int) -> np.ndarray:
    palette = []

    for i in range(num_classes):
        hue = (i * 360) // num_classes
        rgb_img = Image.new("HSV", (1, 1), (hue, 255, 128)).convert("RGB")
        r, g, b = [int(x) for x in rgb_img.getpixel((0, 0))]
        palette.extend([r, g, b])

    palette_np = np.array(palette, dtype=np.uint8)
    full_palette = np.zeros(256 * 3, dtype=np.uint8)
    full_palette[: len(palette)] = palette_np

    return full_palette


def apply_color_map(mask_tensor: torch.Tensor, num_classes: int) -> Image.Image:
    mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
    mask_img = Image.fromarray(mask_np, mode="L")
    palette = generate_color_palette(num_classes)
    mask_img.putpalette(palette)
    return mask_img.convert("RGB")


class SegFormerb0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0", num_labels=self.num_classes, ignore_mismatched_sizes=True
        )

    def forward(self, x):
        outputs = self.model(x)
        logits = outputs.logits

        logits = F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

        return logits


class lt_segformerb0(L.LightningModule):
    def __init__(
        self, learning_rate: float, classes: int = 19, ignore_index: int = 255
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.classes = classes
        self.ignore_index = ignore_index
        self.save_hyperparameters()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self._init_iou_components()

        self.model = SegFormerb0(num_classes=self.classes)

    def _init_iou_components(self):
        num_classes = self.classes
        self.register_buffer(
            "train_intersections", torch.zeros(num_classes, dtype=torch.long)
        )
        self.register_buffer("train_unions", torch.zeros(num_classes, dtype=torch.long))

        self.register_buffer(
            "val_intersections", torch.zeros(num_classes, dtype=torch.long)
        )
        self.register_buffer("val_unions", torch.zeros(num_classes, dtype=torch.long))

        self.register_buffer(
            "test_intersections", torch.zeros(num_classes, dtype=torch.long)
        )
        self.register_buffer("test_unions", torch.zeros(num_classes, dtype=torch.long))

    def _reset_iou_components(self, stage):
        if stage == "train":
            self.train_intersections.zero_()
            self.train_unions.zero_()
        elif stage == "val":
            self.val_intersections.zero_()
            self.val_unions.zero_()
        elif stage == "test":
            self.test_intersections.zero_()
            self.test_unions.zero_()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)

        ce_loss = F.cross_entropy(logits, masks, ignore_index=self.ignore_index)
        dice_loss = DiceLoss(logits, masks, self.classes, self.ignore_index)

        loss = 0.5 * ce_loss + 0.5 * dice_loss

        preds = logits.argmax(dim=1)
        inter, union = iou_component(
            preds, masks, self.classes, ignore_idx=self.ignore_index
        )

        self.train_intersections += inter
        self.train_unions += union

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"])

        return loss

    def on_train_epoch_end(self):
        mIoU, _ = iou_calculation(self.train_intersections, self.train_unions)
        self.log("train/mIoU", mIoU, on_epoch=True)
        self._reset_iou_components("train")

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)

        if batch_idx == 0:
            self.first_val_batch = (images.cpu(), masks.cpu())

        ce_loss = F.cross_entropy(logits, masks, ignore_index=self.ignore_index)
        dice_loss = DiceLoss(logits, masks, self.classes, self.ignore_index)

        loss = 0.5 * ce_loss + 0.5 * dice_loss

        preds = logits.argmax(dim=1)
        inter, union = iou_component(
            preds, masks, self.classes, ignore_idx=self.ignore_index
        )

        self.val_intersections += inter.to(self.device)
        self.val_unions += union.to(self.device)

        self.validation_step_outputs.append({"loss": loss.detach()})

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        self.log("val/loss", avg_loss, on_epoch=True)

        mIoU, _ = iou_calculation(self.val_intersections, self.val_unions)

        if self.first_val_batch:
            first_img_tensor, first_mask_tensor = self.first_val_batch

            self.model.eval()
            with torch.no_grad():
                logits = self.model(first_img_tensor[0:1].to(self.device))
                preds = torch.argmax(logits, dim=1).cpu()
            self.model.train()

            gt_mask_color = apply_color_map(first_mask_tensor[0], self.classes)
            pred_mask_color = apply_color_map(preds[0], self.classes)

            w, h = gt_mask_color.size
            combined = Image.new("RGB", (w * 2, h))

            combined.paste(gt_mask_color, (0, 0))
            combined.paste(pred_mask_color, (w, 0))

            self.logger.experiment.log(
                {
                    "val/visualize": wandb.Image(
                        combined,
                        caption=f"Epoch {self.current_epoch}: Ground Truth | Prediction",
                    ),
                }
            )

            self.first_val_batch = None

        self.log("val/mIoU", mIoU, on_epoch=True)
        self._reset_iou_components("val")

    def test_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)

        ce_loss = F.cross_entropy(logits, masks, ignore_index=self.ignore_index)
        dice_loss = DiceLoss(logits, masks, self.classes, self.ignore_index)

        loss = 0.5 * ce_loss + 0.5 * dice_loss

        preds = logits.argmax(dim=1)

        self.test_step_outputs.append({"loss": loss.detach()})
        inter, union = iou_component(
            preds, masks, self.classes, ignore_idx=self.ignore_index
        )

        self.test_intersections += inter
        self.test_unions += union
        return loss

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
        avg_loss = torch.stack([x["loss"] for x in self.test_step_outputs]).mean()

        self.log("test/loss", avg_loss, on_epoch=True)
        self._reset_iou_components("test")

    def configure_optimizers(self):
        encoder_params = []
        decoder_params = []

        for name, param in self.model.named_parameters():
            if "encoder" in name:
                encoder_params.append((name, param))
            else:
                decoder_params.append((name, param))

        def split_wd(param_list, base_lr):
            wd, nwd = [], []
            for name, p in param_list:
                if p.ndim <= 1 or "bias" in name or "norm" in name.lower():
                    nwd.append(p)
                else:
                    wd.append(p)

            return [
                {"params": wd, "weight_decay": 1e-4, "lr": base_lr},
                {"params": nwd, "weight_decay": 0.0, "lr": base_lr},
            ]

        optim_gropups = []
        optim_gropups += split_wd(encoder_params, self.learning_rate / 5)
        optim_gropups += split_wd(decoder_params, self.learning_rate)

        optimizer = torch.optim.AdamW(optim_gropups, betas=(0.9, 0.999))
        scheduler = CosineAnnealingWithWarmupLR(
            optimizer, T_max=200, eta_min=1e-6, warmup_epochs=10
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }


if __name__ == "__main__":
    model = lt_segformerb0(learning_rate=5e-04)
    summary(model, (1, 3, 512, 1024))
