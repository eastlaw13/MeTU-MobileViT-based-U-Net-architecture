import os
import sys
import timm
import wandb
import torch
import warnings
import numpy as np
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import logging

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.iou import iou_component, iou_calculation
from utils.Loss import DiceLoss
from utils.lr_schedule import CosineAnnealingWithWarmupLR

warnings.filterwarnings("ignore")
logging.set_verbosity_warning()
logging.set_verbosity_error()


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


import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchinfo import summary


class MobileViTASPP(nn.Module):
    def __init__(self, in_channels, aspp_out_channels=256, atrous_rates=[6, 12, 18]):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, aspp_out_channels, 1, bias=False),
                    nn.BatchNorm2d(aspp_out_channels),
                    nn.ReLU(inplace=True),
                )
            ]
        )

        for rate in atrous_rates:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        aspp_out_channels,
                        3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(aspp_out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.image_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, aspp_out_channels, 1, bias=False),
            nn.BatchNorm2d(aspp_out_channels),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(aspp_out_channels * 5, aspp_out_channels, 1, bias=False),
            nn.BatchNorm2d(aspp_out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        size = x.shape[-2:]
        res = [conv(x) for conv in self.convs]

        pooled = self.image_pooling(x)
        pooled = F.interpolate(pooled, size=size, mode="bilinear", align_corners=False)
        res.append(pooled)

        res = torch.cat(res, dim=1)
        return self.project(res)


class MobileViT_DLV3(nn.Module):
    def __init__(self, model_size="xxs", encoder_pretrained=True, classes=1):
        super().__init__()

        base_model = timm.create_model(
            f"mobilevit_{model_size}", pretrained=encoder_pretrained
        )

        self.encoder = nn.Sequential(base_model.stem, base_model.stages)
        del base_model

        in_ch = 80 if model_size == "xxs" else 96
        self.classes = classes

        self.aspp = MobileViTASPP(in_channels=in_ch, aspp_out_channels=256)
        self.segmentation_head = nn.Conv2d(256, self.classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]

        x = self.encoder(x)
        x = self.aspp(x)
        x = self.segmentation_head(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        return x


class lt_mobilevit_dlv3(L.LightningModule):
    def __init__(
        self,
        learning_rate,
        model_size: str = "xxs",
        encoder_pretrained: bool = True,
        classes: int = 19,
        ignore_index: int = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.classes = classes
        self.learning_rate = learning_rate
        self.ignore_index = ignore_index
        self.model = MobileViT_DLV3(
            model_size=model_size,
            encoder_pretrained=encoder_pretrained,
            classes=self.classes,
        )
        self._init_iou_components()
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def _init_iou_components(self):
        self.register_buffer(
            "train_intersections", torch.zeros(self.classes, dtype=torch.long)
        )
        self.register_buffer(
            "train_unions", torch.zeros(self.classes, dtype=torch.long)
        )
        self.register_buffer(
            "val_intersections", torch.zeros(self.classes, dtype=torch.long)
        )
        self.register_buffer("val_unions", torch.zeros(self.classes, dtype=torch.long))
        self.register_buffer(
            "test_intersections", torch.zeros(self.classes, dtype=torch.long)
        )
        self.register_buffer("test_unions", torch.zeros(self.classes, dtype=torch.long))

    def _reset_iou(self, stage):
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

    def _step(self, batch):
        imgs, masks = batch
        logits = self(imgs)
        ce_loss = F.cross_entropy(logits, masks, ignore_index=self.ignore_index)
        dice_loss = DiceLoss(logits, masks, self.classes, self.ignore_index)
        loss = 0.5 * ce_loss + 0.5 * dice_loss
        return loss, logits, masks

    def training_step(self, batch, batch_idx):
        loss, logits, masks = self._step(batch)
        preds = logits.argmax(dim=1)
        inter, union = iou_component(
            preds, masks, self.classes, ignore_idx=self.ignore_index
        )

        self.train_unions += union
        self.train_intersections += inter

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"])
        return loss

    def on_train_epoch_end(self):
        m_iou, _ = iou_calculation(self.train_intersections, self.train_unions)
        self.log("train/mIoU", m_iou, on_epoch=True)
        self._reset_iou("train")

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.first_val_batch = (batch[0].cpu(), batch[1].cpu())

        loss, logits, masks = self._step(batch)
        preds = logits.argmax(dim=1)
        inter, union = iou_component(
            preds, masks, self.classes, ignore_idx=self.ignore_index
        )

        self.val_intersections += inter
        self.val_unions += union

        self.validation_step_outputs.append(loss)

    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            avg_loss = torch.stack(self.validation_step_outputs).mean()
            self.log("val/loss", avg_loss, on_epoch=True)
            self.validation_step_outputs.clear()

        m_iou, _ = iou_calculation(self.val_intersections, self.val_unions)
        self.log("val/mIoU", m_iou, on_epoch=True)

        if hasattr(self, "first_val_batch") and self.first_val_batch:
            img, mask = self.first_val_batch
            self.model.eval()
            with torch.no_grad():
                pred = self.model(img[0:1].to(self.device)).argmax(dim=1).cpu()
            self.model.train()

            gt_vis = apply_color_map(mask[0], self.classes)
            pred_vis = apply_color_map(pred[0], self.classes)

            w, h = gt_vis.size
            combined = Image.new("RGB", (w * 2, h))
            combined.paste(gt_vis, (0, 0))
            combined.paste(pred_vis, (w, 0))

            self.logger.experiment.log(
                {
                    "val/visualize": wandb.Image(
                        combined, caption=f"Epoch {self.current_epoch}"
                    )
                }
            )
            self.first_val_batch = None

        self._reset_iou("val")

    def test_step(self, batch, batch_idx):
        loss, logits, masks = self._step(batch)
        preds = logits.argmax(dim=1)
        inter, union = iou_component(
            preds, masks, self.classes, ignore_idx=self.ignore_index
        )

        self.test_intersections += inter
        self.test_unions += union

        self.test_step_outputs.append(loss)
        return loss

    def on_test_epoch_end(self):
        if self.test_step_outputs:
            avg_loss = torch.stack(self.test_step_outputs).mean()
            self.log("test/loss", avg_loss, on_epoch=True)
            self.test_step_outputs.clear()

        self._reset_iou("test")

    def configure_optimizers(self):
        encoder_params, decoder_params = [], []
        for name, param in self.model.named_parameters():
            if name.startswith("encoder"):
                encoder_params.append((name, param))
            else:
                decoder_params.append((name, param))

        def split_wd(params, lr):
            wd, nwd = [], []
            for name, p in params:
                if p.ndim <= 1 or "bias" in name or "norm" in name.lower():
                    nwd.append(p)
                else:
                    wd.append(p)
            return [
                {"params": wd, "weight_decay": 1e-4, "lr": lr},
                {"params": nwd, "weight_decay": 0.0, "lr": lr},
            ]

        groups = split_wd(encoder_params, self.learning_rate / 5) + split_wd(
            decoder_params, self.learning_rate
        )
        optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999))
        scheduler = CosineAnnealingWithWarmupLR(
            optimizer, T_max=200, eta_min=1e-6, warmup_epochs=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }


if __name__ == "__main__":
    from torchinfo import summary

    model = lt_mobilevit_dlv3(
        learning_rate=5e-04,
        model_size="xs",
        encoder_pretrained=True,
        classes=19,
        ignore_index=255,
    )

    summary(model, (1, 3, 512, 1024))
