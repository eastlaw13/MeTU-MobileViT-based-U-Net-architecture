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
from fvcore.nn import FlopCountAnalysis, parameter_count_table


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


class DWConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1, dilation=1):
        super().__init__()
        effective_padding = padding * dilation
        self.depth = nn.Conv2d(
            in_ch,
            in_ch,
            kernel,
            padding=effective_padding,
            groups=in_ch,
            bias=False,
            dilation=dilation,
        )
        self.point = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        x = self.bn(x)
        return x


class CoordinateAttention(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        _, _, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.bn1(self.conv1(y))
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return identity * a_w * a_h


class RefinementBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dw = DWConv(ch, ch, kernel=3)
        self.act1 = nn.ReLU(inplace=True)

        self.d_conv = nn.Sequential(
            DWConv(ch, ch, dilation=2, padding=1),
        )
        self.ca = CoordinateAttention(ch, ch)

    def forward(self, x):
        res = x
        x_dw = self.act1(self.dw(x))
        x_ca = self.ca(x_dw)
        x_d = self.d_conv(x)

        out = F.relu(x_ca + x_d + res, inplace=True)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_refine=True):
        super().__init__()
        self.use_refine = use_refine
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        fuse_in_ch = in_channels + skip_channels

        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in_ch, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        if self.use_refine:
            self.refine = RefinementBlock(out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )

        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)

        if self.use_refine:
            x = self.refine(x)
        return x


class MeTU(nn.Module):
    def __init__(
        self,
        model_size="xxs",
        encoder_pretrained=True,
        classes=1,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            f"mobilevit_{model_size}", pretrained=encoder_pretrained, features_only=True
        )
        self.classes = classes
        encoder_channels = self.encoder.feature_info.channels()
        in_ch = encoder_channels[-1]

        self.decoder_blocks = nn.ModuleList()

        MIN_CH = 64
        for i, skip_ch in enumerate(encoder_channels[:-1][::-1]):
            out_ch = max(skip_ch, MIN_CH)
            use_refine = True if i < 2 else False
            self.decoder_blocks.append(
                DecoderBlock(in_ch, skip_ch, out_ch, use_refine=use_refine)
            )
            in_ch = out_ch

        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            DWConv(in_ch, in_ch, kernel=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, self.classes, kernel_size=1),
        )

        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.encoder(x)
        x = features[-1]
        skips = features[:-1][::-1]

        for i, dec_block in enumerate(self.decoder_blocks):
            x = dec_block(x, skips[i])

        x = self.final_conv(x)
        return x


class lt_MeTU(L.LightningModule):
    def __init__(
        self,
        learning_rate,
        model_size="xxs",
        encoder_pretrained=True,
        classes=1,
        ingnore_index: int = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.classes = classes
        self.learning_rate = learning_rate
        self.ignore_index = ingnore_index
        self.model = MeTU(model_size, encoder_pretrained, classes)
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

    model = lt_MeTU(
        learning_rate=1e-3,
        model_size="xxs",
        encoder_pretrained=True,
        classes=19,
        ingnore_index=255,
    )
    summary(model, (1, 3, 512, 1024))
