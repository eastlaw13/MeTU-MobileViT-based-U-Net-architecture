import json
import re
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as F

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


def natural_sort_key(path: Path):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", path.stem)
    ]


class Carla(Dataset):
    def __init__(self, mode, transforms=None):
        super().__init__()
        self.mode = mode
        self.transforms = transforms
        self.image_dir = Path(f"../../data/Carla/0.transfer/{mode}/images")
        self.mask_dir = Path(f"../../data/Carla/0.transfer/{mode}/masks")
        self.image_list = sorted(
            list(self.image_dir.rglob("*.jpg")), key=natural_sort_key
        )
        self.mask_list = sorted(
            list(self.mask_dir.rglob("*.png")), key=natural_sort_key
        )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx]).convert("RGB")
        mask = Image.open(self.mask_list[idx])

        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask


class TrainTransforms:

    def __init__(
        self,
        crop_size=(512, 1024),
        hflip_prob=0.5,
        color_jitter_prob=0.4,
        gamma_prob=0.3,
        blur_prob=0.1,
        noise_prob=0.1,
        autocontrast_prob=0.2,
    ):
        self.min_scale = 0.5
        self.max_scale = 2.0
        self.crop_size = crop_size
        self.hflip_prob = hflip_prob
        self.color_jitter_prob = color_jitter_prob
        self.gamma_prob = gamma_prob
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob
        self.autocontrast_prob = autocontrast_prob

        self.color_jitter = T.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01
        )
        self.gausian_blur = T.GaussianBlur(kernel_size=(5), sigma=(0.1, 1.5))

    def __call__(self, image, mask):
        scale = random.uniform(self.min_scale, self.max_scale)
        target_h, target_w = int(1024 * scale), int(2048 * scale)

        image = F.resize(
            image, (target_h, target_w), interpolation=T.InterpolationMode.BILINEAR
        )
        mask = F.resize(
            mask, (target_h, target_w), interpolation=T.InterpolationMode.NEAREST
        )

        if target_h < self.crop_size[0] or target_w < self.crop_size[1]:
            pad_h = max(0, self.crop_size[0] - target_h)
            pad_w = max(0, self.crop_size[1] - target_w)
            image = F.pad(image, (0, 0, pad_w, pad_h), fill=(73, 83, 73))
            mask = F.pad(mask, (0, 0, pad_w, pad_h), fill=255)

        i, j, h, w = T.RandomCrop.get_params(image, output_size=self.crop_size)
        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        if torch.rand(1) < self.hflip_prob:
            image = F.hflip(image)
            mask = F.hflip(mask)

        if torch.rand(1) < self.color_jitter_prob:
            image = self.color_jitter(image)

        if torch.rand(1) < self.gamma_prob:
            gamma = torch.empty(1).uniform_(0.8, 1.2).item()
            image = F.adjust_gamma(image, gamma)

        if torch.rand(1) < self.autocontrast_prob:
            image = F.autocontrast(image)

        blur_noise_rand = torch.rand(1)
        if blur_noise_rand < self.blur_prob:
            image = self.gausian_blur(image)
        elif blur_noise_rand < self.blur_prob + self.noise_prob:
            img_tensor = F.to_tensor(image)
            noise = torch.randn_like(img_tensor) * 0.03
            img_tensor = torch.clamp(img_tensor + noise, 0.0, 1.0)
            image = F.to_pil_image(img_tensor)

        image = F.to_tensor(image)
        image = F.normalize(
            image, mean=[0.288, 0.326, 0.285], std=[0.186, 0.189, 0.186]
        )

        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask


class ValTransforms:

    def __init__(self, target_size=(512, 1024)):
        self.target_size = target_size

    def __call__(self, image, mask):
        image = F.resize(
            image, self.target_size, interpolation=T.InterpolationMode.BILINEAR
        )
        mask = F.resize(
            mask, self.target_size, interpolation=T.InterpolationMode.NEAREST
        )

        image = F.to_tensor(image)
        image = F.normalize(
            image, mean=[0.288, 0.326, 0.285], std=[0.186, 0.189, 0.186]
        )

        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask


if __name__ == "__main__":
    transforms = ValTransforms()
    ds_train = Carla(mode="train", transforms=transforms)

    image, mask = ds_train[1]
    if transforms is not None:
        image = torch.permute(image, (1, 2, 0))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.show()
