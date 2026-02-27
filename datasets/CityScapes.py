import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as F

from PIL import Image
from typing import Union
from pathlib import Path
from torch.utils.data import Dataset

cfg_path = Path("../../data/CityScapes/eda/cfg/meta_info.json")
weights_pt_path = Path("../../data/CityScapes/eda/cfg/train_sampler_weights.pt")

try:
    with cfg_path.open(mode="r", encoding="utf-8") as f:
        CFG = json.load(f)
except FileNotFoundError:
    print(f"Error: File not found. Please check the path: {cfg_path}")
except json.JSONDecodeError:
    print(f"Error: The format of the JSON file is invalid: {cfg_path}")
except Exception as e:
    print(f"An error occurred while processing the file: {e}")


class CityScapes(Dataset):
    """
    CityScapes Datasets.

    Args:
        mode(str): "train", "val", "test"
        transforms: Argumentation and transforms
    """

    def __init__(self, mode, transforms=None):
        super().__init__()
        self.mode = mode
        self.transforms = transforms
        self.image_dir = Path("../../data/CityScapes/images") / self.mode
        self.mask_dir = Path("../../data/CityScapes/masks") / self.mode
        self.image_path = list(self.image_dir.rglob("*.png"))
        self.mask_path = list(self.mask_dir.rglob("*.png"))

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        img = Image.open(self.image_path[idx]).convert("RGB")
        msk = Image.open(self.mask_path[idx])

        if self.transforms:
            img, msk = self.transforms(img, msk)

        return img, msk


class TrainTransforms:
    def __init__(
        self,
        crop_size=(512, 1024),
        hflip_prob=0.5,
        color_jitter_prob=0.5,
        gamma_prob=0.3,
        blur_prob=0.2,
        noise_prob=0.2,
        autocontrast_prob=0.3,
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
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        )
        self.gausian_blur = T.GaussianBlur(kernel_size=(5), sigma=(0.1, 2.0))

    def __call__(self, image, mask):

        # Random scaling
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
            gamma = torch.empty(1).uniform_(0.7, 1.3).item()
            image = F.adjust_gamma(image, gamma)

        if torch.rand(1) < self.autocontrast_prob:
            image = F.autocontrast(image)

        blur_noise_rand = torch.rand(1)

        if blur_noise_rand < self.blur_prob:
            image = self.gausian_blur(image)
        elif blur_noise_rand < self.blur_prob + self.noise_prob:
            img_tensor = F.to_tensor(image)
            noise = torch.randn_like(img_tensor) * 0.05
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


def load_weight_sampler(
    weights_path: Union[str, Path] = weights_pt_path,
    num_samples: Union[int, None] = None,
    replacement: bool = True,
) -> torch.utils.data.WeightedRandomSampler:

    try:
        weights_tensor = torch.load(weights_path)
        if weights_tensor.dim() != 1:
            raise ValueError(
                f"Loaded weights tensor must be 1D, but got {weights_tensor.dim()}D. Shape: {weights_tensor.shape}"
            )

        if num_samples is None:
            num_samples = len(weights_tensor)

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights_tensor, num_samples=num_samples, replacement=replacement
        )

        print(f"WeightedRandomSampler successfully created from: {weights_path}")
        print(f"Sampler Size (num_samples): {num_samples}")
        return sampler

    except FileNotFoundError:
        print(f"Error: Weights file not found at: {weights_path}")
        raise
    except Exception as e:
        print(f"An error occurred while loading or creating sampler: {e}")
        raise


if __name__ == "__main__":
    ts_train = TrainTransforms()
    ds_train = CityScapes(mode="train", transforms=ts_train)

    image, mask = ds_train[0]
    if ts_train is not None:
        image = torch.permute(image, (1, 2, 0))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.show()
