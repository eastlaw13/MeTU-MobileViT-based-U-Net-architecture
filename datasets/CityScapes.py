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
from torchvision.transforms import v2
from torchvision import tv_tensors

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
        image = Image.open(self.image_path[idx]).convert("RGB")
        mask = Image.open(self.mask_path[idx])

        image = tv_tensors.Image(image)
        mask = tv_tensors.Mask(mask)

        if self.transforms:
            image, mask = self.transforms(image, mask)

        mask = torch.as_tensor(mask, dtype=torch.long)

        if mask.ndim == 3:
            mask = mask.squeeze(0)
        return image, mask


class TrainTransforms:
    def __init__(self, size=(512, 1024)):
        self.spatial_transforms = v2.Compose(
            [
                v2.ScaleJitter(target_size=(1024, 2048), scale_range=(0.5, 2.0)),
                v2.RandomCrop(
                    size=size,
                    pad_if_needed=True,
                    fill={tv_tensors.Image: 0, tv_tensors.Mask: 255},
                ),
                v2.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.color_transforms = v2.Compose(
            [
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.5,
                ),
                v2.RandomApply([v2.AugMix(severity=3, mixture_width=3)], p=0.3),
            ]
        )

        self.final_transforms = v2.Compose(
            [
                v2.ToDtype(
                    dtype={
                        tv_tensors.Image: torch.float32,
                        tv_tensors.Mask: torch.int64,
                    },
                    scale=True,
                ),
                v2.Normalize(mean=[0.288, 0.326, 0.285], std=[0.186, 0.189, 0.186]),
            ]
        )

    def __call__(self, image, mask):
        image, mask = self.spatial_transforms(image, mask)
        image = self.color_transforms(image)
        image, mask = self.final_transforms(image, mask)

        return image, mask


class ValTransforms:
    def __init__(self, size=(512, 1024)):
        self.transforms = v2.Compose(
            [
                v2.Resize(size=size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.288, 0.326, 0.285], std=[0.186, 0.189, 0.186]),
            ]
        )

    def __call__(self, image, mask):
        return self.transforms(image, mask)


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
