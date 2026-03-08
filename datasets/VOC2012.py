import json
import torch
import numpy as np

from PIL import Image
from typing import Union
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

cfg_path = Path("../../data/VOC2012/eda/cfg/meta_info.json")
weights_pt_path = Path("../../data/VOC2012/eda/cfg/train_sampler_weights.pt")

try:
    with cfg_path.open(mode="r", encoding="utf-8") as f:
        CFG = json.load(f)
except FileNotFoundError:
    print(f"Error: File not found. Please check the path: {cfg_path}")
except json.JSONDecodeError:
    print(f"Error: The format of the JSON file is invalid: {cfg_path}")
except Exception as e:
    print(f"An error occurred while processing the file: {e}")


class VOC2012(Dataset):
    """
    PASCAL VOC 2012 Dataset.

    Args:
        mode(str): "train", "val", "test"
        transforms: torchvision.transforms.v2 Compose object
    """

    def __init__(self, mode: str = "train", transforms=None):
        super().__init__()
        self.mode = mode
        self.transforms = transforms
        self.image_dir = Path("../../data/VOC2012/images") / self.mode
        self.mask_dir = Path("../../data/VOC2012/masks") / self.mode
        self.image_path = sorted(list(self.image_dir.rglob("*.jpg")))
        self.mask_path = sorted(list(self.mask_dir.rglob("*.png")))

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        image = Image.open(self.image_path[index]).convert("RGB")
        mask = Image.open(self.mask_path[index]).convert("L")

        image = tv_tensors.Image(image)
        mask = tv_tensors.Mask(mask)

        if self.transforms:
            image, mask = self.transforms(image, mask)

        mask = torch.as_tensor(mask, dtype=torch.long)

        if mask.ndim == 3:
            mask = mask.squeeze(0)

        return image, mask


class TrainTransforms:
    def __init__(self, size=(512, 512)):
        self.transforms = v2.Compose(
            [
                v2.RandomResizedCrop(size=size, scale=(0.5, 2.0), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
                        )
                    ],
                    p=0.5,
                ),
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.3
                ),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=CFG["train"]["mean"], std=CFG["train"]["std"]),
            ]
        )

    def __call__(self, image, mask):
        return self.transforms(image, mask)


class ValTransforms:
    def __init__(self, size=512):
        self.transforms = v2.Compose(
            [
                v2.Resize(size, antialias=True),
                v2.CenterCrop(size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=CFG["train"]["mean"], std=CFG["train"]["std"]),
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
