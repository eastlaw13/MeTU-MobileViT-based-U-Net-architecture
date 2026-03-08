import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as F

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2


class Carla(Dataset):
    def __init__(self, mode: str = "train", transforms=None):
        super().__init__()
        self.mode = mode
        self.transforms = transforms
        self.image_dir = Path(f"../../data/Carla/transfer/images/{mode}")
        self.mask_dir = Path(f"../../data/Carla/transfer/masks/{mode}")
        self.image_list = sorted(list(self.image_dir.rglob("*.jpg")))
        self.mask_list = sorted(list(self.mask_dir.rglob("*.png")))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert("RGB")
        mask = Image.open(self.mask_list[index]).convert("L")

        image = tv_tensors.Image(image)
        mask = tv_tensors.Mask(mask)

        if self.transforms:
            image, mask = self.transforms(image, mask)

        mask = torch.as_tensor(mask, dtype=torch.long)

        if mask.ndim == 3:
            mask = mask.squeeze(0)

        return image, mask


IGNORE_INDEX = 255


class TrainTransforms:
    def __init__(self, size=(512, 1024)):
        self.transforms = v2.Compose(
            [
                v2.RandomResize(min_size=256, max_size=1024),
                v2.RandomCrop(
                    size=(512, 1024),
                    pad_if_needed=True,
                    fill={torch.Tensor: 0, "tv_tensor": IGNORE_INDEX},
                ),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.3
                ),
                v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.3
                ),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.288, 0.326, 0.285], std=[0.186, 0.189, 0.186]),
            ]
        )

    def __call__(self, image, mask):
        return self.transforms(image, mask)


class ValTransforms:
    def __init__(self, size=512):
        self.transforms = v2.Compose(
            [
                v2.Resize(size, antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.288, 0.326, 0.285], std=[0.186, 0.189, 0.186]),
            ]
        )

    def __call__(self, image, mask):
        return self.transforms(image, mask)
