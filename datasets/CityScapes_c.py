import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F

c_img_root = "../../data/CityScapes_C"
gt_root = "../../data/CityScapes_C_gtFine"


class CityscapesCDataset(Dataset):
    def __init__(self, corruption, severity, transform=None):
        self.img_dir = os.path.join(
            c_img_root, corruption, str(severity), "leftImg8bit", "val"
        )
        self.gt_dir = os.path.join(gt_root, "val")
        self.images = []
        self.masks = []
        self.transform = transform

        for city in sorted(os.listdir(self.img_dir)):
            city_img_dir = os.path.join(self.img_dir, city)
            city_gt_dir = os.path.join(self.gt_dir, city)
            if not os.path.isdir(city_img_dir):
                continue

            for img_name in sorted(os.listdir(city_img_dir)):
                if img_name.endswith(".png"):
                    self.images.append(os.path.join(city_img_dir, img_name))
                    mask_name = img_name.replace(
                        "_leftImg8bit.png", "_gtFine_labelIds.png"
                    )
                    self.masks.append(os.path.join(city_gt_dir, mask_name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        msk = Image.open(self.masks[idx])

        if self.transform:
            img, msk = self.transform(img, msk)
        return img, msk


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
