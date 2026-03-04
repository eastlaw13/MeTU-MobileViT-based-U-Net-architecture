import os
import sys
import json
import torch
import torch.nn.functional as F
import logging

logging.getLogger("fvcore.nn.jit_analysis").setLevel(logging.ERROR)

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from fvcore.nn import FlopCountAnalysis


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.modelzoo import lt_MeTU, lt_segformerb0, lt_mobilevit_dlv3, lt_lraspp_mv3
from datasets.CityScapes import CityScapes, ValTransforms
from utils.iou import iou_calculation, iou_component

Cityscpaes_label = {
    "Road": 0.0,
    "Sidewalk": 0.0,
    "Building": 0.0,
    "Wall": 0.0,
    "Fence": 0.0,
    "Pole": 0.0,
    "Traffic Light": 0.0,
    "Traffic Sign": 0.0,
    "Vegetation": 0.0,
    "Terrain": 0.0,
    "Sky": 0.0,
    "Person": 0.0,
    "Car": 0.0,
    "Truck": 0.0,
    "Train": 0.0,
    "Motorcycle": 0.0,
    "Bicycle": 0.0,
}


MODEL_LIST = [
    {
        "name": "MeTU-xxs",
        "model": lt_MeTU,
        "ckpt": "logs/Cityscapes/0227/[MeTU-xxs]-v3/best/mIoU=0.671.ckpt",
    },
    {
        "name": "MeTU-xs",
        "model": lt_MeTU,
        "ckpt": "logs/Cityscapes/0228/[MeTU-xs]-v3/best/mIoU=0.707.ckpt",
    },
    {
        "name": "Segformer-b0",
        "model": lt_segformerb0,
        "ckpt": "logs/Cityscapes/0225/Segformer-b0/best/mIoU=0.676.ckpt",
    },
    {
        "name": "MobileViT + DeepLab V3 - xss",
        "model": lt_mobilevit_dlv3,
        "ckpt": "logs/Cityscapes/0225/MobileViT-DL_V3-xxs/best/mIoU=0.600.ckpt",
    },
    {
        "name": "MobileViT + DeepLab V3 - xs",
        "model": lt_mobilevit_dlv3,
        "ckpt": "logs/Cityscapes/0226/MobileViT-DL_V3-xs/best/mIoU=0.620.ckpt",
    },
    {
        "name": "LRASPP-MobileNet V3 -xxs",
        "model": lt_lraspp_mv3,
        "ckpt": "logs/Cityscapes/0226/LRASPP_MV3/best/mIoU=0.587.ckpt",
    },
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transfroms = ValTransforms()
dataset = CityScapes(mode="val", transforms=transfroms)
data_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

results_list = []

for model_cfg in MODEL_LIST:
    temp_dict = Cityscpaes_label.copy()
    keys = list(temp_dict.keys())

    model = model_cfg["model"].load_from_checkpoint(model_cfg["ckpt"])
    model.eval()

    inter_total = torch.zeros(19, dtype=torch.long)
    union_total = torch.zeros(19, dtype=torch.long)

    with torch.no_grad():
        for images, masks in tqdm(data_loader):
            images = images.to(device)

            logits = model(images)

            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            preds = logits.argmax(dim=1).squeeze().cpu()
            inter, union = iou_component(preds, masks, 19, 255)
            inter_total += inter
            union_total += union

        mIoU, IoU_per_class = iou_calculation(inter_total, union_total)
        IoU_per_class = IoU_per_class.tolist()

        for key, val in zip(keys, IoU_per_class):
            temp_dict[key] = round(float(val), 3)
        mIoU = float(mIoU)

    dummy_input = torch.randn((1, 3, 512, 1024)).to(device)
    flops = FlopCountAnalysis(model, dummy_input)
    total_flops = flops.total() / 1e9

    result_data = {
        "model_name": model_cfg["name"],
        "ckpt_path": model_cfg["ckpt"],
        "mIoU": round(mIoU, 4),
        "class_iou": temp_dict,
        "flops_g": round(total_flops, 2),
        "input_resolution": [512, 1024],
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
    }
    results_list.append(result_data)


save_path = Path("metric/res/Cityscapes")
save_path.mkdir(exist_ok=True, parents=True)

with open(save_path / "eval.json", "w", encoding="utf-8") as f:
    json.dump(results_list, f, indent=4, ensure_ascii=False)

print(f"\n[Done] All results saved to {save_path}")
