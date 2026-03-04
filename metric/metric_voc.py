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
from datasets.VOC2012 import VOC2012, ValTransforms
from utils.iou import iou_calculation, iou_component


voc2012_classes = {
    "background": 0.0,
    "aeroplane": 0.0,
    "bicycle": 0.0,
    "bird": 0.0,
    "boat": 0.0,
    "bottle": 0.0,
    "bus": 0.0,
    "car": 0.0,
    "cat": 0.0,
    "chair": 0.0,
    "cow": 0.0,
    "diningtable": 0.0,
    "dog": 0.0,
    "horse": 0.0,
    "motorbike": 0.0,
    "person": 0.0,
    "pottedplant": 0.0,
    "sheep": 0.0,
    "sofa": 0.0,
    "train": 0.0,
    "tvmonitor": 0.0,
}


MODEL_LIST = [
    {
        "name": "MeTU-xxs",
        "model": lt_MeTU,
        "ckpt": "logs/VOC2012/0301/[MeTU-xxs]-v3/best/mIoU=0.670.ckpt",
    },
    {
        "name": "MeTU-xs",
        "model": lt_MeTU,
        "ckpt": "logs/VOC2012/0301/[MeTU-xs]-v3/best/mIoU=0.713.ckpt",
    },
    {
        "name": "Segformer-b0",
        "model": lt_segformerb0,
        "ckpt": "logs/VOC2012/0301/Segformer-b0/best/mIoU=0.624.ckpt",
    },
    {
        "name": "MobileViT + DeepLab V3 - xss",
        "model": lt_mobilevit_dlv3,
        "ckpt": "logs/VOC2012/0301/MobileViT_DLV3-xxs/best/mIoU=0.647.ckpt",
    },
    {
        "name": "MobileViT + DeepLab V3 - xs",
        "model": lt_mobilevit_dlv3,
        "ckpt": "logs/VOC2012/0303/MobileViT_DLV3-xs/best/mIoU=0.692.ckpt",
    },
    {
        "name": "LRASPP-MobileNet V3 -xxs",
        "model": lt_lraspp_mv3,
        "ckpt": "logs/VOC2012/0301/LRASPP_MV3/best/mIoU=0.591.ckpt",
    },
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transfroms = ValTransforms()
dataset = VOC2012(mode="val", transforms=transfroms)
data_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

results_list = []

for model_cfg in MODEL_LIST:
    temp_dict = voc2012_classes.copy()
    keys = list(temp_dict.keys())

    model = model_cfg["model"].load_from_checkpoint(model_cfg["ckpt"])
    model.eval()

    inter_total = torch.zeros(21, dtype=torch.long)
    union_total = torch.zeros(21, dtype=torch.long)

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
            inter, union = iou_component(preds, masks, 21, 255)
            inter_total += inter
            union_total += union

        mIoU, IoU_per_class = iou_calculation(inter_total, union_total)
        IoU_per_class = IoU_per_class.tolist()

        for key, val in zip(keys, IoU_per_class):
            temp_dict[key] = round(float(val), 3)
        mIoU = float(mIoU)

    dummy_input = torch.randn((1, 3, 512, 512)).to(device)
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


save_path = Path("metric/res/VOC")
save_path.mkdir(exist_ok=True, parents=True)

with open(save_path / "voc_eval.json", "w", encoding="utf-8") as f:
    json.dump(results_list, f, indent=4, ensure_ascii=False)

print(f"\n[Done] All results saved to {save_path}")
