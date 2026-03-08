import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np

import onnx
from onnxconverter_common import float16

from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    CalibrationMethod,
)

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models.modelzoo import (
    lt_MeTU,
    lt_segformerb0,
    lt_mobilevit_dlv3,
    lt_lraspp_mv3,
)

from datasets.CityScapes import ValTransforms
from utils.iou import iou_calculation, iou_component


MODEL_LIST = [
    {
        "name": "MeTU-xxs",
        "model": lt_MeTU,
        "ckpt": "ckpt/MeTU_xxs/mIoU=0.685.ckpt",
    },
    {
        "name": "MeTU-xs",
        "model": lt_MeTU,
        "ckpt": "ckpt/MeTU_xs/mIoU=0.735.ckpt",
    },
    {
        "name": "Segformer-b0",
        "model": lt_segformerb0,
        "ckpt": "ckpt/Segformer_b0/mIoU=0.705.ckpt",
    },
    {
        "name": "MobileViT-xxs + DeepLab V3",
        "model": lt_mobilevit_dlv3,
        "ckpt": "ckpt/MobileViT_xxs_DLV3/mIoU=0.615.ckpt",
    },
    {
        "name": "MobileViT-xs + DeepLab V3",
        "model": lt_mobilevit_dlv3,
        "ckpt": "ckpt/MobileViT_xs_DLV3/mIoU=0.653.ckpt",
    },
    {
        "name": "LRASPP-MobileNet V3 - xxs",
        "model": lt_lraspp_mv3,
        "ckpt": "ckpt/LRASPP_MV3/mIoU=0.588.ckpt",
    },
]


subsample_image_path = Path("../../data/CityScapes/subsamples/images")
subsample_mask_path = Path("../../data/CityScapes/subsamples/masks")

image_path_list = sorted(list(subsample_image_path.rglob("*.png")))
mask_path_list = sorted(list(subsample_mask_path.rglob("*.png")))

transforms = ValTransforms(target_size=(128, 256))

device = torch.device("cpu")


os.makedirs("to_onnx/onnx_models/fp32", exist_ok=True)
os.makedirs("to_onnx/onnx_models/fp16", exist_ok=True)
os.makedirs("to_onnx/onnx_models/int8", exist_ok=True)

dummy_input = torch.randn(1, 3, 128, 256)


class CityscapesCalibrationReader(CalibrationDataReader):

    def __init__(self, image_list, transform, max_samples=500):
        self.image_list = image_list[:max_samples]
        self.transform = transform
        self.enum_data = None

    def get_next(self):

        if self.enum_data is None:

            data = []

            for img_path in self.image_list:

                img = Image.open(img_path).convert("RGB")

                img_tensor, _ = self.transform(img, img)

                img_np = img_tensor.unsqueeze(0).numpy().astype(np.float32)

                data.append({"input": img_np})

            self.enum_data = iter(data)

        return next(self.enum_data, None)


def print_model_size(path):

    size = os.path.getsize(path) / (1024 * 1024)

    print(f"{Path(path).name} size : {size:.2f} MB")


for model_cfg in MODEL_LIST:

    name = model_cfg["name"]

    print("\n" + "=" * 60)
    print("Processing:", name)
    print("=" * 60)

    model = model_cfg["model"].load_from_checkpoint(model_cfg["ckpt"]).to(device)
    model.eval()

    inter_total = torch.zeros(19, dtype=torch.long)
    union_total = torch.zeros(19, dtype=torch.long)

    with torch.no_grad():

        for image_path, mask_path in zip(image_path_list, mask_path_list):

            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path)

            input_tensor, gt_mask = transforms(image, mask)
            input_tensor = input_tensor.unsqueeze(0)

            logits = model(input_tensor)

            if logits.shape[-2:] != gt_mask.shape[-2:]:

                logits = F.interpolate(
                    logits,
                    size=gt_mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            pred = logits.argmax(dim=1).squeeze().cpu()

            inter, union = iou_component(pred, gt_mask, 19, 255)

            inter_total += inter
            union_total += union

    mIoU, _ = iou_calculation(inter_total, union_total)

    print(f"{name} FP32 Subsample mIoU: {float(mIoU):.4f}")

    fp32_path = f"to_onnx/onnx_models/fp32/{name}_fp32.onnx"
    fp16_path = f"to_onnx/onnx_models/fp16/{name}_fp16.onnx"
    int8_path = f"to_onnx/onnx_models/int8/{name}_int8.onnx"

    print("Exporting FP32 ONNX...")

    torch.onnx.export(
        model,
        dummy_input,
        fp32_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    onnx_model = onnx.load(fp32_path)
    onnx.checker.check_model(onnx_model)

    print_model_size(fp32_path)

    print("Converting FP16...")

    fp16_model = float16.convert_float_to_float16(onnx_model)

    onnx.save(fp16_model, fp16_path)

    print_model_size(fp16_path)

    print("Running INT8 static quantization...")

    calibration_reader = CityscapesCalibrationReader(
        image_path_list,
        transforms,
        max_samples=500,
    )

    quantize_static(
        model_input=fp32_path,
        model_output=int8_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.Entropy,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
        },
    )

    print_model_size(int8_path)
