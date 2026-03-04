
---

# MeTU: Lightweight Semantic Segmentation for Edge Devices

## 1. Project Overview

**MeTU-v3** is a highly optimized, lightweight semantic segmentation model designed for resource-constrained edge devices. By integrating a **MobileViT** backbone with a **U-Net** based decoder, MeTU effectively captures both global context and high-resolution spatial details. This project comprehensively evaluates the architecture against SOTA lightweight models across urban driving scenes, general objects, adverse weather conditions, and actual edge hardware (Raspberry Pi 4B).

---

## 2. Quantitative Results & Generalization

We evaluated MeTU on both rigid urban scenes (Cityscapes) and non-rigid general objects (PASCAL VOC 2012) at a `512x1024` resolution.

### 2.1. Urban Scene Segmentation (Cityscapes)

| Model | Params (M) | FLOPs (G) | mIoU (%) |
| --- | --- | --- | --- |
| **MeTU-xs (Ours)** | 2.03 | 12.14 | **70.69** |
| **MeTU-xxs (Ours)** | **1.02** | 7.14 | 67.07 |
| Segformer-b0 | 3.72 | 18.01 | 67.57 |
| MobileViT + DeepLab V3 (xs) | 2.94 | 8.02 | 61.97 |
| LRASPP-MobileNet V3 | 1.08 | **0.71** | 58.73 |

* **SOTA Efficiency:** **MeTU-xxs** achieves comparable performance to Segformer-b0 (67.1% vs 67.6% mIoU) while requiring **~72.5% fewer parameters** and **~60% fewer FLOPs**.
* **High-Resolution Detail Preservation:** The U-Net skip-connections explicitly preserve spatial details, allowing MeTU-xs to significantly outperform Segformer-b0 on thin/small objects like `Bicycle` (+8.3%p) and `Pole` (+4.5%p).

### 2.2. General Object Segmentation (PASCAL VOC 2012)

| Model | Params (M) | mIoU (%) |
| --- | --- | --- |
| **MeTU-xs (Ours)** | 2.03 | **71.30** |
| **MeTU-xxs (Ours)** | **1.02** | 66.95 |
| MobileViT + DeepLab V3 (xs) | 2.94 | 69.24 |
| Segformer-b0 | 3.72 | 62.42 |

* **Robust Generalization:** Beyond fixed-perspective driving scenes, MeTU-xs excels on non-rigid objects, outperforming purely hierarchical transformers (Segformer-b0) on complex classes like `cow` (82.3% vs 59.1%) and `horse` (83.9% vs 61.6%).
* **Architectural Superiority:** Compared to the ASPP module in DeepLab V3, MeTU's spatial connectivity prevents gridding artifacts, leading to massive improvements in web-like structures (e.g., `bicycle`: 53.1% vs 29.6%).

---

## 3. Robustness in Adverse Conditions (Cityscapes-C)

To validate reliability in real-world, out-of-distribution (OOD) scenarios, we benchmarked the models against 13 different corruption types across 5 severity levels.

| Model | Clean mIoU (%) | mPC (Mean Performance under Corruption, %) |
| --- | --- | --- |
| **MeTU-xs (Ours)** | **70.69** | **50.90** |
| Segformer-b0 | 67.57 | 49.46 |
| **MeTU-xxs (Ours)** | 67.07 | 48.16 |
| MobileViT + DeepLab V3 (xs) | 61.97 | 45.41 |

* **Overall Resilience:** MeTU-xs secures the highest overall robustness score (**50.90% mPC**). Even the ultra-lightweight MeTU-xxs (1.02M params) closely follows Segformer-b0 (3.72M params).
* **CNN vs. Transformer Trade-off:** While Segformer's global attention acts as a low-pass filter to slightly better resist severe high-frequency noise (like heavy `frost` or `zoom_blur`), MeTU's skip-connections occasionally propagate this noise into the decoder. Recognizing this trade-off is crucial for future architectural enhancements.

---

## 4. Edge Hardware Deployment (Raspberry Pi 4B)

The ultimate test of a lightweight model is its on-device performance. We deployed the models on a **Raspberry Pi 4B (4GB RAM, CPU-only)** using ONNX Runtime.

*(Note: Inference was performed at `128x256` resolution to test zero-shot scale degradation due to resource constraints.)*

| Model | Precision | Params (M) | mIoU (%) | Latency (ms) | Throughput (FPS) |
| --- | --- | --- | --- | --- | --- |
| **MeTU-xxs (Ours)** | FP32 | 1.02 | 33.46 | **82.14** | **12.17** |
| **MeTU-xs (Ours)** | FP32 | 2.03 | **37.23** | 138.16 | 7.24 |
| Segformer-B0 | FP32 | 3.72 | 36.48 | 125.74 | 7.95 |
| MobileViT + DeepLab V3 | FP32 | 1.90 | 22.25 | 45.86 | 21.81 |

### 🛠 Hardware & Quantization Bottleneck Analysis

* **Real-Time Feasibility:** **MeTU-xxs** achieves an impressive **12.17 FPS** directly on an ARM CPU without any specialized NPU/GPU acceleration, proving its immediate real-world utility.
* **Memory-Bound vs. Compute-Bound:** Despite having fewer parameters (1.0M vs 1.9M), MeTU-xxs has lower FPS than MobileViT+DeepLab V3. This empirically proves that U-Net architectures are heavily **Memory-Bound** on edge devices due to the memory bandwidth required for high-resolution feature concatenation, whereas ASPP is more **Compute-Bound** and cache-friendly.
* **The INT8 Quantization Collapse:** Standard Post-Training Quantization (PTQ) caused a complete mIoU collapse (~3%) for all MobileViT-based models. This is a known limitation where depthwise convolutions and non-linear activations create severe activation outliers. Segformer-B0, being a pure transformer, retained ~35.9% mIoU. Future work will require Quantization-Aware Training (QAT) to fully unlock INT8 edge performance for MeTU.

---