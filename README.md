
---

# MeTU: Lightweight Semantic Segmentation for Edge Devices

## 1. Project Overview

**MeTU** is a highly optimized, lightweight semantic segmentation model designed for resource-constrained edge devices. By integrating a **MobileViT** backbone with a **U-Net** based decoder, MeTU effectively captures both global context and high-resolution spatial details. This project comprehensively evaluates the architecture against SOTA lightweight models across urban driving scenes, general objects, adverse weather conditions, and actual edge hardware (Raspberry Pi 4B).

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
<div align="center">
    <a href="https://youtu.be/k2GShFCuxqQ">
        <img src="https://img.youtube.com/vi/k2GShFCuxqQ/maxresdefault.jpg" alt="MeTU Robustness Demo Video" width="80%">
    </a>
    <br>
    <em>Visualization of MeTU-xs robustness against varying out-of-distribution (OOD) corruptions. (Click to watch the full demo)</em>
</div>


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

## 🚀 4. Edge Device Deployment: Raspberry Pi 4B

To validate the practical applicability and inference stability of MeTU in real-world edge scenarios, we benchmarked the models on a **Raspberry Pi 4B (4GB RAM, ARM Cortex-A72 CPU-only)** using ONNX Runtime. 

* **Test Configurations:** 200 random images from Cityscapes (128x256 resolution), preceded by a 20-image warm-up phase to ensure cache stability.
* **Metrics:** Beyond average latency, we report **P50, P90, and P99 tail latencies** to strictly evaluate the execution stability (jitter) required for real-time robotic systems.

### 4.1. FP32 Inference Performance (Accuracy vs. Speed Trade-off)
| Model | Params (M) | Average (ms) | P50 (ms) | P90 (ms) | P99 (ms) | Throughput (FPS) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| LRASPP-MobileNet V3 (xxs) | 1.1 | 13.06 | 13.03 | 13.25 | 13.46 | 76.55 |
| MobileViT-xxs + DeepLab V3 | 1.9 | 44.36 | 44.41 | 44.56 | 44.67 | 22.54 |
| **MeTU-xxs (Ours)** | **1.0** | **83.02** | **82.99** | **83.42** | **84.09** | **12.04** |
| MobileViT-xs + DeepLab V3 | 2.9 | 98.96 | 98.92 | 99.36 | 100.31 | 10.11 |
| Segformer-B0 | 3.7 | 124.75 | 124.69 | 125.19 | 126.61 | 8.02 |
| **MeTU-xs (Ours)** | **2.0** | **134.30** | **134.18** | **135.93** | **136.87** | **7.51** |

### 🛠 Edge Hardware & Bottleneck Analysis

* **High Execution Stability (Low Jitter):** For **MeTU-xxs**, the difference between the median (P50: 82.99ms) and the worst-case (P99: 84.09ms) is merely **~1.1ms**. This extremely low variance proves that our architecture guarantees highly predictable latency on edge CPUs, making it exceptionally reliable for continuous real-time execution.
* **The Ultimate Balance (Accuracy vs. Speed):** While `LRASPP` operates at an extreme 76 FPS, it suffers from a significant mIoU drop (~58% in our Cityscapes evaluation). Conversely, **MeTU-xxs** strikes the optimal Pareto frontier, achieving near SOTA-level accuracy (~67% mIoU) while maintaining a highly practical **12.04 FPS** directly on a standard ARM CPU.
* **Architecture Hardware Constraints (Memory vs. Compute):** Despite having fewer parameters (2.0M), MeTU-xs runs slower (7.51 FPS) than MobileViT-xs+DeepLab V3 (2.9M, 10.11 FPS). This empirically proves that U-Net's high-resolution skip-connections create a **Memory-Bound** bottleneck on the Raspberry Pi's limited memory bandwidth, whereas ASPP modules are more **Compute-Bound** and cache-friendly.
* **The ONNX INT8 Overhead Anomaly:** Interestingly, deploying INT8 models actually *decreased* the throughput for all MobileViT-based architectures (e.g., MeTU-xxs dropped from 12.04 to 11.87 FPS). Since the Cortex-A72 CPU lacks dedicated INT8 tensor cores, the quantization/dequantization operations in ONNX Runtime introduce overhead that outweighs the computational savings for hybrid (CNN+Transformer) structures.

---
