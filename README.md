
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
| **MeTU-xs (Ours)** | 2.03 | 12.14 | **73.46** |
| **MeTU-xxs (Ours)** | **1.02** | 7.14 | 68.49 |
| Segformer-b0 | 3.72 | 18.01 | 70.46 |
| MobileViT + DeepLab V3 (xs) | 2.94 | 8.02 | 65.31 |
| LRASPP-MobileNet V3 | 1.08 | **0.71** | 58.80 |

* **SOTA Efficiency:** **MeTU-xs** outperforms the transformer-based Segformer-b0 (73.46% vs 70.46% mIoU) while using **~45.5% fewer parameters** and **~32.6% fewer FLOPs**. Furthermore, **MeTU-xxs** achieves highly competitive performance (68.49%) requiring **~72.5% fewer parameters** than Segformer-b0.
* **High-Resolution Detail Preservation:** The U-Net skip-connections explicitly preserve spatial details, allowing MeTU-xs to significantly outperform Segformer-b0 on thin/small objects like `Pole` (+5.0%p) and `Traffic Sign` (+3.3%p).

### 2.2. General Object Segmentation (PASCAL VOC 2012)

We also evaluated the models on the PASCAL VOC 2012 dataset to validate their robustness on non-rigid and scale-varying general objects.

| Model | Params (M) | FLOPs (G) | mIoU (%) |
| --- | --- | --- | --- |
| **MeTU-xs (Ours)** | 2.03 | 12.20 | **71.30** |
| MobileViT + DeepLab V3 (xs) | 2.94 | 8.03 | 69.24 |
| **MeTU-xxs (Ours)** | **1.02** | 7.21 | 66.95 |
| MobileViT + DeepLab V3 (xxs) | 1.85 | 3.21 | 64.72 |
| Segformer-b0 | 3.72 | 18.03 | 62.42 |
| LRASPP-MobileNet V3 | 1.08 | **0.71** | 58.99 |

* **Exceptional Robustness to Non-rigid Objects:** While purely transformer-based architectures often struggle with highly deformable objects at very low parameter regimes, **MeTU-xs** outperforms Segformer-b0 by a massive margin (**+8.88%p mIoU**) while utilizing **~45.5% fewer parameters**.
* **Beating Larger Transformers with 1M Params:** Remarkably, even the ultra-lightweight **MeTU-xxs** (1.02M params) easily surpasses the much heavier Segformer-b0 (66.95% vs 62.42% mIoU).
* **Decoder Efficiency:** Compared to the standard `MobileViT + DeepLab V3` counterpart, MeTU architectures consistently achieve higher mIoU using fewer parameters, demonstrating that our proposed U-Net-like skip connection design is highly superior at capturing complex shapes like `bicycle` (+23.5%p) and `bottle` (+4.6%p) compared to ASPP-based decoders.

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
* **CNN vs. Transformer Trade-off:** While Segformer's global attention acts as a low-pass filter to slightly better resist severe high-frequency noise (like heavy `frost` or `snow`), MeTU's skip-connections occasionally propagate this noise into the decoder. Recognizing this trade-off is crucial for future architectural enhancements.

---

##  4. Edge Device Deployment: Raspberry Pi 4B

To validate the practical applicability and inference stability of MeTU in real-world edge scenarios, we benchmarked the models on a **Raspberry Pi 4B (4GB RAM, ARM Cortex-A72 CPU-only)** using **ONNX Runtime**.

* **Test Configurations:** 200 random images from the **Cityscapes dataset (128×256 resolution)** were used for inference benchmarking. Each experiment was preceded by a **20-iteration warm-up phase using tensors with identical input resolution** to stabilize runtime caching and memory allocation.
* **Metrics:** Beyond average latency, we report **P50, P90, and P99 tail latencies** to strictly evaluate the execution stability (runtime jitter) required for **real-time edge and robotic systems**.

---

### 4.1. FP32 Inference Performance (Accuracy vs. Speed Trade-off)

| Model | Params (M) | Average (ms) | P50 (ms) | P90 (ms) | P99 (ms) | Throughput (FPS) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| LRASPP - MobileNet v3 - xxs | 1.1 | 13.06 | 13.03 | 13.25 | 13.46 | 82.64 |
| MobileViT-xxs + DeepLab v3 | 1.9 | 44.36 | 44.41 | 44.56 | 44.67 | 22.54 |
| **MeTU-xxs (Ours)** | **1.0** | **83.02** | **82.99** | **83.42** | **84.09** | **12.17** |
| MobileViT-xs + DeepLab v3 | 2.9 | 98.96 | 98.92 | 99.36 | 100.30 | 10.11 |
| Segformer-B0 | 3.7 | 124.75 | 124.69 | 125.19 | 126.61 | 8.02 |
| **MeTU-xs (Ours)** | **2.0** | **134.30** | **134.21** | **134.88** | **136.27** | **7.44** |

---

### 🛠 Edge Hardware & Bottleneck Analysis

* **High Execution Stability (Low Jitter):**  
  For **MeTU-xxs**, the difference between the median (**P50: 82.99 ms**) and worst-case (**P99: 84.09 ms**) latency is only **~1.10 ms**.  
  Similarly, **MeTU-xs** shows a small variance between **P50 (134.21 ms)** and **P99 (136.27 ms)** of roughly **~2.06 ms**.  
  This low latency spread indicates **highly stable inference behavior**, which is crucial for deterministic execution in real-time robotic or embedded perception pipelines.

* **Accuracy–Speed Trade-off on Edge CPUs:**  
  `LRASPP-MobileNetV3` achieves extremely high throughput (**82.64 FPS**) due to its lightweight CNN-only design, but at the cost of significantly reduced segmentation accuracy.  
  In contrast, **MeTU-xxs** provides a more balanced operating point, achieving competitive segmentation performance while maintaining **12.17 FPS** directly on a CPU-only Raspberry Pi.

* **Architecture Hardware Constraints (Memory vs. Compute):**  
  Although **MeTU-xs (2.0M parameters)** is smaller than **MobileViT-xs + DeepLabV3 (2.9M parameters)**, it runs slower (**7.44 FPS vs. 10.11 FPS**).  
  This observation suggests that **U-Net-style skip connections and multi-scale feature fusion introduce additional memory traffic**, creating a **memory-bandwidth bottleneck** on the Raspberry Pi’s limited memory subsystem.  
  By contrast, architectures dominated by convolutional or attention blocks with fewer high-resolution feature merges tend to be more **compute-bound and cache-friendly** on ARM CPUs.

* **Implications for Edge Deployment:**  
  These results highlight that **model size alone does not determine real-world performance on edge hardware**.  
  Instead, architectural factors such as **memory access patterns, feature map resolution, and operator scheduling** play a critical role in determining inference throughput on resource-constrained CPUs.

---
