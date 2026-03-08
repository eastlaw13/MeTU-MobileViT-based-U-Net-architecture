
---

# MeTU: Lightweight Semantic Segmentation for Edge Devices

## 1. Project Overview

**MeTU** is a highly optimized, lightweight semantic segmentation model designed for resource-constrained edge devices. By integrating a **MobileViT** backbone with a **U-Net** based decoder, MeTU effectively captures both global context and high-resolution spatial details. This project comprehensively evaluates the architecture against SOTA lightweight models across urban driving scenes, general objects, adverse weather conditions, and actual edge hardware (Raspberry Pi 4B).

---

## 2. Quantitative Results & Generalization

We evaluated MeTU on both rigid urban scenes (Cityscapes) and non-rigid general objects (PASCAL VOC 2012) at a `512x1024` resolution.


### 2.1. Urban Scene Segmentation (Cityscapes)

We evaluated the models on the Cityscapes dataset to validate their performance in rigid, high-resolution urban environments.

| Model | Params (M) | FLOPs (G) | mIoU (%) |
| --- | --- | --- | --- |
| **MeTU-xxs (Ours)** | **1.02** | 7.14 | 68.49 |
| **MeTU-xs (Ours)** | 2.03 | 12.14 | **73.46** |
| Segformer-b0 | 3.72 | 18.01 | 70.46 |
| MobileViT-xxs + DeepLab V3| 1.86 | 3.21 | 61.50 |
| MobileViT-xs + DeepLab V3| 2.94 | 8.02 | 65.31 |
| LRASPP-MobileNet V3 (xxs) | 1.08 | **0.71** | 58.80 |

* **SOTA Efficiency:** **MeTU-xs** outperforms the transformer-based Segformer-b0 (73.46% vs 70.46% mIoU) while using **~45.5% fewer parameters** and **~32.6% fewer FLOPs**. Furthermore, **MeTU-xxs** achieves highly competitive performance (68.49%), requiring **~72.5% fewer parameters** than Segformer-b0 and consistently beating the equivalently sized `MobileViT-xxs + DeepLab V3` by a large margin (+6.99%p).
* **High-Resolution Detail Preservation:** The U-Net skip-connections explicitly preserve spatial details lost during aggressive downsampling. This allows **MeTU-xs** to significantly outperform Segformer-b0 on thin/small objects that require fine-grained localization, such as `Pole` (+5.0%p) and `Traffic Sign` (+3.3%p).

### 2.2. General Object Segmentation (PASCAL VOC 2012)

We evaluated the models on the PASCAL VOC 2012 dataset (excluding the background class) to rigorously validate their generalization capabilities and robustness on non-rigid, scale-varying general objects.

| Model | Params (M) | FLOPs (G) | mIoU (%) |
| --- | --- | --- | --- |
| **MeTU-xxs (Ours)** | **1.02** | 7.21 | 65.78 |
| **MeTU-xs (Ours)** | 2.03 | 12.20 | **73.46** |
| Segformer-b0 | 3.72 | 18.01 | 61.08 |
| MobileViT-xxs + DeepLab V3 | 1.86 | 3.21 | 63.50 |
| MobileViT-xs + DeepLab V3 | 2.94 | 8.03 | 68.23 |
| LRASPP-MobileNet V3 (xxs) | 1.08 | **0.71** | 57.56 |

* **Inductive Bias & Non-rigid Object Robustness:** While purely transformer-based architectures like Segformer-b0 struggle significantly with highly deformable objects due to a lack of inductive bias at low-parameter regimes (dropping to 61.08% on foreground objects), our hybrid **MeTU-xs** effectively leverages both local CNN features and global self-attention to outperform it by a massive margin (**+12.38%p mIoU**), despite using **~45% fewer parameters**.
* **Consistent Decoder Superiority across Scales:** When compared to the traditional `MobileViT + DeepLab V3` counterparts using the exact same backbones, MeTU architectures demonstrate absolute parameter efficiency. **MeTU-xs** achieves +5.23%p higher mIoU with ~31% fewer parameters, and our ultra-lightweight **MeTU-xxs** achieves +2.28%p higher mIoU while requiring **nearly half the parameters** (~45% reduction) of the `DeepLab V3 (xxs)` decoder. This decisively proves the efficiency of our U-Net-like skip connections over heavy ASPP blocks in strict edge constraints.
---

## 3. Robustness in Adverse Conditions (Cityscapes-C)
<div align="center">
    <a href="https://youtu.be/k2GShFCuxqQ">
        <img src="https://img.youtube.com/vi/k2GShFCuxqQ/maxresdefault.jpg" alt="MeTU Robustness Demo Video" width="80%">
    </a>
    <br>
    <em>Visualization of MeTU-xs robustness against varying out-of-distribution (OOD) corruptions. (Click to watch the full demo)</em>
</div>


To assess zero-shot robustness against unseen real-world corruptions, we benchmarked our models on the Cityscapes-C dataset across 5 severity levels under strictly out-of-distribution (OOD) conditions (i.e., no corruption augmentations were applied during training).


| Model | Params (M) | Clean mIoU (%) | mPC (%) |
| --- | --- | --- | --- |
| **MeTU-xxs (Ours)** | **1.02** | 68.49 | 40.58 |
| **MeTU-xs (Ours)** | 2.03 | **73.46** | 43.87 |
| Segformer-b0 | 3.72 | 70.46 | **47.38** |
| MobileViT-xxs + DeepLab V3 | 1.85 | 61.50 | 36.85 |
| MobileViT-xs + DeepLab V3 | 2.94 | 65.31 | 41.12 |
| LRASPP-MobileNet V3 (xxs) | 1.08 | 58.80 | 36.86 |

* **Architecture Trade-offs:** While our hybrid CNN-Transformer architecture (MeTU-xs) dominates on clean data (+3.0%p mIoU over Segformer-b0), the pure-transformer Segformer-b0 shows stronger resistance to extreme corruptions. This highlights the known phenomenon that pure transformers rely more on global shape bias rather than local textures, granting them inherent robustness at the cost of heavier parameters (~1.8x larger than ours).
* **Superior Decoder Design:** Compared to the standard `MobileViT-xs + DeepLab V3` counterpart, **MeTU-xs** demonstrates significantly higher robustness (**+2.75%p mPC**) while remaining much more parameter-efficient. Furthermore, even our ultra-lightweight **MeTU-xxs (1.02M)** outperforms the heavier `MobileViT-xxs + DeepLab V3 (1.85M)` by **+3.73%p mPC**. This consistently proves that our explicit skip-connection mechanism is functionally more reliable under unseen conditions than traditional ASPP blocks in lightweight regimes.
---

##  4. Edge Device Deployment: Raspberry Pi 4B

To validate the practical applicability and inference stability of MeTU in real-world edge scenarios, we benchmarked the models on a **Raspberry Pi 4B (4GB RAM, ARM Cortex-A72 CPU-only)** using **ONNX Runtime**.

* **Test Configurations:** 200 random images from the **Cityscapes dataset (128×256 resolution)** were used for inference benchmarking. Each experiment was preceded by a **20-iteration warm-up phase using tensors with identical input resolution** to stabilize runtime caching and memory allocation.
* **Metrics:** Beyond average latency, we report **P50, P90, and P99 tail latencies** to strictly evaluate the execution stability (runtime jitter) required for **real-time edge and robotic systems**.

---

### 4.1. FP32 Inference Performance (Accuracy vs. Speed Trade-off)

| Model | Params (M) | Average (ms) | P50 (ms) | P90 (ms) | P99 (ms) | Throughput (FPS) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **MeTU-xxs (Ours)** | **1.0** | **83.02** | **82.99** | **83.42** | **84.09** | **12.17** |
| **MeTU-xs (Ours)** | **2.0** | **134.30** | **134.21** | **134.88** | **136.27** | **7.44** |
| Segformer-B0 | 3.7 | 124.75 | 124.69 | 125.19 | 126.61 | 8.02 |
| MobileViT-xxs + DeepLab V3 | 1.9 | 44.36 | 44.41 | 44.56 | 44.67 | 22.54 |
| MobileViT-xs + DeepLab V3 | 2.9 | 98.96 | 98.92 | 99.36 | 100.30 | 10.11 |
| LRASPP-MobileNet V3 (xxs) | 1.1 | 13.06 | 13.03 | 13.25 | 13.46 | 76.57 |

---

### 🛠 Edge Hardware & Bottleneck Analysis

* **High Execution Stability (Low Jitter):**  
  For **MeTU-xxs**, the difference between the median (**P50: 82.99 ms**) and worst-case (**P99: 84.09 ms**) latency is only **~1.10 ms**.  
  Similarly, **MeTU-xs** shows a small variance between **P50 (134.21 ms)** and **P99 (136.27 ms)** of roughly **~2.06 ms**.  
  This low latency spread indicates **highly stable inference behavior**, which is crucial for deterministic execution in real-time robotic or embedded perception pipelines.

* **Accuracy–Speed Trade-off on Edge CPUs:**  
  `LRASPP-MobileNetV3 (xxs)` achieves extremely high throughput (**76.57 FPS**) due to its lightweight CNN-only design, but at the cost of significantly reduced segmentation accuracy.  
  In contrast, **MeTU-xxs** provides a more balanced operating point, achieving competitive segmentation performance while maintaining **12.17 FPS** directly on a CPU-only Raspberry Pi.

* **Architecture Hardware Constraints (Memory vs. Compute):**  
  Although **MeTU-xs (2.0M parameters)** is smaller than **MobileViT-xs + DeepLabV3 (2.9M parameters)**, it runs slower (**7.44 FPS vs. 10.11 FPS**).  
  This observation suggests that **U-Net-style skip connections and multi-scale feature fusion introduce additional memory traffic**, creating a **memory-bandwidth bottleneck** on the Raspberry Pi’s limited memory subsystem.  
  By contrast, architectures dominated by convolutional or attention blocks with fewer high-resolution feature merges tend to be more **compute-bound and cache-friendly** on ARM CPUs.

* **Implications for Edge Deployment:**  
  These results highlight that **model size alone does not determine real-world performance on edge hardware**.  
  Instead, architectural factors such as **memory access patterns, feature map resolution, and operator scheduling** play a critical role in determining inference throughput on resource-constrained CPUs.

---
