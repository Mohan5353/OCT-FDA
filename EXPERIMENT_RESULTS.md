# OCT Cross-Vendor Domain Adaptation: Experiment Results

This report documents the performance of various Domain Adaptation (DA) techniques used to bridge the gap between **Ze Zeiss Cirrus (Source)** and **Heidelberg Spectralis (Target)** for retinal fluid segmentation.

## Dataset Overview
- **Classes:** 4 (Background, IRF, SRF, PED)
- **Source Domain:** Zeiss Cirrus (3072 slices)
- **Target Domain:** Heidelberg Spectralis (1176 slices)
- **Primary Metric:** Overall Dice Score (F1-Score)

---

## 1. Summary of Results (Target: Spectralis) - Methods Comparison
This table compares general Domain Adaptation methods using the default backbone (ResNet-101 / U-Net) unless otherwise specified.

| Rank | Method                          | Target Dice | Target IoU | Key Strength / Observation                          |
| :--- | :------------------------------ | :---------: | :--------: | :-------------------------------------------------- |
| 🏆 1 | **DDSP (Feature Disruption)** | **0.7569**  | 0.6435     | Current Champion. Best SRF (0.84) & PED (0.66).     |
| 🦖 2 | **Baseline (ConvNeXt-L)**       | **0.7567**  | 0.6388     | Strongest Baseline. Zero-Shot Modern Backbone.      |
| 🥈 3 | DANN (Domain Adversarial)       | 0.7527      | 0.6342     | Very strong feature-level alignment.                |
| 🌟 4 | Dist. FDA (ResNet-101)          | 0.7458      | 0.6288     | New Spectral SOTA. Batch-Mean Stability.            |
| 🚀 5 | Multi-Scale Feature FDA         | 0.7383      | 0.6212     | Top MS. Aligns multi-level textures.                |
| 🚀 6 | Adv. Feature-Space FDA          | 0.7301      | 0.6128     | Physics-Informed. Robust & Disentangled.            |
| 🚀 7 | Feature-Space FDA               | 0.7272      | 0.6092     | Standard bottleneck spectral swapping.              |
| 🦖 8 | Dist. FDA (ConvNeXt-L)          | 0.7057      | 0.5891     | Modern Backbone. Stalled at 320x320.                |
| 🥉 9 | FDA Fine-tuned                  | 0.6970      | 0.5695     | Classic style transfer on raw images.               |
| 📊 10| Baseline (Zero-Shot)            | 0.6685      | 0.5387     | Standard transfer without adaptation.               |
| 🔗 11| CLUDA (Contrastive Alignment)   | 0.5306      | 0.4249     | Class-wise feature clustering.                      |
| ⚡ 12| Energy-Regularized UDA          | 0.5276      | 0.4190     | OOD scoring for pseudo-labeling.                    |
| 🌟 13| Dist. FDA (SegResNet)           | 0.4117      | 0.3210     | Batch-Averaged Style. High-VRAM. Stalled by sink.  |
| 📉 14| FMC (Fourier Mixup Consistency) | 0.2871      | 0.2672     | Unstable consistency regularization.                |
| 🛡️ 15| SFDA (Source-Free Adaptation)   | 0.2488      | 0.2477     | Privacy-preserving entropy minimization.            |
| ⏱️ 16| TENT (Test-Time Adaptation)     | 0.2495      | 0.2480     | Inference-time BN optimization.                     |

---

## 1.1 Cross-Method-Model Comparison (Dice Score)
Comprehensive evaluation of all key methods across all implemented architectures.

| Architecture | Baseline | FDA (Bot) | MS-FDA | Adv-1to1 | Dist-FDA | DANN (Adv) | DDSP (Mix) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-101 (U-Net)** | 0.6685 | 0.7272 | 0.7383 | 0.7301 | **0.7458** | 0.7527 | **0.7569** |
| **ResNet-50 (U-Net)** | 0.6402 | 0.6633 | 0.7115 | 0.6303 | 0.7336 | **0.7519** | 0.7298 |
| **ResNet-18 (U-Net)** | **0.6827** | 0.6186 | 0.6360 | 0.6354 | 0.6538 | 0.5844 | 0.5881 |
| **ResNet-10 (U-Net)** | 0.5146 | *TBD* | **0.6398** | *TBD* | *TBD* | *TBD* | *TBD* |
| **MobileNetV2 (U-Net)** | **0.6635** | *TBD* | 0.6170 | *TBD* | *TBD* | *TBD* | *TBD* |
| **ConvNeXt-L (U-Net)** | **0.7567** | 0.6594 | 0.6764 | 0.7194 | 0.7057 | 0.6879 | 0.6794 |
| **ConvNeXt-T (U-Net)** | 0.5441 | 0.3583 | 0.4139 | **0.5449** | 0.4152 | 0.3708 | 0.2846 |
| **AnamNet** | 0.3025 | 0.4059 | 0.5198 | **0.5362** | 0.3297 | 0.2775 | 0.4625 |
| **SegResNet** | **0.6557** | 0.5707 | 0.3834 | 0.4798 | 0.4117 | 0.5597 | 0.6043 |
| **MISSFormer** | 0.2350 | 0.2350 | **0.2560** | 0.2350 | 0.0620 | 0.0645 | 0.0959 |
| **TinyUnet** | 0.3697 | *TBD* | **0.5619** | *TBD* | *TBD* | *TBD* | *TBD* |



---

## 1.2 Cross-Architecture Comparison (Fixed Strategy: Multi-Scale FDA)
This table compares different model backbones while keeping the Domain Adaptation strategy constant (MS-FDA, L=0.01).

| Model Backbone | Complexity | Target Dice | Inference Speed | Verdict |
| :--- | :--- | :---: | :---: | :--- |
| ResNet-101 (U-Net) | High (Pretrained) | **0.7383** | Moderate | Best Overall. ImageNet weights are critical. |
| AnamNet | Ultra-Lightweight | 0.5198 | Very Fast | Best for Edge. High efficiency/accuracy ratio. |
| SegResNet | Moderate | 0.3834 | Fast | Needs medical-specific pretraining to shine. |
| MISSFormer | High (Transformer) | 0.2560 | Slow | Limited by resolution (128x128) and data volume. |

---

## 1.5 Advanced Clinical Evaluation Metrics
To prove clinical utility and deployment safety, models are evaluated beyond Dice/IoU.

### 1.5.1 Expected Calibration Error (ECE)
Lower is better (safer).
*   DDSP: IRF (0.0028), SRF (0.0009), PED (0.0011)
*   Baseline: IRF (0.0026), SRF (0.0011), PED (0.0013)

### 1.5.2 Boundary/Surface Distances (HD95 & ASSD)
Lower is better.
*   DDSP: SRF (HD95 = 34.62 px, ASSD = 13.64 px), PED (HD95 = 51.77 px, ASSD = 16.26 px)
*   Baseline: SRF (HD95 = 52.49 px, ASSD = 19.58 px), PED (HD95 = 66.74 px, ASSD = 22.08 px)

---

## 2. Detailed Experiment Descriptions

### 2.9. DDSP (Dual Domain Distribution Disruption)
- **Method:** Published in 2024. Stochastically mixes channel-wise statistics (mean/std) between Source and Target in the shallow layers.
- **Insight:** The Winner. It forces the model to be distribution-agnostic without adversarial instability. Achieved exceptional results on SRF (0.84) and PED (0.66).

### 2.12. Advanced Feature-Space FDA (Regularized)
- **Method:** Extends Feature FDA with MI Minimization, Physics-Informed Attenuation, and Topological Persistence.
- **Result:** 0.7301 Dice.
- **Insight:** By tuning the MI weight ($\lambda_{mi}=1e-6$), we achieved a robust model that outperforms standard FDA.

### 2.13. Multi-Scale Feature FDA
- **Method:** Applies FDA spectral swapping at every resolution level of the ResNet-101 encoder.
- **Result:** 0.7383 Dice.
- **Insight:** Most effective FDA variant. Bridges the scanner gap more comprehensively than bottleneck-only FDA.

### 2.17. Batch-Averaged Distribution FDA (ResNet-101 & SegResNet)
- **Method:** Calculates the Batch Mean of target amplitudes in feature space.
- **Result:** 0.7458 Dice (ResNet-101).
- **Insight:** Extracting a stable "style" from the target domain using large batches (64) provides the highest spectral adaptation performance to date.

### 2.19. ConvNeXt-L Robustness Analysis
- **Result:** Baseline 0.7567 Dice (Rank 2).
- **Insight:** ConvNeXt-L displayed remarkable zero-shot robustness, outperforming all adapted ResNet-101 models by default. Interestingly, applying spectral adaptation (FDA/Dist-FDA) slightly decreased its performance. This suggests that modern, high-capacity backbones like ConvNeXt are inherently more invariant to scanner styles, and spectral swapping may perturb their highly optimized feature representations.

---

## 3. Final Recommendation
For production or clinical research, the **DDSP** model remains the performance leader, but **ResNet-101 + Dist-FDA** is the strongest alternative for interpretable spectral alignment.
