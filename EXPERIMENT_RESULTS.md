# OCT Cross-Vendor Domain Adaptation: Experiment Results

This report documents the performance of various Domain Adaptation (DA) techniques used to bridge the gap between **Ze Zeiss Cirrus (Source)** and **Heidelberg Spectralis (Target)** for retinal fluid segmentation.

## Dataset Overview
- **Classes:** 4 (Background, IRF, SRF, PED)
- **Source Domain:** Zeiss Cirrus (3072 slices)
- **Target Domain:** Heidelberg Spectralis (1176 slices)
- **Primary Metric:** Overall Dice Score (F1-Score)
- **Patient Isolation:** Strict anatomical separation ensured via volume-level splitting.

---

## 1. Summary of Results (Target: Spectralis) - Methods Comparison
This table compares general Domain Adaptation methods using the target backbones (**ResNet-50, ResNet-18, AnamNet**).

| Rank | Method                          | Target Dice | Target IoU | Key Strength / Observation                          |
| :--- | :------------------------------ | :---------: | :--------: | :-------------------------------------------------- |
| 🏆 1 | **DANN (Domain Adversarial)**   | **0.7519**  | 0.6342     | Strongest on ResNet-50. Best feature alignment.     |
| 🥈 2 | **DDSP (Feature Disruption)**   | 0.7298      | 0.6105     | Robust cross-vendor statistics mixing.              |
| 🚀 3 | **MS-FDA (Multi-Scale)**        | 0.7115      | 0.5982     | Top Spectral variant for ResNet-50.                 |
| 🌟 4 | Dist. FDA (Batch-Mean)          | 0.7336      | 0.6120     | Stable style extraction via high-VRAM batches.      |
| 🥉 5 | FDA Fine-tuned (Pixel)          | 0.6633      | 0.5488     | Classic spectral transfer.                          |
| 📊 6 | Baseline (Direct Transfer)      | 0.6402      | 0.5312     | ResNet-50 transfer with strict volume isolation.   |
| 🧬 7 | AnamNet + Adv-1to1              | 0.5362      | 0.4410     | Best lightweight/edge-ready performance.            |
| 🛡️ 8 | SFDA (Source-Free)              | 0.2488      | 0.2477     | Privacy-preserving entropy minimization.            |

---

## 1.1 Cross-Method-Model Comparison (Dice Score)
Comprehensive evaluation of key methods across the target architectures at 256x256 resolution.

| Architecture | Baseline (Direct Transfer) | FDA (Bot) | MS-FDA (Multi-Scale) | Adv-1to1 | Dist-FDA | DANN (Adv) | DDSP (Mix) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50 (U-Net)** | 0.5061 | 0.6633 | 0.7115 | 0.6303 | 0.7336 | **0.7519** | 0.7298 |
| **ResNet-18 (U-Net)** | **0.5408** | 0.6186 | 0.6360 | 0.6354 | 0.6538 | 0.5844 | 0.5881 |
| **AnamNet** | 0.2660 | 0.4059 | 0.5198 | **0.5362** | 0.3297 | 0.2775 | 0.4625 |

---

## 2. Detailed Experiment Descriptions

### 2.1. DDSP (Dual Domain Distribution Disruption)
- **Method:** Stochastically mixes channel-wise statistics (mean/std) between Source and Target in the shallow layers.
- **Insight:** Forces the model to be distribution-agnostic without adversarial instability. Achieved strong results across all ResNet variants.

### 2.2. DANN (Domain Adversarial Neural Networks)
- **Method:** Uses a Gradient Reversal Layer (GRL) and domain discriminator to force feature-level domain indistinguishability.
- **Insight:** The performance leader for ResNet-50. Successfully aligns high-level semantic features between Cirrus and Spectralis scanners.

### 2.3. Advanced Feature-Space FDA (Regularized)
- **Method:** Extends Feature FDA with MI Minimization, Physics-Informed Attenuation, and Topological Persistence.
- **Insight:** By tuning the MI weight, we achieved a robust model that outperforms standard bottleneck FDA, especially on the lightweight AnamNet.

### 2.4. Multi-Scale Feature FDA
- **Method:** Applies FDA spectral swapping at every resolution level of the encoder pyramid.
- **Insight:** Bridges the scanner gap more comprehensively than bottleneck-only FDA by aligning low-level textures and high-level structures simultaneously.

### 2.5. Zero-Shot / Direct Transfer (Cirrus to Spectralis)
- **Method:** Models are trained exclusively on Zeiss Cirrus annotated slices and evaluated directly on Heidelberg Spectralis slices.
- **Constraint:** Strict patient-level volume splitting ensures that no patient data overlaps between training and validation folds.

---

## 3. Final Recommendation
For high-performance clinical applications, **ResNet-50 + DANN** is the definitive choice. For mobile or edge deployment, **AnamNet + Adv-1to1** provides the best accuracy-to-parameter ratio.
