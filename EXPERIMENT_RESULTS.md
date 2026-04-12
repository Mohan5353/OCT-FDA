# OCT Cross-Vendor Domain Adaptation: Experiment Results

This report documents the performance of various Domain Adaptation (DA) techniques used to bridge the gap between **Ze Zeiss Cirrus (Source)** and **Heidelberg Spectralis (Target)** for retinal fluid segmentation.

## Dataset Overview
- **Classes:** 4 (Background, IRF, SRF, PED)
- **Source Domain:** Zeiss Cirrus (3072 slices)
- **Target Domain:** Heidelberg Spectralis (1176 slices)
- **Primary Metric:** Overall Dice Score (F1-Score)

---

## 1. Summary of Results (Target: Spectralis)

| Rank | Method                          | Target Dice | Target IoU | Key Strength / Observation                          |
| :--- | :------------------------------ | :---------: | :--------: | :-------------------------------------------------- |
| 🏆 **1** | **DDSP (Feature Disruption)** | **0.7569**  | **0.6435** | **Current Champion.** Best SRF (0.84) & PED (0.66). |
| 🥈 2 | DANN (Domain Adversarial)       | 0.7527      | 0.6342     | Very strong feature-level alignment.                |
| 🥉 3 | **FDA Fine-tuned**              | 0.6970      | 0.5695     | Strongest pixel-level style adaptation.             |
| 📊 4 | Baseline (Zero-Shot)            | 0.6685      | 0.5387     | High anatomical accuracy, but scanner biased.       |
| 🔗 5 | CLUDA (Contrastive Alignment)   | 0.5306      | 0.4249     | Class-wise feature clustering; noisy on target.     |
| 📉 6 | FMC (Fourier Mixup Consistency) | 0.2871      | 0.2672     | High variance; unstable consistency regularization. |
| 🧠 5 | Hyperbolic DA                   | 0.5702      | 0.4533     | **Best PED Class Performance (0.5061).**           |
| 🔄 6 | UDA (Pseudo-labeling)           | 0.5631      | 0.4484     | Limited by label noise in small structures.         |
| 📉 7 | ADVENT (Entropy Min)            | 0.4965      | 0.3998     | Unstable training on sparse fluid data.             |
| ⚖️ 8 | DSBN (Domain BN)                | 0.3907      | 0.3366     | Reliable for inference, weak for adaptation.        |
| 🤖 9 | SegFormer (Transformer)         | 0.2839      | 0.2490     | Struggles with tiny fluid footprint resolution.     |
| ❌ 10| Hyperbolic + KL                 | 0.2488      | 0.2477     | Failed. Collapsed to background class.              |

---

## 2. Detailed Experiment Descriptions

### 2.1. Baseline (Zero-Shot)
- **Method:** Standard Unet with ResNet-101 encoder trained only on Cirrus.
- **Goal:** Establish the "Domain Gap" by evaluating on Spectralis without any adaptation.
- **Insight:** The model generalizes surprisingly well to anatomy but loses precision on scanner-specific textures.

### 2.2. FDA (Fourier Domain Adaptation)
- **Method:** Swaps the low-frequency components (amplitudes) of Source images with Target styles using FFT.
- **Configuration:** $L=0.05$.
- **Result:** Successfully closed ~3% of the gap. Very robust because it keeps the high-quality source labels intact.

### 2.3. UDA (Unsupervised Domain Adaptation via Self-Training)
- **Method:** Generate "Pseudo-labels" on Spectralis using the Baseline and then fine-tune.
- **Insight:** Even with confidence filtering (Threshold > 0.95), label noise causes the model to drift away from small structures like IRF.

### 2.4. Edge-Guided FDA
- **Method:** Multi-task learning. The model predicts both fluid masks and retinal layer boundaries (edge maps).
- **Insight:** Added geometric constraints help the model ignore vendor-specific noise but increased complexity slightly lowered the peak Dice.

### 2.5. Hyperbolic Domain Adaptation
- **Method:** Projects features into a Poincaré Ball (Non-Euclidean space).
- **Insight:** Excellent at modeling the hierarchical nature of fluids (PED). Achieved the highest individual class score for PED.

### 2.6. ADVENT (Adversarial Entropy Minimization)
- **Method:** A discriminator tries to distinguish between the "certainty" (entropy) maps of source and target predictions.
- **Insight:** Theoretically sound for structural alignment, but highly unstable in training for this specific OCT dataset.

### 2.7. DSBN (Domain-Specific Batch Normalization)
- **Method:** Uses separate BN layers for each scanner to capture vendor-specific noise statistics while sharing weights.
- **Insight:** Very useful for a single model supporting multiple scanners, but doesn't adapt as aggressively as DANN.

### 2.8. DANN (Domain Adversarial Neural Network)
- **Method:** Uses a Gradient Reversal Layer (GRL) to force the encoder to extract features that the discriminator cannot use to identify the scanner vendor.
- **Insight:** Strong feature alignment but can be unstable during training.

### 2.9. DDSP (Dual Domain Distribution Disruption)
- **Method:** Published in 2024. Stochastically mixes channel-wise statistics (mean/std) between Source and Target in the shallow layers.
- **Insight:** **The Winner.** It forces the model to be distribution-agnostic without the instability of adversarial training. Achieved exceptional results on SRF (0.84) and PED (0.66).

### 2.10. CLUDA (Contrastive Class-Aware Alignment)
- **Method:** Uses InfoNCE loss to pull features of the same class (from different domains) closer together in latent space.
- **Insight:** Highly effective in theory, but limited here by pseudo-label noise on the target domain.

### 2.11. FMC (Fourier Mixup Consistency)
- **Method:** Blends Source and Target styles at random ratios in Fourier space and enforces prediction consistency via KL-divergence.
- **Insight:** Introduced high variance and training instability. Proved that for OCT, aligning feature distributions (DDSP) is superior to enforcing consistency on style augmentations.

---

## 3. Final Recommendation
For production or clinical research, the **DDSP** model (`checkpoints/best_model_ddsp.pth`) is the definitive choice. It provides the highest accuracy across all fluid types, is the most robust to scanner-induced artifacts, and exhibited the most stable training behavior among all unsupervised methods.
