# GEMINI.md - Project Context

## Project Overview
**Name:** Cross-Vendor OCT Fluid Segmentation via Fourier Domain Adaptation (FDA)
**Domain:** Medical Image Segmentation (Optical Coherence Tomography - OCT)
**Objective:** Train a model on annotated OCT scans from one manufacturer (Source: Zeiss Cirrus) and ensure it generalizes to unannotated scans from another manufacturer (Target: Heidelberg Spectralis) without requiring target-domain labels.

### Core Methodology
The project employs **Fourier Domain Adaptation (FDA)** as a style-transfer and data augmentation strategy. By swapping low-frequency components (amplitude spectrum) of source images with those of target images, the model learns to be invariant to scanner-specific "styles" and noise while focusing on invariant anatomical features.

---

## Technical Specifications & Tech Stack
- **Framework:** PyTorch
- **Segmentation Models:** `segmentation-models-pytorch` (`smp`)
- **Core Algorithm:** Fourier Transform via `torch.fft`
- **Data Processing:** `numpy`, `opencv-python` (`cv2`)
- **Evaluation Metrics:** `torchmetrics` (Dice Score, Intersection over Union - IoU)

---

## Dataset Configuration
- **Dataset Root:** `./data/`
- **Structure:**
  - `data/Cirrus/`: Contains `cropped_images`, `cropped_masks`, `denoised_images`, and `edge_map_images` for the Zeiss Cirrus vendor.
  - `data/Spectralis/`: Contains `cropped_images`, `cropped_masks`, `denoised_images`, and `edge_map_images` for the Heidelberg Spectralis vendor.
  - `data/Topcon/`: Contains `cropped_images`, `cropped_masks`, `denoised_images`, and `edge_map_images` for the Topcon vendor.
  - `data/slice_gt.csv`: Metadata and labels for each slice.

---

## Current Project Structure
```text
oct_fda_project/
├── data/
│   ├── Cirrus/
│   ├── Spectralis/
│   ├── Topcon/
│   └── slice_gt.csv
├── MICCAI_RETOUCH/       # Original dataset folder (with tree.txt)
├── README.md
└── GEMINI.md
```

---

## Intended Directory Structure (TODO: Implement)
The project is currently in the initial phase. The planned structure is as follows:
```text
oct_fda_project/
├── data/
│   ├── Cirrus/           # Source data (Annotated)
│   └── Spectralis/       # Target data (Unannotated for train, Annotated for test)
├── src/
│   ├── dataset.py        # PyTorch Dataset and DataLoader implementations
│   ├── fda.py            # Core Fourier Transform augmentation logic
│   ├── train.py          # Training scripts for baseline and FDA models
│   └── evaluate.py       # Evaluation scripts for cross-domain validation
├── requirements.txt      # Project dependencies
├── README.md             # Project overview and AI generation manifest
└── GEMINI.md             # This instruction file
```

---

## Development & Implementation Guidelines
1.  **Surgical Changes:** When implementing the `src/` modules, ensure each file handles its specific responsibility (e.g., `fda.py` should only contain the FFT logic).
2.  **Validation:** Use `torchmetrics` to consistently evaluate performance across both Source-to-Source (Upper bound) and Source-to-Target (Domain Gap) scenarios.
3.  **Efficiency:** Prioritize using `torch.fft` for performance during training-time augmentations.
4.  **Reproducibility:** Fix seeds for both `numpy` and `torch` to ensure reproducible domain adaptation experiments.

---

## Usage Instructions

### 1. Setup
Install the necessary dependencies:
```bash
pip install -r requirements.txt
```

### 2. Training (Baseline)
Train a model on the source domain (Cirrus) without any adaptation:
```bash
python src/train.py --data_root data --epochs 20 --batch_size 8
```

### 3. Training (with FDA)
Train a model on the source domain (Cirrus) while adapting to the target domain (Spectralis) style:
```bash
python src/train.py --data_root data --epochs 20 --batch_size 8 --use_fda --fda_L 0.01
```

### 4. Evaluation
Evaluate the model on the target domain (Spectralis):
```bash
python src/evaluate.py --data_root data --vendor Spectralis --checkpoint checkpoints/best_model_fda.pth
```

---

## Roadmap & Status
- [x] Dataset rearrangement by vendor.
- [x] Initialized `requirements.txt`.
- [x] Implemented `src/dataset.py` for multi-vendor loading.
- [x] Implemented `src/fda.py` for spectrum swapping logic.
- [x] Implemented `src/train.py` for both Baseline and FDA training.
- [x] Implemented `src/evaluate.py` for performance validation.

