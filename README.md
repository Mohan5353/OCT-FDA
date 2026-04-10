# Cross-Vendor OCT Fluid Segmentation via Fourier Domain Adaptation (FDA)

## 📌 Project Overview
This project implements a lightweight, high-impact Domain Adaptation strategy for Medical Image Segmentation. The goal is to train a model on Optical Coherence Tomography (OCT) scans from one hardware manufacturer (Source: Zeiss Cirrus) and ensure it segments accurately on unseen scans from another manufacturer (Target: Heidelberg Spectralis) without requiring target annotations.

Instead of complex adversarial networks, this project uses **Fourier Domain Adaptation (FDA)** as a data augmentation strategy during training. By swapping the low-frequency amplitude spectrums (which encode image "style" and scanner noise) between source and target images, the model is forced to learn scanner-agnostic anatomical structures.

---

## 🤖 Gemini CLI Instructions (Code Generation Manifest)
**To the AI Code Generator:** Use the following file manifest and technical specifications to generate the complete PyTorch codebase. Adhere strictly to the algorithms and library choices specified below.

### Tech Stack Constraints
* **Framework:** PyTorch
* **Segmentation Library:** `segmentation-models-pytorch` (`smp`)
* **Image Processing:** `torch.fft`, `numpy`, `cv2`
* **Metrics:** `torchmetrics` (Dice, IoU)

---

## 📁 Suggested Directory Structure
```text
oct_fda_project/
├── data/
│   ├── Cirrus/           # Source data (Annotated)
│   └── Spectralis/       # Target data (Unannotated for train, Annotated for test)
├── src/
│   ├── dataset.py        # PyTorch Dataset and DataLoader
│   ├── fda.py            # Core Fourier Transform augmentation logic
│   ├── train.py          # Training loop for Baseline and FDA models
│   └── evaluate.py       # Evaluation script for unseen target data
├── requirements.txt
└── README.md
# OCT-FDA
