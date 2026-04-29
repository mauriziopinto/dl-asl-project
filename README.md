# ASL Alphabet Recognition — Deep Learning Systems Project

## Overview

This project compares two deep learning approaches for American Sign Language (ASL) alphabet recognition using image classification:

1. **Baseline:** A custom CNN trained from scratch (4 convolutional blocks)
2. **Experiment:** ResNet18 with pretrained ImageNet weights (frozen backbone, trained classifier only)

Both models are trained and evaluated on the ASL Alphabet dataset (87,000 images, 29 classes) using PyTorch.

## Dataset

**ASL Alphabet** from Kaggle: https://www.kaggle.com/datasets/grassknoted/asl-alphabet

- 87,000 training images (3,000 per class × 29 classes)
- Classes: A–Z, space, delete, nothing
- Image size: 200×200 RGB

Download and extract `asl_alphabet_train/` into the `dataset/` directory so the structure is:

```
dataset/
└── asl_alphabet_train/
    ├── A/
    ├── B/
    ├── ...
    └── nothing/
```

## Setup

```bash
# Create and activate the virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv sync
```

## Running the Notebook

```bash
jupyter notebook deep_learning.ipynb
```

Run all cells top-to-bottom. Training the custom CNN takes ~10–15 minutes on an NVIDIA 4070 Ti. The ResNet18 experiment takes ~5 minutes (frozen backbone means fewer parameters to update).

## Results

| Model | Test Accuracy |
|-------|--------------|
| Custom CNN (baseline) | 99.67% |
| ResNet18 (transfer learning) | 98.28% |

The custom CNN outperformed ResNet18 with a frozen backbone, likely because the ASL alphabet task differs substantially from ImageNet — the pretrained features were less transferable than learning domain-specific features from scratch on a large dataset.

## Bias Awareness

The ASL Alphabet dataset was collected in a controlled environment with uniform backgrounds and consistent lighting. The dataset likely has limited diversity in skin tone, hand size, and age. A model trained on this data may not generalize well to real-world settings with diverse users. For accessibility applications, testing across a wide range of hand appearances and environments is essential.

## Dependencies

See `requirements.txt` for the full list of pinned packages. Key libraries:

- Python 3.11+
- PyTorch + torchvision
- NumPy, pandas, matplotlib, seaborn
- scikit-learn (for confusion matrix)
- Jupyter

## Future Improvements

- Data augmentation (random rotation, brightness, varied backgrounds)
- Fine-tune all ResNet18 layers (unfreeze backbone with lower learning rate)
- Collect real-world test data with diverse hand appearances
- Expand to dynamic signs and full ASL vocabulary
