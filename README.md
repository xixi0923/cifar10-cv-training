# CIFAR-10 Computer Vision Training Project

> Based on the CIFAR-10 dataset, using OpenCV for image processing and visual model training.

## Overview

This project focuses on computer vision model training using the **CIFAR-10** dataset and **OpenCV** for image preprocessing and data augmentation. CIFAR-10 consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Tech Stack

| Technology | Purpose |
|-----------|--------|
| Python | Primary programming language |
| OpenCV | Image preprocessing and augmentation |
| CIFAR-10 | Training dataset (10 classes, 60K images) |
| NumPy | Numerical computation |

## Dataset

**CIFAR-10** contains 60,000 32x32 color images:

| Class | Description |
|-------|-------------|
| airplane | |
| automobile | |
| bird | |
| cat | |
| deer | |
| dog | |
| frog | |
| horse | |
| ship | |
| truck | |

- Training set: 50,000 images
- Test set: 10,000 images

## Project Structure

```
.
├── README.md
├── LICENSE
├── .gitignore
└── src/
    ├── data/
    │   └── cifar10/          # Dataset files
    ├── preprocessing/
    │   └── image_process.py  # OpenCV preprocessing
    ├── models/
    │   └── train.py         # Model training
    └── utils/
        └── helpers.py       # Utility functions
```

## Quick Start

### Prerequisites

```bash
pip install opencv-python numpy
```

### Download Dataset

```python
# Using keras (recommended)
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Or download manually from https://www.cs.toronto.edu/~kriz/cifar.html
```

### Run Training

```bash
python src/models/train.py
```

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE). All modifications and distributions must remain open source under the same license.

---

## Repository Optimization Log

The following optimizations were applied on 2026-06-01:

| # | Optimization | Details |
|---|-------------|--------|
| 1 | README rewrite | Added project overview, tech stack, dataset description, project structure, quick start guide, license section |
| 2 | .gitignore added | Excludes Python bytecode, model weights, dataset files, IDE configs, OS files |
| 3 | LICENSE added | GNU General Public License v3.0 (GPL-3.0) |
| 4 | Description optimized | Updated from informal text to professional description |
| 5 | Topics added | `cifar10`, `opencv`, `computer-vision`, `machine-learning`, `python` |
| 6 | Repository renamed | `ai-` → `cifar10-cv-training` (more descriptive) |
