# CIFAR-10 Computer Vision Training Project

> Based on the CIFAR-10 dataset, using OpenCV for image preprocessing and PyTorch for model training. Features configurable CNN architecture, composable data augmentation pipeline, and comprehensive evaluation tools.

## Overview

This project provides a **complete, production-ready training pipeline** for image classification on the CIFAR-10 dataset. It emphasizes:

- **Robustness** — input validation, error handling, reproducible seeds, early stopping
- **Reusability** — modular architecture, config-driven design, composable augmentation
- **Extensibility** — easy to swap models, augmentation strategies, or datasets

## Tech Stack

| Technology | Purpose |
|-----------|--------|
| Python 3.10+ | Primary programming language |
| PyTorch 2.0+ | Model definition, training loop, GPU acceleration |
| OpenCV | Image preprocessing and data augmentation |
| NumPy | Numerical computation |
| Matplotlib | Training curves and confusion matrix visualization |
| Torchvision | CIFAR-10 dataset loading |

## Dataset

**CIFAR-10** contains 60,000 32×32 color images in 10 classes:

| Class | Samples | Class | Samples |
|-------|---------|-------|---------|
| airplane | 6,000 | dog | 6,000 |
| automobile | 6,000 | frog | 6,000 |
| bird | 6,000 | horse | 6,000 |
| cat | 6,000 | ship | 6,000 |
| deer | 6,000 | truck | 6,000 |

- **Training set**: 50,000 images
- **Test set**: 10,000 images

## Project Structure

```
cifar10-cv-training/
├── main.py                    # Entry point: train / evaluate
├── config.py                  # Centralized hyperparameters & paths
├── requirements.txt           # Python dependencies
├── README.md
├── LICENSE                    # GPL-3.0
├── .gitignore
└── src/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   └── loader.py          # CIFAR-10 loading (torchvision + local fallback)
    ├── preprocessing/
    │   ├── __init__.py
    │   └── image_process.py   # OpenCV augmentation pipeline
    ├── models/
    │   ├── __init__.py
    │   ├── network.py         # Configurable CNN architecture
    │   └── train.py           # Training loop, checkpointing, evaluation
    └── utils/
        ├── __init__.py
        └── helpers.py         # Metrics, confusion matrix, logging, plotting
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train (Default Settings)

```bash
python main.py
```

### 3. Train with Custom Parameters

```bash
# Train longer with lower learning rate
python main.py --epochs 100 --lr 0.0005

# Use a wider model
python main.py --filters 64,128,256 --dropout 0.3

# Disable augmentation (for baseline)
python main.py --no-augment

# Run on CPU only
python main.py --device cpu --epochs 20
```

### 4. Evaluate a Saved Checkpoint

```bash
python main.py --eval output/checkpoints/best.pt
```

## Configuration

All hyperparameters are centralized in `config.py` and overridable via CLI:

### Training Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--lr` | 0.001 | Learning rate |
| `--batch-size` | 64 | Batch size |
| `--weight-decay` | 1e-4 | L2 regularization |
| `--device` | auto | auto / cuda / cpu / mps |
| `--seed` | 42 | Random seed for reproducibility |

### Model Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--filters` | 32,64,128 | Conv layer filter counts (comma-separated) |
| `--dropout` | 0.5 | Dropout rate |

### Augmentation Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--no-augment` | false | Disable all data augmentation |
| `--rotation` | 15 | Max rotation angle in degrees |
| `--noise-std` | 5.0 | Gaussian noise standard deviation |

### Output Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--save-every` | 10 | Save checkpoint every N epochs |
| `--early-stop` | 10 | Early stopping patience (0=off) |
| `--output` | output | Output directory |

## Features

### Configurable CNN Architecture

The model is driven by `ModelConfig`, allowing depth/width tuning without code changes:

```
Input (3×32×32)
  → ConvBlock(32)  → ConvBlock(64)  → ConvBlock(128)
  → AdaptiveAvgPool → FC(256) → FC(128) → Output(10)
```

Each `ConvBlock` = Conv2d → BatchNorm → ReLU → MaxPool.

### Composable Augmentation Pipeline

Built on OpenCV, each augmentation is independently configurable:

| Augmentation | Method | Configurable |
|-------------|--------|-------------|
| Random crop | `cv2.copyMakeBorder` + crop | Padding size |
| Horizontal flip | `cv2.flip` | Probability |
| Rotation | `cv2.warpAffine` | Angle range, probability |
| Brightness | `cv2.convertScaleAbs` | Delta range, probability |
| Contrast | `cv2.convertScaleAbs` | Alpha range, probability |
| Gaussian noise | `numpy.random.normal` | Std, probability |

### Training Pipeline

- **AdamW optimizer** with configurable weight decay
- **Label smoothing** (0.1) for better generalization
- **LR scheduling**: step decay or cosine annealing
- **Mixed precision** (AMP) support for faster GPU training
- **Early stopping** with configurable patience
- **Automatic checkpointing** (best + periodic)

### Evaluation

- Overall and per-class accuracy
- Confusion matrix (OpenCV-rendered, no matplotlib required)
- Training curves (loss, accuracy, learning rate)

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE). All modifications and distributions must remain open source under the same license.

---

## Changelog

### v1.0.0 (2026-06-01)

| Change | Details |
|--------|---------|
| Initial codebase | Complete training pipeline with configurable CNN |
| Data module | CIFAR-10 loading via torchvision with local binary fallback |
| Preprocessing | Composable OpenCV augmentation pipeline (6 augmentations) |
| Model | Configurable CNN with BatchNorm, Dropout, AdaptiveAvgPool |
| Training | Full loop with checkpointing, early stopping, LR scheduling, AMP |
| Evaluation | Confusion matrix, per-class accuracy, training curves |
| CLI | Full argument parsing with sensible defaults |
| Config | Centralized dataclass-based configuration |

---

## Repository Optimization Log

| # | Optimization | Details |
|---|-------------|--------|
| 1 | README rewrite | Complete project documentation with usage, config, features |
| 2 | .gitignore added | Python, model weights, datasets, IDE, OS files |
| 3 | LICENSE added | GNU General Public License v3.0 |
| 4 | Description optimized | Professional repository description |
| 5 | Topics added | `cifar10`, `opencv`, `computer-vision`, `machine-learning`, `python` |
| 6 | Repository renamed | `ai-` → `cifar10-cv-training` |
| 7 | Full codebase | 12 source files: CNN, training pipeline, augmentation, evaluation, CLI |
