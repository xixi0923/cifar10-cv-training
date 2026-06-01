"""
Global configuration for the CIFAR-10 CV training project.

All hyperparameters, paths, and runtime settings are centralized here
so they can be modified without touching source code.
"""

from pathlib import Path
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "cifar10"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "output"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Configuration for CIFAR-10 data loading."""

    data_dir: Path = DATA_DIR
    val_split: float = 0.1          # fraction of training set used for validation
    batch_size: int = 64
    num_workers: int = 2             # DataLoader worker threads
    pin_memory: bool = True          # faster transfer to GPU

    # CIFAR-10 constants
    num_classes: int = 10
    image_size: int = 32             # 32x32 images
    channels: int = 3                # RGB

    # Class label mapping
    class_names: tuple = (
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    )


# ---------------------------------------------------------------------------
# Preprocessing / Augmentation
# ---------------------------------------------------------------------------
@dataclass
class AugmentationConfig:
    """Configuration for OpenCV-based data augmentation.

    Set probability to 0.0 to disable a specific augmentation.
    """

    random_horizontal_flip: float = 0.5
    random_vertical_flip: float = 0.0
    random_rotation_range: tuple = (-15, 15)   # degrees
    random_rotation_prob: float = 0.3
    brightness_delta: float = 20              # OpenCV brightness range [-delta, +delta]
    contrast_alpha: float = 0.1              # contrast factor: 1 +/- alpha
    brightness_prob: float = 0.3
    contrast_prob: float = 0.3
    gaussian_noise_std: float = 5.0           # 0.0 to disable
    gaussian_noise_prob: float = 0.2
    random_crop: bool = True                  # crop to original size with padding
    random_crop_padding: int = 4


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """CNN architecture hyperparameters."""

    conv_filters: tuple = (32, 64, 128)      # filters per conv block
    conv_kernel_size: int = 3
    conv_padding: int = 1
    pool_size: int = 2
    fc_hidden_units: tuple = (256, 128)       # fully-connected layers
    dropout_rate: float = 0.5
    num_classes: int = 10
    input_channels: int = 3


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """Training loop configuration."""

    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    lr_scheduler: str = "step"                 # "step", "cosine", or "none"
    lr_step_size: int = 15
    lr_gamma: float = 0.5
    checkpoint_dir: Path = CHECKPOINT_DIR
    save_every: int = 10                       # save checkpoint every N epochs
    early_stopping_patience: int = 10          # 0 to disable
    output_dir: Path = OUTPUT_DIR
    device: str = "auto"                       # "auto", "cuda", "cpu", "mps"
    seed: int = 42
    mixed_precision: bool = False             # AMP (requires CUDA)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
@dataclass
class LogConfig:
    """Logging configuration."""

    log_dir: Path = OUTPUT_DIR / "logs"
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
