"""
Utility Helpers.

Provides reusable utilities for:
- Running-average metrics (AvgMeter)
- Accuracy computation
- Confusion matrix
- Training curve visualization
- Logging setup
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import LogConfig


# ---------------------------------------------------------------------------
# Average Meter
# ---------------------------------------------------------------------------

class AvgMeter:
    """Tracks the running average of a scalar value.

    Usage::

        meter = AvgMeter("loss")
        for batch in dataloader:
            loss = compute_loss(...)
            meter.update(loss, batch_size)
        print(meter.avg)
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        return f"AvgMeter({self.name!r}, avg={self.avg:.4f}, count={self.count})"


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def compute_accuracy(output: np.ndarray | "torch.Tensor", target: np.ndarray | "torch.Tensor") -> float:
    """Compute top-1 accuracy from model output and ground-truth labels.

    Works with both PyTorch tensors and NumPy arrays.

    Parameters
    ----------
    output : Tensor (B, C) or ndarray
        Raw logits or softmax probabilities.
    target : Tensor (B,) or ndarray
        Ground-truth class indices.

    Returns
    -------
    float
        Fraction of correct predictions in [0, 1].
    """
    if hasattr(output, "detach"):
        output = output.detach().cpu()
        target = target.detach().cpu()

    preds = np.argmax(output, axis=1) if output.ndim > 1 else output
    targets = np.asarray(target).flatten()
    preds = np.asarray(preds).flatten()

    if len(preds) != len(targets):
        raise ValueError(
            f"Prediction count ({len(preds)}) != target count ({len(targets)})"
        )
    return float(np.mean(preds == targets))


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------

def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute confusion matrix.

    Parameters
    ----------
    y_true, y_pred : np.ndarray of shape (N,)
    num_classes : int

    Returns
    -------
    np.ndarray of shape (num_classes, num_classes)
        ``matrix[i][j]`` = number of samples with true label *i*
        predicted as *j*.
    """
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        matrix[int(t)][int(p)] += 1
    return matrix


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Optional[list[str]] = None,
    save_path: Optional[str | Path] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Render a confusion matrix as a color-coded image using OpenCV.

    Parameters
    ----------
    matrix : (C, C) ndarray
    class_names : list of str, optional
    save_path : str or Path, optional
        If provided, save the image to this file.
    normalize : bool
        If True, normalize rows to show proportions.

    Returns
    -------
    np.ndarray
        The BGR image of the confusion matrix.
    """
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix_display = matrix.astype(np.float64) / row_sums
    else:
        matrix_display = matrix.astype(np.float64)

    num_classes = matrix_display.shape[0]

    # Scale to pixel space
    cell_size = 60
    label_margin = 50
    total_w = label_margin + num_classes * cell_size
    total_h = label_margin + num_classes * cell_size

    # Background
    img = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255

    # Draw cells
    for i in range(num_classes):
        for j in range(num_classes):
            val = matrix_display[i][j]
            # Blue channel: higher value = darker blue
            intensity = int(val * 200)
            color = (255 - intensity, 255 - intensity, 255)  # BGR: blue-ish

            x1 = label_margin + j * cell_size
            y1 = label_margin + i * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (80, 80, 80), 1)

            # Value text
            text = f"{val:.2f}" if normalize else str(int(val))
            text_color = (0, 0, 0) if val > 0.5 else (50, 50, 50)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, text, (x1 + 8, y1 + cell_size // 2 + 5),
                        font, 0.35, text_color, 1, cv2.LINE_AA)

    # Class labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx in range(num_classes):
        name = class_names[idx] if class_names and idx < len(class_names) else str(idx)
        # Y-axis labels (rows)
        cv2.putText(img, name, (2, label_margin + idx * cell_size + cell_size // 2 + 5),
                    font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        # X-axis labels (columns)
        cv2.putText(img, name, (label_margin + idx * cell_size + 5, label_margin - 8),
                    font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), img)

    return img


# ---------------------------------------------------------------------------
# Training Curves (text-based for environments without matplotlib)
# ---------------------------------------------------------------------------

def plot_training_history(
    history: dict,
    save_path: Optional[str | Path] = None,
) -> str:
    """Generate a text-based summary of training history.

    For environments where matplotlib is not available, this provides
    a clear terminal-friendly summary.  If matplotlib is available,
    generates an actual plot image.

    Parameters
    ----------
    history : dict
        Must contain keys: ``train_loss``, ``val_loss``,
        ``train_acc``, ``val_acc``, ``lr``.
    save_path : str or Path, optional
        Save plot image here (requires matplotlib).

    Returns
    -------
    str
        Text summary of training history.
    """
    lines = []
    lines.append("=" * 65)
    lines.append("TRAINING HISTORY SUMMARY")
    lines.append("=" * 65)

    epochs = len(history.get("train_loss", []))
    if epochs == 0:
        return "No training history recorded."

    lines.append(f"Total epochs completed: {epochs}")

    if history.get("train_acc"):
        best_train_acc = max(history["train_acc"])
        lines.append(f"Best train accuracy:  {best_train_acc * 100:.2f}%")

    if history.get("val_acc"):
        best_val_acc = max(history["val_acc"])
        best_epoch = history["val_acc"].index(best_val_acc) + 1
        lines.append(f"Best val accuracy:    {best_val_acc * 100:.2f}% (epoch {best_epoch})")
        lines.append(f"Final val accuracy:    {history['val_acc'][-1] * 100:.2f}%")

    if history.get("train_loss"):
        lines.append(f"Final train loss:      {history['train_loss'][-1]:.4f}")

    if history.get("val_loss"):
        lines.append(f"Final val loss:        {history['val_loss'][-1]:.4f}")

    if history.get("lr"):
        lines.append(f"Final learning rate:   {history['lr'][-1]:.2e}")

    lines.append("-" * 65)

    # Try matplotlib for actual plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Loss
        axes[0].plot(history["train_loss"], label="Train Loss", linewidth=1.5)
        if history.get("val_loss"):
            axes[0].plot(history["val_loss"], label="Val Loss", linewidth=1.5)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss Curve")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot([a * 100 for a in history["train_acc"]], label="Train Acc", linewidth=1.5)
        if history.get("val_acc"):
            axes[1].plot([a * 100 for a in history["val_acc"]], label="Val Acc", linewidth=1.5)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Accuracy Curve")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Learning rate
        axes[2].plot(history.get("lr", []), color="orange", linewidth=1.5)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")
        axes[2].set_title("LR Schedule")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
            lines.append(f"Plot saved to: {save_path}")

        plt.close(fig)
        lines.append("Matplotlib plots generated successfully.")

    except ImportError:
        lines.append("(matplotlib not available — text summary only)")

    lines.append("=" * 65)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def setup_logging(cfg: LogConfig) -> None:
    """Configure project-wide logging."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)-20s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if cfg.log_to_console:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    if cfg.log_to_file:
        cfg.log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(cfg.log_dir / "training.log")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Import torch locally to avoid hard dependency at module level
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError:
    torch = None  # type: ignore
