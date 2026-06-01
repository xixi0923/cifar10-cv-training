"""
OpenCV Image Preprocessing & Data Augmentation Module.

All transformations operate on **single images** with the signature::

    input:  np.ndarray of shape (H, W, C), dtype uint8
    output: np.ndarray of shape (H, W, C), dtype uint8

This design keeps augmentation composable and framework-agnostic.
"""

import logging
import random
from typing import Optional

import cv2
import numpy as np

from config import AugmentationConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual augmentation functions
# ---------------------------------------------------------------------------

def horizontal_flip(image: np.ndarray) -> np.ndarray:
    """Randomly flip the image horizontally (p=0.5)."""
    if random.random() < 0.5:
        return cv2.flip(image, 1)
    return image


def vertical_flip(image: np.ndarray) -> np.ndarray:
    """Randomly flip the image vertically (p=0.5)."""
    if random.random() < 0.5:
        return cv2.flip(image, 0)
    return image


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate the image by *angle* degrees around the center.

    Uses ``cv2.warpAffine`` with ``BORDER_REFLECT_101`` padding to avoid
    introducing black borders.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def adjust_brightness(image: np.ndarray, delta: int = 20) -> np.ndarray:
    """Add a random integer offset in [-delta, +delta] to each pixel.

    Uses OpenCV ``convertScaleAbs`` for efficient, clamped arithmetic.
    """
    offset = random.randint(-delta, delta)
    return cv2.convertScaleAbs(image, alpha=1.0, beta=offset)


def adjust_contrast(image: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Scale pixel values by ``(1 + random_uniform(-alpha, alpha))``.

    Simulates contrast change around the original level.
    """
    factor = 1.0 + random.uniform(-alpha, alpha)
    return cv2.convertScaleAbs(image, alpha=factor, beta=0.0)


def add_gaussian_noise(image: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Add zero-mean Gaussian noise with standard deviation *sigma*."""
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def random_crop(image: np.ndarray, padding: int = 4) -> np.ndarray:
    """Pad the image by *padding* pixels then crop back to original size
    at a random offset.

    Padding uses ``cv2.BORDER_REFLECT_101`` to keep content realistic.
    """
    h, w = image.shape[:2]
    padded = cv2.copyMakeBorder(
        image, padding, padding, padding, padding, cv2.BORDER_REFLECT_101
    )
    y = random.randint(0, 2 * padding)
    x = random.randint(0, 2 * padding)
    return padded[y:y + h, x:x + w].copy()


# ---------------------------------------------------------------------------
# Normalization (always applied)
# ---------------------------------------------------------------------------

def normalize(image: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalize image channels: ``(pixel - mean) / std``.

    Parameters
    ----------
    image : (H, W, C) float32 in [0, 1]
    mean, std : (C,) float32

    Returns
    -------
    np.ndarray of same shape, float32
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    return (image - mean[None, None, :]) / std[None, None, :]


# Standard CIFAR-10 per-channel mean and std
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


# ---------------------------------------------------------------------------
# Composable augmentation pipeline
# ---------------------------------------------------------------------------

class AugmentationPipeline:
    """Build a composable augmentation pipeline from configuration.

    Each augmentation is applied independently with its own probability,
    making the pipeline easy to customize and extend.

    Usage::

        pipeline = AugmentationPipeline(AugmentationConfig())
        augmented = pipeline(image)   # (H, W, C) uint8 -> (H, W, C) uint8

    To add a custom augmentation, append a ``(callable, probability)`` pair
    to ``pipeline.steps``.
    """

    def __init__(self, cfg: Optional[AugmentationConfig] = None) -> None:
        cfg = cfg or AugmentationConfig()
        self.steps: list[tuple[callable, float]] = []

        self._build_from_config(cfg)
        logger.info("Augmentation pipeline built with %d steps", len(self.steps))

    def _build_from_config(self, cfg: AugmentationConfig) -> None:
        """Register augmentation steps based on configuration."""

        if cfg.random_crop:
            self.add_step(
                lambda img: random_crop(img, cfg.random_crop_padding), 1.0
            )

        if cfg.random_horizontal_flip > 0:
            self.add_step(horizontal_flip, cfg.random_horizontal_flip)

        if cfg.random_vertical_flip > 0:
            self.add_step(vertical_flip, cfg.random_vertical_flip)

        if cfg.random_rotation_prob > 0 and cfg.random_rotation_range != (0, 0):
            lo, hi = cfg.random_rotation_range
            self.add_step(
                lambda img: rotate(img, random.uniform(lo, hi)),
                cfg.random_rotation_prob,
            )

        if cfg.brightness_prob > 0:
            self.add_step(
                lambda img: adjust_brightness(img, cfg.brightness_delta),
                cfg.brightness_prob,
            )

        if cfg.contrast_prob > 0:
            self.add_step(
                lambda img: adjust_contrast(img, cfg.contrast_alpha),
                cfg.contrast_prob,
            )

        if cfg.gaussian_noise_prob > 0 and cfg.gaussian_noise_std > 0:
            self.add_step(
                lambda img: add_gaussian_noise(img, cfg.gaussian_noise_std),
                cfg.gaussian_noise_prob,
            )

    def add_step(self, func: callable, probability: float) -> None:
        """Append an augmentation step.

        Parameters
        ----------
        func : callable
            Takes a single (H, W, C) uint8 image and returns one of the same.
        probability : float
            Probability of applying this step (0.0 – 1.0).
        """
        if not (0.0 <= probability <= 1.0):
            raise ValueError(f"Probability must be [0, 1], got {probability}")
        self.steps.append((func, probability))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply all registered augmentations to *image*."""
        result = image.copy()
        for func, prob in self.steps:
            if random.random() < prob:
                result = func(result)
        return result


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_train_augmentation(
    cfg: Optional[AugmentationConfig] = None,
) -> AugmentationPipeline:
    """Create the training augmentation pipeline."""
    return AugmentationPipeline(cfg)


def create_val_augmentation() -> callable:
    """Create a no-op transform for validation/test (identity function)."""
    return lambda img: img.copy()
