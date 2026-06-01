"""
CIFAR-10 Data Loader Module.

Handles downloading, loading, splitting, and batching the CIFAR-10 dataset.
Uses torchvision for standardized access; supports local caching.
"""

import logging
import random
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from config import DataConfig

logger = logging.getLogger(__name__)


class CIFAR10NumpyDataset(Dataset):
    """A PyTorch Dataset wrapping CIFAR-10 numpy arrays.

    This is the bridge between raw numpy arrays (from torchvision or local
    files) and PyTorch's DataLoader.  Preprocessing callbacks can be
    attached via ``transform_fn``.

    Parameters
    ----------
    images : np.ndarray
        Array of shape ``(N, H, W, C)`` with dtype ``uint8``.
    labels : np.ndarray
        Array of shape ``(N,)`` with dtype ``int64``.
    transform_fn : callable or None
        A function that takes a single image (H, W, C) uint8 numpy array
        and returns a transformed version of the same shape and dtype.
        Useful for OpenCV-based augmentation.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform_fn: Optional[callable] = None,
    ) -> None:
        self.images = images
        self.labels = labels
        self.transform_fn = transform_fn

        # Input validation
        if len(images) != len(labels):
            raise ValueError(
                f"Image count ({len(images)}) != label count ({len(labels)})"
            )
        if images.ndim != 4:
            raise ValueError(
                f"Expected images with shape (N, H, W, C), got {images.shape}"
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx].copy()       # (H, W, C) uint8
        label = self.labels[idx]

        if self.transform_fn is not None:
            image = self.transform_fn(image)

        # Convert HWC uint8 -> CHW float32 [0, 1]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
        return torch.from_numpy(image), torch.tensor(label, dtype=torch.long)


def load_cifar10_from_local(
    data_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIFAR-10 from local binary files (Alex Krizhevsky format).

    Expected directory structure::

        data_dir/
        ├── data_batch_1.bin
        ├── data_batch_2.bin
        ├── ...
        ├── data_batch_5.bin
        └── test_batch.bin

    Each file contains 10000 samples in row-major order:
    ``<1 x label><3072 x pixel>`` where pixels are in R,G,B order.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the binary batch files.

    Returns
    -------
    train_images, train_labels, test_images, test_labels : np.ndarray
    """
    import os

    data_dir = os.path.expanduser(str(data_dir))

    def _load_batch(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        with open(filepath, "rb") as f:
            raw = np.frombuffer(f.read(), dtype=np.uint8)
        labels = raw[::3073]
        images = raw[1::3073].reshape(-1, 3, 32, 32)
        images = images.transpose(0, 2, 3, 1)   # -> (N, H, W, C)
        return images, labels

    train_images_list = []
    train_labels_list = []
    for i in range(1, 6):
        path = os.path.join(data_dir, f"data_batch_{i}.bin")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"CIFAR-10 batch file not found: {path}. "
                "Download from https://www.cs.toronto.edu/~kriz/cifar.html"
            )
        imgs, lbls = _load_batch(path)
        train_images_list.append(imgs)
        train_labels_list.append(lbls)

    test_path = os.path.join(data_dir, "test_batch.bin")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"CIFAR-10 test file not found: {test_path}. "
            "Download from https://www.cs.toronto.edu/~kriz/cifar.html"
        )
    test_images, test_labels = _load_batch(test_path)

    train_images = np.concatenate(train_images_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)

    logger.info(
        "Loaded CIFAR-10 from local: %d train, %d test samples",
        len(train_images), len(test_images),
    )
    return train_images, train_labels, test_images, test_labels


def load_cifar10(
    cfg: DataConfig,
    transform_fn: Optional[callable] = None,
    val_transform_fn: Optional[callable] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Load CIFAR-10 and return DataLoaders.

    Tries to load from torchvision (auto-download if missing), then falls
    back to local binary files.

    Parameters
    ----------
    cfg : DataConfig
        Data configuration.
    transform_fn : callable or None
        Augmentation applied to **training** images.
    val_transform_fn : callable or None
        Augmentation applied to **validation** images.  If ``None``,
        no augmentation is applied to validation data.

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader (or test_loader if val_split == 0)
    test_loader : DataLoader
    """

    train_images, train_labels, test_images, test_labels = _attempt_load(cfg)

    # Build datasets
    train_dataset = CIFAR10NumpyDataset(
        train_images, train_labels, transform_fn=transform_fn,
    )
    test_dataset = CIFAR10NumpyDataset(
        test_images, test_labels, transform_fn=val_transform_fn,
    )

    # Train/val split
    if cfg.val_split > 0:
        val_size = int(len(train_dataset) * cfg.val_split)
        train_size = len(train_dataset) - val_size
        train_subset, val_subset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg.val_split * 10000),  # type: ignore
        )
        # Apply val_transform to the val subset
        val_dataset = _SubsetWithTransform(val_subset, val_transform_fn)

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
    else:
        train_subset = train_dataset
        val_loader = None

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    logger.info(
        "DataLoaders ready — train: %d, val: %s, test: %d batches",
        len(train_loader),
        len(val_loader) if val_loader else "N/A",
        len(test_loader),
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _SubsetWithTransform(Dataset):
    """A Subset wrapper that applies a transform to every item."""

    def __init__(self, subset: Subset, transform_fn: Optional[callable]) -> None:
        self.subset = subset
        self.transform_fn = transform_fn

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self.subset[idx]
        if self.transform_fn is not None:
            # tensor (C, H, W) float32 -> numpy (H, W, C) uint8 -> augment -> back
            img_np = (
                image.numpy()
                .transpose(1, 2, 0)
                .clip(0, 1)
                .astype(np.uint8)
            )
            img_np = self.transform_fn(img_np)
            img_np = np.transpose(img_np, (2, 0, 1)).astype(np.float32) / 255.0
            image = torch.from_numpy(img_np)
        return image, label


def _attempt_load(cfg: DataConfig) -> Tuple:
    """Try torchvision first, then local binary files."""
    try:
        return _load_via_torchvision(cfg)
    except Exception as exc:
        logger.warning(
            "torchvision load failed (%s), trying local files…", exc
        )
        return load_cifar10_from_local(cfg.data_dir)


def _load_via_torchvision(
    cfg: DataConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIFAR-10 via torchvision.datasets.CIFAR10."""
    from torchvision.datasets import CIFAR10 as TVCIFAR10

    root = str(cfg.data_dir.parent)
    train_set = TVCIFAR10(root=root, train=True, download=True)
    test_set = TVCIFAR10(root=root, train=False, download=True)

    train_images = train_set.data                   # (50000, 32, 32, 3) uint8
    train_labels = np.array(train_set.targets)
    test_images = test_set.data
    test_labels = np.array(test_set.targets)

    logger.info(
        "Loaded CIFAR-10 via torchvision: %d train, %d test",
        len(train_images), len(test_images),
    )
    return train_images, train_labels, test_images, test_labels
