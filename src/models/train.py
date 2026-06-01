"""
Training Pipeline.

Handles the full training loop including:
- Forward/backward pass with optional mixed precision
- Learning rate scheduling
- Validation evaluation
- Checkpoint saving & loading
- Early stopping
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import TrainingConfig
from src.utils.helpers import AvgMeter, compute_accuracy

logger = logging.getLogger(__name__)


class Trainer:
    """Configurable training manager.

    Parameters
    ----------
    model : nn.Module
        The neural network to train.
    cfg : TrainingConfig
        Training hyperparameters.
    """

    def __init__(self, model: nn.Module, cfg: TrainingConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.device = self._resolve_device()

        # Move model to device
        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # Loss
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Learning rate scheduler
        self.scheduler = self._build_scheduler()

        # Mixed precision
        self.scaler = GradScaler(enabled=cfg.mixed_precision and self.device.type == "cuda")

        # Tracking
        self.history: dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "lr": [],
        }
        self.best_val_acc: float = 0.0
        self.patience_counter: int = 0

        # Ensure checkpoint directory exists
        self.cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Trainer initialized — device: %s, epochs: %d, lr: %.6f",
            self.device, cfg.epochs, cfg.learning_rate,
        )

    def _resolve_device(self) -> torch.device:
        if self.cfg.device != "auto":
            return torch.device(self.cfg.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        if self.cfg.lr_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.cfg.lr_step_size,
                gamma=self.cfg.lr_gamma,
            )
        elif self.cfg.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.cfg.epochs,
            )
        return None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> dict:
        """Run the full training loop.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader : DataLoader or None

        Returns
        -------
        dict
            Training history with keys: ``train_loss``, ``train_acc``,
            ``val_loss``, ``val_acc``, ``lr``.
        """
        logger.info("=" * 60)
        logger.info("Training started — %d epochs", self.cfg.epochs)
        logger.info("=" * 60)

        for epoch in range(1, self.cfg.epochs + 1):
            epoch_start = time.time()

            # --- Train ---
            train_loss, train_acc = self._train_epoch(train_loader, epoch)

            # --- Validate ---
            val_loss, val_acc = 0.0, 0.0
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader)

            # --- LR step ---
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()

            # --- Record ---
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            epoch_time = time.time() - epoch_start

            logger.info(
                "Epoch [%3d/%d] %.1fs — "
                "train_loss: %.4f  train_acc: %.2f%% | "
                "val_loss: %.4f  val_acc: %.2f%% | lr: %.2e",
                epoch, self.cfg.epochs, epoch_time,
                train_loss, train_acc * 100,
                val_loss, val_acc * 100,
                current_lr,
            )

            # --- Checkpoint ---
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self._save_checkpoint("best.pt", epoch)
                logger.info("  ★ New best val_acc: %.2f%%", val_acc * 100)
            else:
                self.patience_counter += 1

            if self.cfg.save_every > 0 and epoch % self.cfg.save_every == 0:
                self._save_checkpoint(f"epoch_{epoch:03d}.pt", epoch)

            # --- Early stopping ---
            if (self.cfg.early_stopping_patience > 0
                    and self.patience_counter >= self.cfg.early_stopping_patience):
                logger.info(
                    "Early stopping triggered at epoch %d "
                    "(no improvement for %d epochs)",
                    epoch, self.cfg.early_stopping_patience,
                )
                break

        logger.info("=" * 60)
        logger.info("Training complete — best val_acc: %.2f%%", self.best_val_acc * 100)
        return self.history

    def _train_epoch(self, loader: DataLoader, epoch: int) -> tuple:
        """Single training epoch."""
        self.model.train()
        loss_meter = AvgMeter("loss")
        acc_meter = AvgMeter("acc")

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.cfg.mixed_precision and self.device.type == "cuda":
                with autocast(enabled=True):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            acc = compute_accuracy(outputs, labels)
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc, images.size(0))

        return loss_meter.avg, acc_meter.avg

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> tuple:
        """Validation / evaluation pass."""
        self.model.eval()
        loss_meter = AvgMeter("loss")
        acc_meter = AvgMeter("acc")

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            acc = compute_accuracy(outputs, labels)
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc, images.size(0))

        return loss_meter.avg, acc_meter.avg

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def _save_checkpoint(self, filename: str, epoch: int) -> None:
        path = self.cfg.checkpoint_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_acc": self.best_val_acc,
                "history": self.history,
            },
            path,
        )
        logger.debug("Checkpoint saved: %s", path)

    def load_checkpoint(self, filepath: str | Path) -> None:
        """Load model weights and optimizer state from a checkpoint."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        ckpt = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_val_acc = ckpt.get("best_val_acc", 0.0)
        logger.info("Checkpoint loaded from %s (epoch %d)", filepath, ckpt.get("epoch", "?"))

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple:
        """Generate predictions for a dataset.

        Returns
        -------
        all_preds : np.ndarray of shape (N,)
        all_labels : np.ndarray of shape (N,)
        all_probs : np.ndarray of shape (N, num_classes)
        """
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())

        return (
            np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_probs),
        )
