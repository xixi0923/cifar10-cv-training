"""
CIFAR-10 Computer Vision Training — Main Entry Point.

Usage examples::

    # Train with default settings
    python main.py

    # Train with custom epochs and learning rate
    python main.py --epochs 100 --lr 0.0005

    # Evaluate a saved checkpoint
    python main.py --eval checkpoints/best.pt

    # Run on CPU only
    python main.py --device cpu
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from config import (
    AugmentationConfig,
    DataConfig,
    LogConfig,
    ModelConfig,
    TrainingConfig,
)
from src.data.loader import load_cifar10
from src.models.network import CIFAR10Net
from src.models.train import Trainer
from src.preprocessing.image_process import (
    create_train_augmentation,
    create_val_augmentation,
)
from src.utils.helpers import (
    compute_accuracy,
    confusion_matrix,
    plot_confusion_matrix,
    plot_training_history,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CIFAR-10 Computer Vision Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cuda/cpu/mps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Data
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory")

    # Augmentation
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--rotation", type=int, default=15, help="Max rotation angle (degrees)")
    parser.add_argument("--noise-std", type=float, default=5.0, help="Gaussian noise std")

    # Model
    parser.add_argument("--filters", type=str, default="32,64,128",
                        help="Comma-separated conv filter counts")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")

    # Output
    parser.add_argument("--eval", type=str, default=None,
                        help="Path to checkpoint for evaluation only")
    parser.add_argument("--output", type=str, default="output",
                        help="Output directory for logs and results")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--early-stop", type=int, default=10, help="Early stopping patience (0=off)")

    return parser.parse_args()


def build_configs(args: argparse.Namespace) -> tuple:
    """Build configuration dataclasses from CLI arguments."""
    data_cfg = DataConfig(
        val_split=args.val_split,
        batch_size=args.batch_size,
    )
    if args.data_dir:
        data_cfg.data_dir = Path(args.data_dir)

    aug_cfg = None if args.no_augment else AugmentationConfig(
        random_rotation_range=(-args.rotation, args.rotation),
        gaussian_noise_std=args.noise_std,
    )

    model_cfg = ModelConfig(
        conv_filters=tuple(int(x) for x in args.filters.split(",")),
        dropout_rate=args.dropout,
    )

    train_cfg = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
        save_every=args.save_every,
        early_stopping_patience=args.early_stop,
    )

    log_cfg = LogConfig()

    return data_cfg, aug_cfg, model_cfg, train_cfg, log_cfg


def evaluate(trainer: Trainer, test_loader, class_names: tuple, output_dir: Path) -> None:
    """Run full evaluation: accuracy, confusion matrix, per-class report."""
    logger.info("Running evaluation on test set…")

    preds, labels, probs = trainer.predict(test_loader)

    overall_acc = compute_accuracy(preds, labels)
    logger.info("Test Accuracy: %.2f%%", overall_acc * 100)

    # Confusion matrix
    cm = confusion_matrix(labels, preds, len(class_names))
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, list(class_names), save_path=cm_path)
    logger.info("Confusion matrix saved to %s", cm_path)

    # Per-class accuracy
    logger.info("Per-class accuracy:")
    for i, name in enumerate(class_names):
        mask = labels == i
        if mask.sum() > 0:
            class_acc = (preds[mask] == labels[mask]).mean()
            logger.info("  %-12s: %.2f%%  (%d samples)", name, class_acc * 100, mask.sum())

    return overall_acc


def main() -> None:
    args = parse_args()
    data_cfg, aug_cfg, model_cfg, train_cfg, log_cfg = build_configs(args)

    # Setup
    output_dir = Path(args.output)
    train_cfg.output_dir = output_dir
    train_cfg.checkpoint_dir = output_dir / "checkpoints"
    log_cfg.log_dir = output_dir / "logs"

    setup_logging(log_cfg)
    set_seed(train_cfg.seed)

    logger.info("CIFAR-10 CV Training — Configuration loaded")
    logger.info("Model config: filters=%s, dropout=%.2f", model_cfg.conv_filters, model_cfg.dropout_rate)
    logger.info("Train config: epochs=%d, lr=%.6f, batch_size=%d",
                train_cfg.epochs, train_cfg.learning_rate, data_cfg.batch_size)

    # Data
    train_aug = create_train_augmentation(aug_cfg) if aug_cfg else None
    val_aug = create_val_augmentation()

    train_loader, val_loader, test_loader = load_cifar10(
        data_cfg, transform_fn=train_aug, val_transform_fn=val_aug,
    )

    # Model
    model = CIFAR10Net(model_cfg)
    param_info = model.get_num_params()
    logger.info("Model parameters: %d total, %d trainable", param_info["total"], param_info["trainable"])

    # Trainer
    trainer = Trainer(model, train_cfg)

    # Eval-only mode
    if args.eval:
        trainer.load_checkpoint(args.eval)
        test_loader_no_aug, _, _ = load_cifar10(
            data_cfg, transform_fn=None, val_transform_fn=None,
        )
        evaluate(trainer, test_loader_no_aug, data_cfg.class_names, output_dir)
        return

    # Training
    history = trainer.train(train_loader, val_loader)

    # Save training history plot
    history_path = output_dir / "training_history.png"
    summary = plot_training_history(history, save_path=history_path)
    logger.info("\n%s", summary)

    # Final evaluation on test set
    test_loader_no_aug, _, _ = load_cifar10(
        data_cfg, transform_fn=None, val_transform_fn=None,
    )
    evaluate(trainer, test_loader_no_aug, data_cfg.class_names, output_dir)

    logger.info("All done. Results saved to: %s", output_dir.resolve())


if __name__ == "__main__":
    main()
