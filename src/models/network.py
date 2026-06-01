"""
CNN Network Architecture.

Provides a configurable convolutional neural network for CIFAR-10
classification.  Architecture is driven by ``ModelConfig`` so depth and
width can be tuned without modifying this file.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Convolutional block: Conv2d → BatchNorm → ReLU → MaxPool (optional).

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int
    padding : int
    use_pool : bool
        Apply MaxPool2d after activation (default True).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_pool: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2) if use_pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class CIFAR10Net(nn.Module):
    """Configurable CNN for CIFAR-10 classification.

    Architecture (configurable via ``ModelConfig``)::

        [ConvBlock x N] → AdaptiveAvgPool → [FC x M] → Output

    Parameters
    ----------
    cfg : ModelConfig
        Model hyperparameters controlling depth, width, and regularization.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Build convolutional blocks
        conv_blocks = []
        in_ch = cfg.input_channels
        feature_size = cfg.image_size

        for i, out_ch in enumerate(cfg.conv_filters):
            conv_blocks.append(
                ConvBlock(in_ch, out_ch, cfg.conv_kernel_size, cfg.conv_padding)
            )
            in_ch = out_ch
            feature_size = feature_size // 2   # MaxPool halves spatial dims

        self.features = nn.Sequential(*conv_blocks)

        # Adaptive average pooling to handle varying spatial sizes
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Fully-connected classifier head
        fc_layers = []
        flat_size = cfg.conv_filters[-1]
        prev_size = flat_size

        for hidden in cfg.fc_hidden_units:
            fc_layers.append(nn.Linear(prev_size, hidden))
            fc_layers.append(nn.BatchNorm1d(hidden))
            fc_layers.append(nn.ReLU(inplace=True))
            fc_layers.append(nn.Dropout(cfg.dropout_rate))
            prev_size = hidden

        fc_layers.append(nn.Linear(prev_size, cfg.num_classes))
        self.classifier = nn.Sequential(*fc_layers)

        # Log model info
        self._log_model_info()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (B, C, H, W) float32 tensor

        Returns
        -------
        (B, num_classes) float32 logits
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _log_model_info(self) -> None:
        params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("CIFAR10Net — total params: %d, trainable: %d", params, trainable)

    def get_num_params(self) -> dict:
        """Return parameter statistics."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
