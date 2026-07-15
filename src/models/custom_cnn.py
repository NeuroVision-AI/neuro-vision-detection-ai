"""
Lightweight custom CNN for brain tumor classification.

A compact ~2 M-parameter network built from scratch (no pre-training)
with five convolutional blocks, global average pooling, and a two-layer
classifier.  Designed as a fast-training baseline to compare against the
transfer-learning EfficientNet model.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src import config


class BrainTumorCNN(nn.Module):
    """Custom 5-block CNN for 4-class brain tumor classification.

    Architecture
    ------------
    Five convolutional blocks (Conv2d → BatchNorm2d → ReLU → MaxPool2d)
    followed by global average pooling and a two-layer classifier with
    dropout regularisation.

    Parameters
    ----------
    num_classes : int, optional
        Number of output classes (default: ``config.NUM_CLASSES``).
    dropout_rate : float, optional
        Dropout probability in the classifier head
        (default: ``config.CUSTOM_CNN_DROPOUT``).
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        dropout_rate: float = config.CUSTOM_CNN_DROPOUT,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # ── Convolutional feature extractor ───────────────────
        # Block channels: 3 → 32 → 64 → 128 → 256 → 512
        self.block1 = self._make_conv_block(in_channels=3, out_channels=32)
        self.block2 = self._make_conv_block(in_channels=32, out_channels=64)
        self.block3 = self._make_conv_block(in_channels=64, out_channels=128)
        self.block4 = self._make_conv_block(in_channels=128, out_channels=256)
        self.block5 = self._make_conv_block(in_channels=256, out_channels=512)

        # ── Global Average Pooling ────────────────────────────
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # ── Classifier head ──────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

    # ── Building blocks ───────────────────────────────────────

    @staticmethod
    def _make_conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a single convolutional block.

        Conv2d(3×3, pad=1) → BatchNorm2d → ReLU → MaxPool2d(2).

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.

        Returns
        -------
        nn.Sequential
            The assembled convolutional block.
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,  # BN absorbs the bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    # ── Forward pass ──────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, 3, 224, 224)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)``.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.global_avg_pool(x)  # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)   # (B, 512)

        logits = self.classifier(x)  # (B, num_classes)
        return logits

    # ── Grad-CAM support ──────────────────────────────────────

    def get_gradcam_target_layer(self) -> nn.Module:
        """Return the last convolutional block's Conv2d layer for Grad-CAM.

        Returns
        -------
        nn.Module
            The Conv2d layer inside ``self.block5`` (index 0).
        """
        # block5 is Sequential(Conv2d, BN, ReLU, MaxPool) — return the Conv2d
        return self.block5[0]

    # ── Parameter counting ────────────────────────────────────

    @classmethod
    def count_parameters(cls, model: nn.Module | None = None) -> Tuple[int, int]:
        """Count total and trainable parameters.

        Parameters
        ----------
        model : nn.Module, optional
            The model to inspect.  If ``None``, a new ``BrainTumorCNN``
            instance is created with default settings.

        Returns
        -------
        tuple[int, int]
            ``(total_params, trainable_params)``
        """
        if model is None:
            model = cls()

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    # ── Convenience helpers ───────────────────────────────────

    def __repr__(self) -> str:
        total, trainable = self.count_parameters(self)
        return (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}, "
            f"dropout={self.dropout_rate}, "
            f"total_params={total:,}, "
            f"trainable_params={trainable:,})"
        )
