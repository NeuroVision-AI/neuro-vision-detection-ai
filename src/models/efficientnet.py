"""
EfficientNet-B0 transfer learning model for brain tumor classification.

Loads an ImageNet-pretrained EfficientNet-B0 via the `timm` library
and replaces the classifier head for 4-class brain tumor prediction.
Supports selective layer freezing for staged fine-tuning and exposes
the target layer needed by Grad-CAM for explainability.
"""

from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn

from src import config


class BrainTumorEfficientNet(nn.Module):
    """EfficientNet-B0 adapted for brain tumor classification.

    Parameters
    ----------
    num_classes : int, optional
        Number of output classes (default: ``config.NUM_CLASSES``).
    pretrained : bool, optional
        Whether to load ImageNet-pretrained weights (default: ``config.EFFICIENTNET_PRETRAINED``).
    dropout_rate : float, optional
        Dropout probability before the classifier head
        (default: ``config.EFFICIENTNET_DROPOUT``).
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = config.EFFICIENTNET_PRETRAINED,
        dropout_rate: float = config.EFFICIENTNET_DROPOUT,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Load the EfficientNet-B0 backbone (without the original classifier)
        self.backbone = timm.create_model(
            config.EFFICIENTNET_VARIANT,
            pretrained=pretrained,
            num_classes=0,           # removes the original head
            global_pool="avg",       # keep global average pooling
        )

        # Number of features produced by the backbone
        self.num_features: int = self.backbone.num_features  # 1280 for B0

        # Custom classifier head
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(self.num_features, num_classes)

        # Optionally freeze early layers on construction
        if config.EFFICIENTNET_FREEZE_LAYERS:
            self.freeze_backbone()

    # ── Forward pass ──────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features and classify.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, 3, 224, 224)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)``.
        """
        features = self.backbone(x)          # (B, 1280)
        features = self.dropout(features)
        logits = self.classifier(features)   # (B, num_classes)
        return logits

    # ── Layer freezing utilities ──────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze all backbone layers except the classifier head and the last 2 blocks.

        This is the recommended starting point for transfer learning:
        only the task-specific head and the highest-level feature
        extractors are trainable.
        """
        # First, freeze everything in the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last 2 EfficientNet blocks
        # timm stores the blocks in `self.backbone.blocks`
        if hasattr(self.backbone, "blocks"):
            num_blocks = len(self.backbone.blocks)
            for block in self.backbone.blocks[max(0, num_blocks - 2):]:
                for param in block.parameters():
                    param.requires_grad = True

        # The classifier head is always trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Unfreeze every parameter for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    # ── Grad-CAM support ──────────────────────────────────────

    def get_gradcam_target_layer(self) -> nn.Module:
        """Return the last convolutional layer for Grad-CAM visualisation.

        For EfficientNet-B0 loaded via ``timm`` this is the final
        convolutional block inside ``self.backbone.blocks``.

        Returns
        -------
        nn.Module
            The target layer suitable for hooking by Grad-CAM.
        """
        # The last block in timm's EfficientNet contains the deepest
        # convolutional features before the global pool.
        return self.backbone.blocks[-1]

    # ── Convenience helpers ───────────────────────────────────

    def __repr__(self) -> str:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}, "
            f"dropout={self.dropout_rate}, "
            f"trainable_params={trainable:,}, "
            f"total_params={total:,})"
        )
