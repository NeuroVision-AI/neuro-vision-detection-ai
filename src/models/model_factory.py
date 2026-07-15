"""
Model factory for brain tumor classification.

Provides a single entry-point — :func:`get_model` — that instantiates the
requested architecture, moves it to the configured device, and returns it
ready for training or inference.  Also exposes utilities for listing
available architectures and printing human-readable model summaries.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from src import config
from src.models.efficientnet import BrainTumorEfficientNet
from src.models.custom_cnn import BrainTumorCNN


# ── Registry ──────────────────────────────────────────────────

_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "efficientnet": BrainTumorEfficientNet,
    "custom_cnn": BrainTumorCNN,
}


# ── Public API ────────────────────────────────────────────────


def get_model(
    model_name: str,
    num_classes: int | None = None,
    pretrained: bool = True,
) -> nn.Module:
    """Instantiate a model by name and move it to the configured device.

    Parameters
    ----------
    model_name : str
        Key identifying the architecture.  Must be one of the names
        returned by :func:`list_available_models` (e.g. ``"efficientnet"``
        or ``"custom_cnn"``).
    num_classes : int, optional
        Number of output classes.  Defaults to ``config.NUM_CLASSES``.
    pretrained : bool, optional
        Whether to use pretrained weights (applicable to transfer-learning
        models only).  Defaults to ``True``.

    Returns
    -------
    nn.Module
        The model placed on ``config.DEVICE``.

    Raises
    ------
    ValueError
        If *model_name* is not found in the registry.
    """
    model_name_lower = model_name.strip().lower()

    if model_name_lower not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available}"
        )

    cls = _MODEL_REGISTRY[model_name_lower]

    # Build keyword arguments (only pass what each class accepts)
    kwargs: dict = {}
    if num_classes is not None:
        kwargs["num_classes"] = num_classes
    else:
        kwargs["num_classes"] = config.NUM_CLASSES

    # `pretrained` is only meaningful for transfer-learning models
    if model_name_lower == "efficientnet":
        kwargs["pretrained"] = pretrained

    model = cls(**kwargs)
    model = model.to(config.DEVICE)
    return model


def list_available_models() -> List[str]:
    """Return a sorted list of registered model names.

    Returns
    -------
    list[str]
        Available model identifiers (e.g. ``["custom_cnn", "efficientnet"]``).
    """
    return sorted(_MODEL_REGISTRY.keys())


def get_model_summary(
    model: nn.Module,
    input_size: tuple[int, ...] = (1, 3, 224, 224),
) -> str:
    """Generate a human-readable summary of the given model.

    Parameters
    ----------
    model : nn.Module
        The model to summarise.
    input_size : tuple[int, ...], optional
        Shape of a dummy input tensor used only for naming.
        Default is ``(1, 3, 224, 224)``.

    Returns
    -------
    str
        Formatted multi-line string containing model name, parameter
        counts, and estimated model size.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    frozen_params = total_params - trainable_params

    # Estimate size: each param is float32 → 4 bytes
    size_mb = (total_params * 4) / (1024 ** 2)

    model_name = model.__class__.__name__
    device = next(model.parameters()).device if total_params > 0 else "N/A"

    lines = [
        "=" * 60,
        f"  Model Summary: {model_name}",
        "=" * 60,
        f"  Input size:          {tuple(input_size)}",
        f"  Device:              {device}",
        "-" * 60,
        f"  Total parameters:    {total_params:>12,}",
        f"  Trainable params:    {trainable_params:>12,}",
        f"  Frozen params:       {frozen_params:>12,}",
        f"  Est. model size:     {size_mb:>11.2f} MB",
        "=" * 60,
    ]
    return "\n".join(lines)
