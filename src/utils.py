"""
Utility functions for the Brain Tumor Classification pipeline.

Provides reproducibility seeding, logging, plotting helpers,
and common utility functions used across all modules.
"""

import os
import random
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for full reproducibility across Python, NumPy,
    and PyTorch (CPU + GPU).

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Deterministic algorithms (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, log_file: Optional[Path] = None,
               level: int = logging.INFO) -> logging.Logger:
    """
    Create a configured logger with console and optional file output.

    Args:
        name: Logger name.
        log_file: Optional path to a log file.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy across batches.
    """

    def __init__(self, name: str = "Metric"):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


def count_parameters(model: torch.nn.Module) -> dict:
    """
    Count total and trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with 'total', 'trainable', and 'non_trainable' counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
        "total_mb": total * 4 / (1024 ** 2),  # Approximate size in MB (float32)
    }


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, metrics: dict, filepath: Path) -> None:
    """
    Save a training checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer state.
        epoch: Current epoch number.
        metrics: Dictionary of current metrics.
        filepath: Path to save the checkpoint.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model: torch.nn.Module, filepath: Path,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    device: Optional[torch.device] = None) -> dict:
    """
    Load a training checkpoint.

    Args:
        model: Model to load weights into.
        filepath: Path to the checkpoint file.
        optimizer: Optional optimizer to restore state.
        device: Device to load tensors onto.

    Returns:
        Checkpoint dictionary with epoch and metrics.
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def format_metrics(metrics: dict, prefix: str = "") -> str:
    """
    Format a metrics dictionary into a readable string.

    Args:
        metrics: Dictionary of metric_name -> value.
        prefix: Optional prefix for each line.

    Returns:
        Formatted string.
    """
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{prefix}{key}: {value:.4f}")
        else:
            lines.append(f"{prefix}{key}: {value}")
    return "\n".join(lines)


def denormalize(tensor: torch.Tensor,
                mean: list = [0.485, 0.456, 0.406],
                std: list = [0.229, 0.224, 0.225]) -> torch.Tensor:
    """
    Reverse ImageNet normalization for visualization.

    Args:
        tensor: Normalized image tensor (C, H, W).
        mean: Channel means used for normalization.
        std: Channel stds used for normalization.

    Returns:
        Denormalized tensor clamped to [0, 1].
    """
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)

    if tensor.device != mean_t.device:
        mean_t = mean_t.to(tensor.device)
        std_t = std_t.to(tensor.device)

    return torch.clamp(tensor * std_t + mean_t, 0.0, 1.0)


def show_images(images: list, titles: list = None, cols: int = 4,
                figsize: tuple = (16, 4), save_path: Optional[Path] = None) -> None:
    """
    Display a grid of images.

    Args:
        images: List of numpy arrays or tensors.
        titles: Optional list of titles for each image.
        cols: Number of columns in the grid.
        figsize: Figure size.
        save_path: Optional path to save the figure.
    """
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if rows == 1:
        axes = [axes] if cols == 1 else axes.tolist()
    else:
        axes = [ax for row in axes for ax in row]

    for idx, (ax, img) in enumerate(zip(axes, images)):
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()
        ax.imshow(img, cmap="gray" if img.ndim == 2 or img.shape[-1] == 1 else None)
        if titles and idx < len(titles):
            ax.set_title(titles[idx], fontsize=10)
        ax.axis("off")

    # Hide unused axes
    for ax in axes[len(images):]:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
