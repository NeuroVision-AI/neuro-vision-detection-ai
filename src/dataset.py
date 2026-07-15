"""
Dataset module for the Brain Tumor Classification pipeline.

Provides a custom PyTorch Dataset for loading brain tumor MRI images
and factory functions for creating DataLoaders with appropriate
transforms and augmentations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

from src import config


class BrainTumorDataset(Dataset):
    """
    Custom Dataset for brain tumor MRI classification.

    Expects data organized in ImageFolder format:
        root/
            class_1/
                img001.jpg
                img002.jpg
            class_2/
                img003.jpg
                ...

    Supports Kaggle Brain Tumor MRI Dataset (4-class: glioma,
    meningioma, no_tumor, pituitary) and any folder-structured
    MRI dataset.
    """

    def __init__(
        self,
        root_dir: Path,
        transform: Optional[transforms.Compose] = None,
        class_names: List[str] = None,
    ):
        """
        Args:
            root_dir: Path to the dataset root (e.g., data/processed/train/).
            transform: Optional torchvision transforms to apply.
            class_names: List of class names. If None, auto-discovered
                         from subdirectory names (sorted alphabetically).
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Auto-discover classes from subdirectories
        if class_names is not None:
            self.class_names = class_names
        else:
            self.class_names = sorted([
                d.name for d in self.root_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ])

        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}

        # Build the list of (image_path, label) pairs
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Scan directories and collect all image file paths with labels."""
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".dcm"}

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            class_idx = self.class_to_idx[class_name]
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append((img_path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and return a single image-label pair.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple of (image_tensor, label_int).
        """
        img_path, label = self.samples[idx]

        # Load image — convert to RGB (handles grayscale MRI stored as single-channel)
        if img_path.suffix.lower() == ".dcm":
            try:
                import pydicom
                from pydicom.pixel_data_handlers.util import apply_voi_lut

                ds = pydicom.dcmread(img_path)
                pixel_array = ds.pixel_array

                # Apply windowing if available
                try:
                    pixel_array = apply_voi_lut(pixel_array, ds)
                except Exception:
                    pass

                # Normalize
                px_min, px_max = pixel_array.min(), pixel_array.max()
                if px_max > px_min:
                    normalized = (pixel_array - px_min) / (px_max - px_min)
                else:
                    normalized = np.zeros_like(pixel_array)

                uint8_img = (normalized * 255).astype(np.uint8)
                image = Image.fromarray(uint8_img).convert("RGB")
            except Exception as e:
                # A synthetic black image would silently corrupt both training and
                # evaluation. Invalid inputs must be fixed during the manifest audit.
                raise RuntimeError(f"Failed to decode DICOM {img_path}: {e}") from e
        else:
            image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """Return a dictionary mapping class names to sample counts."""
        dist = {name: 0 for name in self.class_names}
        for _, label in self.samples:
            dist[self.idx_to_class[label]] += 1
        return dist

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for balanced training.

        Returns:
            Tensor of shape (num_classes,) with weights on the configured device.
        """
        dist = self.get_class_distribution()
        total = sum(dist.values())
        num_classes = len(self.class_names)

        weights = []
        for name in self.class_names:
            count = dist[name]
            # Inverse frequency: total / (num_classes * count)
            w = total / (num_classes * count) if count > 0 else 1.0
            weights.append(w)

        return torch.tensor(weights, dtype=torch.float32, device=config.DEVICE)

    def get_sample_weights(self) -> torch.Tensor:
        """
        Compute per-sample weights for WeightedRandomSampler.

        Returns:
            Tensor of shape (num_samples,) with a weight for each sample.
        """
        class_weights = self.get_class_weights().cpu().numpy()
        sample_weights = [class_weights[label] for _, label in self.samples]
        return torch.tensor(sample_weights, dtype=torch.float64)


# ──────────────────────────────────────────────
# Transform Pipelines
# ──────────────────────────────────────────────

def get_train_transforms(img_size: int = config.IMG_SIZE) -> transforms.Compose:
    """
    Build training augmentation pipeline.

    Uses medically appropriate augmentations for brain MRI:
    - Rotation (±15°): safe for axial brain MRI
    - Horizontal flip: valid due to left-right brain symmetry
    - NO vertical flip: anatomically invalid for brain
    - Mild colour jitter: simulates scanner variation
    - Random affine: small translations and scaling
    - Random erasing: cutout-style regularisation

    Args:
        img_size: Target image size (square).

    Returns:
        Composed transform pipeline.
    """
    aug = config.AUGMENTATION
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=aug["random_rotation"]),
        transforms.RandomHorizontalFlip(p=0.5 if aug["horizontal_flip"] else 0.0),
        transforms.RandomAffine(
            degrees=0,
            translate=aug["random_affine_translate"],
            scale=aug["random_affine_scale"],
        ),
        transforms.ColorJitter(
            brightness=aug["color_jitter_brightness"],
            contrast=aug["color_jitter_contrast"],
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.PIXEL_MEAN, std=config.PIXEL_STD),
        transforms.RandomErasing(p=aug["random_erasing_prob"]),
    ])


def get_val_transforms(img_size: int = config.IMG_SIZE) -> transforms.Compose:
    """
    Build validation / test transform pipeline (no augmentation).

    Args:
        img_size: Target image size (square).

    Returns:
        Composed transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.PIXEL_MEAN, std=config.PIXEL_STD),
    ])


# ──────────────────────────────────────────────
# DataLoader Factory
# ──────────────────────────────────────────────

def get_dataloaders(
    train_dir: Path = config.TRAIN_DIR,
    val_dir: Path = config.VAL_DIR,
    test_dir: Path = config.TEST_DIR,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    use_weighted_sampler: bool = config.USE_WEIGHTED_SAMPLER,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Args:
        train_dir: Path to training data.
        val_dir: Path to validation data.
        test_dir: Path to test data.
        batch_size: Batch size.
        num_workers: Number of worker processes.
        use_weighted_sampler: If True, use WeightedRandomSampler for
                              class-balanced training batches.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Create datasets
    train_dataset = BrainTumorDataset(
        root_dir=train_dir,
        transform=get_train_transforms(),
    )
    val_dataset = BrainTumorDataset(
        root_dir=val_dir,
        transform=get_val_transforms(),
    )
    test_dataset = BrainTumorDataset(
        root_dir=test_dir,
        transform=get_val_transforms(),
    )

    # Print dataset info
    print(f"\n{'='*50}")
    print("  Dataset Summary")
    print(f"{'='*50}")
    print(f"  Train: {len(train_dataset):,} images")
    print(f"  Val:   {len(val_dataset):,} images")
    print(f"  Test:  {len(test_dataset):,} images")
    print(f"\n  Training class distribution:")
    for cls_name, count in train_dataset.get_class_distribution().items():
        pct = 100 * count / len(train_dataset) if len(train_dataset) > 0 else 0
        print(f"    {cls_name:15s}: {count:5d} ({pct:.1f}%)")
    print(f"{'='*50}\n")

    # Training DataLoader (with optional weighted sampling)
    pin_memory = config.DEVICE.type == "cuda"
    if use_weighted_sampler and len(train_dataset) > 0:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

    # Validation DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Test DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def prepare_data_from_single_folder(
    source_dir: Path,
    output_dir: Path = config.PROCESSED_DATA_DIR,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = config.RANDOM_SEED,
) -> None:
    """Reject the retired image-level splitter.

    The parameters remain only to provide a clear failure for older callers.
    """
    raise RuntimeError(
        "Legacy image-level splitting is disabled because it can leak patients or duplicates. "
        "Use scripts/prepare_research_data.py and retain its manifest/audit artifacts."
    )
