"""
Configuration module for the Brain Tumor Classification pipeline.

Centralizes all hyperparameters, paths, and settings so that
the entire pipeline can be reconfigured from a single place.
"""

from __future__ import annotations

import os
from pathlib import Path


# ──────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MANIFEST_DIR = DATA_DIR / "manifests"
TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "val"
TEST_DIR = PROCESSED_DATA_DIR / "test"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_SAVE_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"
METRICS_DIR = OUTPUT_DIR / "metrics"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"

# Create directories if they don't exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MANIFEST_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
          MODEL_SAVE_DIR, LOG_DIR, METRICS_DIR, HEATMAP_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# Class Definitions
# ──────────────────────────────────────────────
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

# Class label to index mapping
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Colors for visualisation (per class)
CLASS_COLORS = {
    "glioma": "#E74C3C",       # Red
    "meningioma": "#3498DB",   # Blue
    "pituitary": "#2ECC71",    # Green
    "no_tumor": "#9B59B6",     # Purple
}


# ──────────────────────────────────────────────
# Image Settings
# ──────────────────────────────────────────────
IMG_SIZE = 224           # EfficientNet-B0 native input size
IMG_CHANNELS = 3         # RGB (grayscale MRI stored as 3-channel)
PIXEL_MEAN = [0.485, 0.456, 0.406]   # ImageNet means (for transfer learning)
PIXEL_STD = [0.229, 0.224, 0.225]    # ImageNet stds


# ──────────────────────────────────────────────
# Training Hyperparameters
# ──────────────────────────────────────────────
BATCH_SIZE = 32
# Single-process loading is the portable deterministic default. Set
# AI_NEURO_NUM_WORKERS on hosts that explicitly support shared-memory workers.
NUM_WORKERS = int(os.getenv("AI_NEURO_NUM_WORKERS", "0"))
EPOCHS = 100             # Max epochs (early stopping will likely trigger first)
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1    # Helps with over-confident predictions

# Use one imbalance correction mechanism at a time. Combining a weighted
# sampler with a class-weighted loss double-corrects minority classes.
USE_WEIGHTED_SAMPLER = False
USE_CLASS_WEIGHTED_LOSS = True

# Learning rate scheduler
LR_SCHEDULER = "cosine"  # Options: "cosine", "step", "plateau"
LR_MIN = 1e-6            # Minimum LR for cosine annealing
LR_STEP_SIZE = 10        # For StepLR
LR_GAMMA = 0.5           # For StepLR

# Early stopping
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-4

# Mixed precision training
USE_AMP = True           # Automatic Mixed Precision

# Gradient clipping
GRAD_CLIP_MAX_NORM = 1.0


# ──────────────────────────────────────────────
# Data Augmentation
# ──────────────────────────────────────────────
# Based on literature (Papers #10-17) and medical imaging best practices
AUGMENTATION = {
    "random_rotation": 15,        # ±15 degrees (anatomically safe for axial brain MRI)
    "horizontal_flip": True,      # Valid for axial brain MRI (left-right symmetry)
    "vertical_flip": False,       # NOT valid for brain MRI
    "color_jitter_brightness": 0.2,
    "color_jitter_contrast": 0.2,
    "random_affine_translate": (0.05, 0.05),
    "random_affine_scale": (0.95, 1.05),
    "random_erasing_prob": 0.1,   # Cutout-style regularisation
}


# ──────────────────────────────────────────────
# Model Settings
# ──────────────────────────────────────────────
DEFAULT_MODEL = "efficientnet"   # Options: "efficientnet", "custom_cnn"

# EfficientNet settings
EFFICIENTNET_VARIANT = "efficientnet_b0"
EFFICIENTNET_PRETRAINED = True
EFFICIENTNET_DROPOUT = 0.3
EFFICIENTNET_FREEZE_LAYERS = True   # Freeze early layers for transfer learning

# Custom CNN settings
CUSTOM_CNN_DROPOUT = 0.5

# Model checkpoint naming
CHECKPOINT_BEST_ACC = "best_accuracy.pth"
CHECKPOINT_BEST_LOSS = "best_loss.pth"
CHECKPOINT_LAST = "last_epoch.pth"


# ──────────────────────────────────────────────
# Evaluation Settings
# ──────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.7   # Flag cases below this for human review
TOP_K = 2                    # Show top-K predictions in demo
CALIBRATION_BINS = 15
BOOTSTRAP_SAMPLES = 1000
ABSTENTION_COVERAGE_POINTS = 20


# ──────────────────────────────────────────────
# Grad-CAM / Explainability Settings
# ──────────────────────────────────────────────
GRADCAM_TARGET_LAYER = None  # Auto-detected (last conv layer)
GRADCAM_COLORMAP = "jet"     # Heatmap colormap
GRADCAM_ALPHA = 0.5          # Overlay transparency


# ──────────────────────────────────────────────
# Demo / Gradio Settings
# ──────────────────────────────────────────────
DEMO_PORT = 7860
DEMO_SHARE = False           # Set True for public sharing link


# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
RANDOM_SEED = 42


# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")


def print_config(model_name: str | None = None):
    """Print the current configuration for logging / debugging."""
    print("=" * 60)
    print("  Brain Tumor Classification — Configuration")
    print("=" * 60)
    print(f"  Device:            {DEVICE}")
    print(f"  Project Root:      {PROJECT_ROOT}")
    print(f"  Data Dir:          {PROCESSED_DATA_DIR}")
    print(f"  Classes:           {CLASS_NAMES}")
    print(f"  Image Size:        {IMG_SIZE}×{IMG_SIZE}")
    print(f"  Batch Size:        {BATCH_SIZE}")
    print(f"  Learning Rate:     {LEARNING_RATE}")
    print(f"  Epochs (max):      {EPOCHS}")
    print(f"  Early Stopping:    patience={EARLY_STOPPING_PATIENCE}")
    print(f"  Model:             {model_name or DEFAULT_MODEL}")
    print(f"  Data workers:      {NUM_WORKERS}")
    print(f"  AMP:               {USE_AMP}")
    print(f"  Seed:              {RANDOM_SEED}")
    print("=" * 60)
