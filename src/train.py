"""
Training pipeline for Brain Tumor Classification.

Includes:
    - EarlyStopping with checkpoint management
    - Inverse-frequency class weighting
    - Trainer with AdamW, CosineAnnealingLR, AMP, gradient clipping,
      TensorBoard logging, and checkpoint saving
    - CLI entry-point via argparse
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.dataset import BrainTumorDataset, get_dataloaders
from src.data_integrity import group_ids_for_materialized_samples
from src.models.model_factory import get_model
from src.utils import seed_everything


# ──────────────────────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stop training when the validation loss stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement before stopping.
    min_delta : float
        Minimum decrease in validation loss to qualify as improvement.
    """

    def __init__(
        self,
        patience: int = config.EARLY_STOPPING_PATIENCE,
        min_delta: float = config.EARLY_STOPPING_MIN_DELTA,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float | None = None
        self.counter: int = 0

    # ── public interface ──

    def __call__(self, val_loss: float) -> bool:
        """Return ``True`` if training should stop."""
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            print(
                f"\n[EarlyStopping] No improvement for {self.patience} "
                f"epochs. Stopping training."
            )
            return True
        return False

    # ── checkpoint helpers ──

    @staticmethod
    def save_checkpoint(
        model: nn.Module,
        path: Path | str,
        metadata: dict | None = None,
    ) -> None:
        """Save model weights with the minimum provenance needed for inference."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
                "model_state_dict": model.state_dict(),
                "class_names": list(config.CLASS_NAMES),
                "image_size": config.IMG_SIZE,
                "random_seed": config.RANDOM_SEED,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "intended_use": "research-only 2D public-data proof-of-concept",
            }
        if metadata:
            payload.update(metadata)
        torch.save(payload, path)
        print(f"  ✓ Checkpoint saved → {path.name}")

    @staticmethod
    def load_checkpoint(model: nn.Module, path: Path | str) -> nn.Module:
        """Load model state dict from *path*."""
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state)
        print(f"  ✓ Checkpoint loaded ← {path.name}")
        return model


# ──────────────────────────────────────────────────────────────
# Class-weight computation
# ──────────────────────────────────────────────────────────────

def compute_class_weights(dataset: BrainTumorDataset) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from a training dataset.

    Returns a 1-D tensor of shape ``(NUM_CLASSES,)`` on ``config.DEVICE``.
    """
    # Gather all labels
    if hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
    else:
        # Fallback: iterate the dataset (slow but universal)
        labels = [dataset[i][1] for i in range(len(dataset))]

    counts = Counter(labels)
    total = sum(counts.values())
    weights = torch.zeros(config.NUM_CLASSES, dtype=torch.float32)
    for cls_idx in range(config.NUM_CLASSES):
        count = counts.get(cls_idx, 1)          # avoid division by zero
        weights[cls_idx] = total / count

    # Normalise so the weights sum to NUM_CLASSES (keeps loss magnitude stable)
    weights = weights / weights.sum() * config.NUM_CLASSES

    return weights.to(config.DEVICE)


# ──────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────

class Trainer:
    """End-to-end training loop with logging, checkpointing, and AMP."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device=config.DEVICE,
        model_name: str = "model",
        run_metadata: dict | None = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name
        self.run_metadata = dict(run_metadata or {})

        # ── optimiser ──
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # ── loss (with class weights + label smoothing) ──
        class_weights = (
            compute_class_weights(train_loader.dataset)
            if config.USE_CLASS_WEIGHTED_LOSS
            else None
        )
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.LABEL_SMOOTHING,
        )

        # ── scheduler ──
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.EPOCHS,
            eta_min=config.LR_MIN,
        )

        # ── AMP ──
        self.use_amp = bool(config.USE_AMP and self.device.type == "cuda")
        # Determine autocast device type for both CUDA and MPS
        if self.device.type == "cuda":
            self._autocast_dtype = "cuda"
        elif self.device.type == "mps":
            self._autocast_dtype = "cpu"  # MPS autocast falls back to CPU
        else:
            self._autocast_dtype = "cpu"
        self.scaler = GradScaler(enabled=self.use_amp and self.device.type == "cuda")

        # ── logging ──
        log_subdir = config.LOG_DIR / f"{model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=str(log_subdir))

        # ── early stopping ──
        self.early_stopping = EarlyStopping()

        # ── tracking ──
        self.best_acc: float = 0.0
        self.best_loss: float = float("inf")
        self.epochs_completed: int = 0
        self.stopped_early: bool = False

    # ────────────────────── train one epoch ──────────────────────

    def train_one_epoch(self, epoch: int) -> tuple[float, float]:
        """Train for one epoch. Returns (avg_loss, accuracy)."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"  Train  [{epoch + 1}/{config.EPOCHS}]",
            leave=False,
            ncols=120,
        )

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self._autocast_dtype, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()

            # Gradient clipping (unscale first for correct norm)
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=config.GRAD_CLIP_MAX_NORM,
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Metrics
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            _, preds = outputs.max(dim=1)
            correct += preds.eq(labels).sum().item()
            total += batch_size

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct / total:.4f}",
            )

        avg_loss = running_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    # ────────────────────── validate ──────────────────────

    @torch.no_grad()
    def validate(self, epoch: int) -> tuple[float, float]:
        """Evaluate on the validation set. Returns (avg_loss, accuracy)."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"  Valid  [{epoch + 1}/{config.EPOCHS}]",
            leave=False,
            ncols=120,
        )

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(device_type=self._autocast_dtype, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            _, preds = outputs.max(dim=1)
            correct += preds.eq(labels).sum().item()
            total += batch_size

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct / total:.4f}",
            )

        avg_loss = running_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    # ────────────────────── full training loop ──────────────────────

    def fit(self) -> dict:
        """
        Run the full training loop.

        Returns
        -------
        history : dict
            Keys: ``train_loss``, ``val_loss``, ``train_acc``,
            ``val_acc``, ``lr``  (each a list of per-epoch values).
        """
        history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }

        print(f"\n{'=' * 60}")
        print(f"  Training  {self.model_name.upper()}")
        print(f"  Device: {self.device}  |  Epochs: {config.EPOCHS}")
        print(f"{'=' * 60}\n")

        for epoch in range(config.EPOCHS):
            epoch_start = time.time()

            # ── train & validate ──
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            # ── record history ──
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["lr"].append(current_lr)

            epoch_time = time.time() - epoch_start
            self.epochs_completed = epoch + 1

            # ── console summary ──
            print(
                f"  Epoch {epoch + 1:03d}/{config.EPOCHS}  "
                f"│  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"│  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                f"│  lr={current_lr:.2e}  "
                f"│  {epoch_time:.1f}s"
            )

            # ── TensorBoard ──
            self.writer.add_scalars(
                "Loss", {"train": train_loss, "val": val_loss}, epoch
            )
            self.writer.add_scalars(
                "Accuracy", {"train": train_acc, "val": val_acc}, epoch
            )
            self.writer.add_scalar("Learning Rate", current_lr, epoch)

            # ── checkpoints ──
            save_dir = config.MODEL_SAVE_DIR / self.model_name
            checkpoint_metadata = {**self.run_metadata, "epoch": epoch + 1}

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                EarlyStopping.save_checkpoint(
                    self.model, save_dir / config.CHECKPOINT_BEST_ACC, checkpoint_metadata
                )

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                EarlyStopping.save_checkpoint(
                    self.model, save_dir / config.CHECKPOINT_BEST_LOSS, checkpoint_metadata
                )

            EarlyStopping.save_checkpoint(
                self.model, save_dir / config.CHECKPOINT_LAST, checkpoint_metadata
            )

            # ── early stopping ──
            if self.early_stopping(val_loss):
                self.stopped_early = True
                print(f"\n  ⏹  Early stopping at epoch {epoch + 1}")
                break

        self.writer.close()

        print(f"\n{'─' * 60}")
        print(f"  Training complete.")
        print(f"  Best val accuracy : {self.best_acc:.4f}")
        print(f"  Best val loss     : {self.best_loss:.4f}")
        print(f"{'─' * 60}\n")

        return history


# ──────────────────────────────────────────────────────────────
# CLI Entry-point
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Brain Tumor Classification model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.DEFAULT_MODEL,
        choices=["efficientnet", "custom_cnn"],
        help="Model architecture to train (default: %(default)s).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Override max epochs (default: {config.EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Override batch size (default: {config.BATCH_SIZE}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=f"Override learning rate (default: {config.LEARNING_RATE}).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    args = parser.parse_args()

    protocol_conformant = args.epochs is None and args.batch_size is None and args.lr is None

    # ── apply CLI overrides ──
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        config.LEARNING_RATE = args.lr

    # ── reproducibility ──
    seed_everything(config.RANDOM_SEED)

    # ── print configuration ──
    config.print_config(model_name=args.model)

    # ── data ──
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dir=config.TRAIN_DIR,
        val_dir=config.VAL_DIR,
        test_dir=config.TEST_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")

    def file_sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""

    run_id = f"{args.model}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_metadata = {
        "run_id": run_id,
        "model_name": args.model,
        "dataset_manifest_sha256": file_sha256(config.MANIFEST_DIR / "dataset_manifest.csv"),
        "experiment_config_sha256": file_sha256(config.PROJECT_ROOT / "configs" / "experiment.yaml"),
        "requested_max_epochs": config.EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "weight_decay": config.WEIGHT_DECAY,
        "protocol_conformant": protocol_conformant,
    }

    # ── model ──
    model = get_model(
        model_name=args.model,
        num_classes=config.NUM_CLASSES,
        pretrained=config.EFFICIENTNET_PRETRAINED,
    )
    if args.resume:
        EarlyStopping.load_checkpoint(model, args.resume)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters    : {total_params:,} total, {trainable_params:,} trainable\n")

    # ── train ──
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.DEVICE,
        model_name=args.model,
        run_metadata=run_metadata,
    )
    history = trainer.fit()

    # ── calibration on validation, then final evaluation on test ──
    from src.evaluate import Evaluator
    from src.calibration import fit_temperature

    print("  Running final evaluation on test set …")
    best_ckpt = config.MODEL_SAVE_DIR / args.model / config.CHECKPOINT_BEST_ACC
    EarlyStopping.load_checkpoint(model, best_ckpt)
    model.to(config.DEVICE)

    temperature = fit_temperature(model, val_loader, device=config.DEVICE)
    calibration_path = config.MODEL_SAVE_DIR / args.model / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "temperature": temperature,
                "fit_split": "validation",
                "checkpoint": str(best_ckpt),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                **run_metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    group_ids = group_ids_for_materialized_samples(
        test_loader.dataset.samples,
        config.MANIFEST_DIR / "dataset_manifest.csv",
        config.PROCESSED_DATA_DIR,
    )
    metrics_dir = config.METRICS_DIR / args.model
    evaluator = Evaluator(
        model=model,
        data_loader=test_loader,
        device=config.DEVICE,
        temperature=temperature,
        group_ids=group_ids,
        output_dir=metrics_dir,
        evaluation_split="internal_locked_test",
    )
    results = evaluator.full_evaluation(history=history)
    (metrics_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    run_summary = {
        **run_metadata,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "epochs_completed": trainer.epochs_completed,
        "stopped_early": trainer.stopped_early,
        "training_complete": True,
        "evaluation_complete": True,
        "calibration_complete": True,
        "best_validation_accuracy": trainer.best_acc,
        "best_validation_loss": trainer.best_loss,
        "internal_test_macro_f1": results["f1_macro"],
        "metrics_path": str(metrics_dir / "research_metrics.json"),
    }
    (config.MODEL_SAVE_DIR / args.model / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
