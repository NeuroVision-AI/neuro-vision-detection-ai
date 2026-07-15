"""
Evaluation pipeline for Brain Tumor Classification.

Includes:
    - Evaluator class with per-class and macro metrics
    - Publication-quality confusion matrix, ROC curves, and training
      history plots
    - Classification report (text + JSON)
    - CLI entry-point
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for servers)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.amp import autocast
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from src import config


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Top-label expected calibration error using equally spaced bins."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == y_true
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        lower, upper = boundaries[idx], boundaries[idx + 1]
        mask = (confidence > lower) & (confidence <= upper)
        if idx == 0:
            mask |= confidence == 0.0
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean squared error between multiclass probabilities and one-hot labels."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0) or not np.all(np.isfinite(row_sums)):
        raise ValueError("Probability rows must have a positive finite sum")
    probabilities = probabilities / row_sums
    targets = np.eye(probabilities.shape[1], dtype=float)[y_true]
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric,
    groups: np.ndarray | None = None,
    samples: int = 1000,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap CI, resampling patients/groups when provided."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        raise ValueError("Cannot bootstrap an empty evaluation set")
    grouped_bootstrap = groups is not None
    groups = np.asarray(groups) if grouped_bootstrap else np.arange(len(y_true))
    if len(groups) != len(y_true):
        raise ValueError("groups must have one entry per evaluated sample")
    unique_groups = np.unique(groups)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(max(1, samples)):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate([group_indices[group] for group in sampled_groups])
        estimates.append(float(metric(y_true[sampled_indices], y_pred[sampled_indices])))
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return {
        "estimate": float(metric(y_true, y_pred)),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "bootstrap_samples": int(max(1, samples)),
        "bootstrap_unit": "patient_or_duplicate_group" if grouped_bootstrap else "image",
    }


def risk_coverage_curve(y_true: np.ndarray, probabilities: np.ndarray, points: int = 20) -> list:
    """Return error risk when progressively retaining only confident cases."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    order = np.argsort(-confidence)
    result = []
    for coverage in np.linspace(0.05, 1.0, max(2, points)):
        count = max(1, int(np.ceil(len(y_true) * coverage)))
        selected = order[:count]
        result.append(
            {
                "coverage": float(count / len(y_true)),
                "risk": float(1.0 - accuracy_score(y_true[selected], prediction[selected])),
                "minimum_confidence": float(confidence[selected].min()),
            }
        )
    return result


def compute_research_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    groups: np.ndarray | None = None,
    bootstrap_samples: int = 1000,
) -> dict:
    """Compute the prespecified discrimination, calibration, and uncertainty metrics."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    y_pred = probabilities.argmax(axis=1)
    num_classes = len(class_names)
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    y_bin = label_binarize(y_true, classes=labels)

    precision_per_class = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    try:
        roc_auc_macro = float(roc_auc_score(y_bin, probabilities, average="macro", multi_class="ovr"))
    except ValueError:
        roc_auc_macro = None
    try:
        pr_auc_macro = float(average_precision_score(y_bin, probabilities, average="macro"))
    except ValueError:
        pr_auc_macro = None

    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc_macro_ovr": roc_auc_macro,
        "pr_auc_macro_ovr": pr_auc_macro,
        "negative_log_likelihood": float(log_loss(y_true, probabilities, labels=labels)),
        "expected_calibration_error": expected_calibration_error(
            y_true, probabilities, n_bins=config.CALIBRATION_BINS
        ),
        "multiclass_brier_score": multiclass_brier_score(y_true, probabilities),
        "confidence_intervals": {},
        "risk_coverage": risk_coverage_curve(
            y_true, probabilities, points=config.ABSTENTION_COVERAGE_POINTS
        ),
        "per_class": {},
    }
    for idx, name in enumerate(class_names):
        tp = int(cm[idx, idx])
        fn = int(cm[idx, :].sum() - tp)
        fp = int(cm[:, idx].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        specificity = tn / (tn + fp) if tn + fp else 0.0
        results["per_class"][name] = {
            "precision": float(precision_per_class[idx]),
            "recall": float(recall_per_class[idx]),
            "sensitivity_recall": float(recall_per_class[idx]),
            "specificity": float(specificity),
            "f1": float(f1_per_class[idx]),
            "support": int((y_true == idx).sum()),
        }

    group_array = np.asarray(groups) if groups is not None else None
    results["confidence_intervals"]["accuracy"] = bootstrap_confidence_interval(
        y_true,
        y_pred,
        accuracy_score,
        groups=group_array,
        samples=bootstrap_samples,
        seed=config.RANDOM_SEED,
    )
    results["confidence_intervals"]["f1_macro"] = bootstrap_confidence_interval(
        y_true,
        y_pred,
        lambda truth, pred: f1_score(truth, pred, average="macro", zero_division=0),
        groups=group_array,
        samples=bootstrap_samples,
        seed=config.RANDOM_SEED + 1,
    )
    return results


# ──────────────────────────────────────────────────────────────
# Matplotlib publication-quality defaults
# ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ──────────────────────────────────────────────────────────────
# Evaluator
# ──────────────────────────────────────────────────────────────

class Evaluator:
    """
    Comprehensive model evaluator for multi-class classification.

    Parameters
    ----------
    model : nn.Module
        Trained classification model.
    data_loader : DataLoader
        DataLoader for the evaluation split (typically *test*).
    device : torch.device
        Compute device.
    class_names : list[str]
        Human-readable class labels.
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader,
        device=config.DEVICE,
        class_names: list[str] | None = None,
        temperature: float = 1.0,
        group_ids: list[str] | np.ndarray | None = None,
        output_dir: Path | str | None = None,
        evaluation_split: str = "test",
    ):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.class_names = class_names or config.CLASS_NAMES
        self.num_classes = len(self.class_names)
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.group_ids = np.asarray(group_ids) if group_ids is not None else None
        self.output_dir = Path(output_dir or config.METRICS_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_split = evaluation_split

        # Determine autocast device type for both CUDA and MPS
        self._autocast_dtype = "cuda" if self.device.type == "cuda" else "cpu"
        self._use_amp = bool(config.USE_AMP and self.device.type == "cuda")

        # Will be populated by evaluate()
        self._all_labels: np.ndarray | None = None
        self._all_preds: np.ndarray | None = None
        self._all_probs: np.ndarray | None = None
        self._all_uncalibrated_probs: np.ndarray | None = None
        self._all_group_ids: np.ndarray | None = None

    # ────────────────────── core evaluation ──────────────────────

    @torch.no_grad()
    def evaluate(self) -> dict:
        """
        Run the model on the full data loader and compute metrics.

        Returns
        -------
        results : dict
            Contains accuracy, per-class precision/recall/F1,
            macro-averaged precision/recall/F1.
        """
        self.model.eval()

        all_labels: list[int] = []
        all_preds: list[int] = []
        all_probs: list[np.ndarray] = []
        all_uncalibrated_probs: list[np.ndarray] = []
        observed_groups: list[str] = []

        pbar = tqdm(self.data_loader, desc="  Evaluating", leave=False, ncols=100)

        for batch in pbar:
            images, labels = batch[0], batch[1]
            if len(batch) >= 3:
                metadata = batch[2]
                if isinstance(metadata, dict) and "group_id" in metadata:
                    observed_groups.extend([str(value) for value in metadata["group_id"]])
                elif isinstance(metadata, (list, tuple)):
                    observed_groups.extend([str(value) for value in metadata])
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(device_type=self._autocast_dtype, enabled=self._use_amp):
                outputs = self.model(images)

            calibrated_outputs = outputs / self.temperature
            uncalibrated_probs = torch.softmax(outputs, dim=1).cpu().numpy()
            probs = torch.softmax(calibrated_outputs, dim=1).cpu().numpy()
            preds = calibrated_outputs.argmax(dim=1).cpu().numpy()

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.append(probs)
            all_uncalibrated_probs.append(uncalibrated_probs)

        self._all_labels = np.array(all_labels)
        self._all_preds = np.array(all_preds)
        self._all_probs = np.concatenate(all_probs, axis=0)
        self._all_uncalibrated_probs = np.concatenate(all_uncalibrated_probs, axis=0)
        if self.group_ids is not None:
            if len(self.group_ids) != len(self._all_labels):
                raise ValueError("group_ids must have one entry per evaluated sample")
            self._all_group_ids = self.group_ids
        elif observed_groups:
            if len(observed_groups) != len(self._all_labels):
                raise ValueError("Observed group metadata is incomplete")
            self._all_group_ids = np.asarray(observed_groups)

        results = compute_research_metrics(
            self._all_labels,
            self._all_probs,
            self.class_names,
            groups=self._all_group_ids,
            bootstrap_samples=config.BOOTSTRAP_SAMPLES,
        )
        results["n_images"] = int(len(self._all_labels))
        results["n_bootstrap_groups"] = int(
            len(np.unique(self._all_group_ids))
            if self._all_group_ids is not None
            else len(self._all_labels)
        )
        results["bootstrap_group_source"] = (
            "manifest_patient_or_duplicate_component"
            if self._all_group_ids is not None
            else "image"
        )
        labels = list(range(self.num_classes))
        uncalibrated_probabilities = self._all_uncalibrated_probs.astype(float)
        uncalibrated_probabilities /= uncalibrated_probabilities.sum(axis=1, keepdims=True)
        results["calibration_comparison"] = {
            "temperature": self.temperature,
            "fit_split": "validation",
            "uncalibrated": {
                "negative_log_likelihood": float(
                    log_loss(self._all_labels, uncalibrated_probabilities, labels=labels)
                ),
                "expected_calibration_error": expected_calibration_error(
                    self._all_labels,
                    uncalibrated_probabilities,
                    n_bins=config.CALIBRATION_BINS,
                ),
                "multiclass_brier_score": multiclass_brier_score(
                    self._all_labels, uncalibrated_probabilities
                ),
            },
            "calibrated": {
                "negative_log_likelihood": results["negative_log_likelihood"],
                "expected_calibration_error": results["expected_calibration_error"],
                "multiclass_brier_score": results["multiclass_brier_score"],
            },
        }
        return results

    # ────────────────────── confusion matrix ──────────────────────

    def plot_confusion_matrix(
        self,
        save_path: Path | str | None = None,
    ) -> None:
        """
        Plot normalised and unnormalised confusion matrices side by side.
        """
        if self._all_labels is None:
            self.evaluate()

        save_path = Path(save_path or self.output_dir / "confusion_matrix.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        cm = confusion_matrix(self._all_labels, self._all_preds)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        fig.suptitle("Confusion Matrix — Brain Tumor Classification", fontweight="bold")

        # ── unnormalised ──
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            linewidths=0.5,
            ax=axes[0],
        )
        axes[0].set_title("Counts")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")

        # ── normalised ──
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            linewidths=0.5,
            vmin=0,
            vmax=1,
            ax=axes[1],
        )
        axes[1].set_title("Normalised")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("True")

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(save_path)
        plt.close(fig)
        print(f"  ✓ Confusion matrix saved → {save_path.name}")

    # ────────────────────── ROC curves ──────────────────────

    def plot_roc_curves(
        self,
        save_path: Path | str | None = None,
    ) -> None:
        """
        Plot one-vs-rest ROC curve for each class plus micro/macro AUC.
        """
        if self._all_labels is None:
            self.evaluate()

        save_path = Path(save_path or self.output_dir / "roc_curves.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Binarise labels
        from sklearn.preprocessing import label_binarize

        y_bin = label_binarize(self._all_labels, classes=list(range(self.num_classes)))

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set_title("ROC Curves — One vs Rest", fontweight="bold")

        class_colors = list(config.CLASS_COLORS.values())

        # Per-class ROC
        fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
        for i in range(self.num_classes):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(y_bin[:, i], self._all_probs[:, i])
            roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

            color = class_colors[i] if i < len(class_colors) else None
            ax.plot(
                fpr_dict[i],
                tpr_dict[i],
                color=color,
                lw=2,
                label=f"{self.class_names[i]}  (AUC = {roc_auc_dict[i]:.3f})",
            )

        # Micro-average
        fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), self._all_probs.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)
        ax.plot(
            fpr_micro,
            tpr_micro,
            color="navy",
            lw=2,
            linestyle="--",
            label=f"Micro-avg  (AUC = {auc_micro:.3f})",
        )

        # Macro-average
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(self.num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.num_classes):
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        mean_tpr /= self.num_classes
        auc_macro = auc(all_fpr, mean_tpr)
        ax.plot(
            all_fpr,
            mean_tpr,
            color="darkorange",
            lw=2,
            linestyle=":",
            label=f"Macro-avg  (AUC = {auc_macro:.3f})",
        )

        # Diagonal reference
        ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", alpha=0.6)

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", frameon=True, framealpha=0.9)

        fig.savefig(save_path)
        plt.close(fig)
        print(f"  ✓ ROC curves saved → {save_path.name}")

    # ────────────────────── training history ──────────────────────

    def plot_training_history(
        self,
        history: dict,
        save_path: Path | str | None = None,
    ) -> None:
        """
        Plot 2×2 grid: Loss, Accuracy, Learning Rate, and per-class F1
        (if available, else macro F1 from the run).
        """
        save_path = Path(save_path or self.output_dir / "training_history.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        epochs = range(1, len(history["train_loss"]) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle("Training History", fontweight="bold", fontsize=14)

        # ── Loss ──
        ax = axes[0, 0]
        ax.plot(epochs, history["train_loss"], label="Train", linewidth=2)
        ax.plot(epochs, history["val_loss"], label="Val", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss")
        ax.legend()

        # ── Accuracy ──
        ax = axes[0, 1]
        ax.plot(epochs, history["train_acc"], label="Train", linewidth=2)
        ax.plot(epochs, history["val_acc"], label="Val", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy")
        ax.legend()

        # ── Learning Rate ──
        ax = axes[1, 0]
        ax.plot(epochs, history["lr"], color="tab:green", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log")

        # ── F1 / Summary bar chart ──
        ax = axes[1, 1]
        if self._all_labels is not None:
            # Per-class F1 bar chart
            f1_per = f1_score(
                self._all_labels, self._all_preds, average=None, zero_division=0
            )
            bars = ax.bar(
                self.class_names,
                f1_per,
                color=[config.CLASS_COLORS.get(c, "#888888") for c in self.class_names],
                edgecolor="white",
                linewidth=0.5,
            )
            for bar, val in zip(bars, f1_per):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    fontsize=10,
                )
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("F1 Score")
            ax.set_title("Per-Class F1 (Test)")
        else:
            ax.text(
                0.5, 0.5, "F1 data\nnot available",
                ha="center", va="center",
                fontsize=12, color="grey",
                transform=ax.transAxes,
            )
            ax.set_title("Per-Class F1")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(save_path)
        plt.close(fig)
        print(f"  ✓ Training history saved → {save_path.name}")

    def plot_calibration_curve(self, save_path: Path | str | None = None) -> None:
        """Plot top-label reliability before and after temperature scaling."""
        if self._all_labels is None:
            self.evaluate()
        save_path = Path(save_path or self.output_dir / "calibration_curve.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        boundaries = np.linspace(0.0, 1.0, config.CALIBRATION_BINS + 1)

        def reliability(probabilities):
            confidence = probabilities.max(axis=1)
            prediction = probabilities.argmax(axis=1)
            correct = prediction == self._all_labels
            mean_confidence, mean_accuracy = [], []
            for idx in range(config.CALIBRATION_BINS):
                mask = (confidence > boundaries[idx]) & (confidence <= boundaries[idx + 1])
                if idx == 0:
                    mask |= confidence == 0.0
                if np.any(mask):
                    mean_confidence.append(float(confidence[mask].mean()))
                    mean_accuracy.append(float(correct[mask].mean()))
            return confidence, mean_confidence, mean_accuracy

        calibrated_confidence, calibrated_mean_confidence, calibrated_mean_accuracy = reliability(
            self._all_probs
        )
        _, uncalibrated_mean_confidence, uncalibrated_mean_accuracy = reliability(
            self._all_uncalibrated_probs
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
        axes[0].plot(
            uncalibrated_mean_confidence,
            uncalibrated_mean_accuracy,
            marker="o",
            color="#C0504D",
            label="Uncalibrated",
        )
        axes[0].plot(
            calibrated_mean_confidence,
            calibrated_mean_accuracy,
            marker="o",
            color="#2F75B5",
            label="Temperature-scaled",
        )
        axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Observed accuracy", title="Reliability diagram")
        axes[0].legend()
        axes[1].hist(calibrated_confidence, bins=boundaries, color="#1F8A70", edgecolor="white")
        axes[1].set(xlim=(0, 1), xlabel="Top-class confidence", ylabel="Images", title="Confidence distribution")
        fig.suptitle(f"Calibration (temperature={self.temperature:.3f})", fontweight="bold")
        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        print(f"  ✓ Calibration curve saved → {save_path.name}")

    def plot_risk_coverage(self, save_path: Path | str | None = None) -> None:
        """Plot retained-case error as low-confidence cases are abstained."""
        if self._all_labels is None:
            self.evaluate()
        save_path = Path(save_path or self.output_dir / "risk_coverage.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        points = risk_coverage_curve(
            self._all_labels,
            self._all_probs,
            points=config.ABSTENTION_COVERAGE_POINTS,
        )
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.plot(
            [point["coverage"] for point in points],
            [point["risk"] for point in points],
            marker="o",
            color="#C00000",
        )
        ax.set(xlim=(0, 1.02), ylim=(0, 1), xlabel="Coverage (fraction retained)", ylabel="Error risk", title="Selective prediction risk–coverage")
        fig.savefig(save_path)
        plt.close(fig)
        print(f"  ✓ Risk–coverage curve saved → {save_path.name}")

    # ────────────────────── classification report ──────────────────────

    def generate_classification_report(
        self,
        save_path: Path | str | None = None,
    ) -> str:
        """
        Generate sklearn classification report as text + JSON.

        Returns the text report.
        """
        if self._all_labels is None:
            self.evaluate()

        save_path = Path(save_path or self.output_dir / "classification_report")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Text report
        text_report = classification_report(
            self._all_labels,
            self._all_preds,
            target_names=self.class_names,
            digits=4,
            zero_division=0,
        )
        text_path = save_path.with_suffix(".txt")
        text_path.write_text(text_report, encoding="utf-8")
        print(f"  ✓ Classification report (text) → {text_path.name}")

        # JSON report
        json_report = classification_report(
            self._all_labels,
            self._all_preds,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )
        json_path = save_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(json_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  ✓ Classification report (JSON)  → {json_path.name}")

        return text_report

    def save_prediction_table(self, save_path: Path | str | None = None) -> Path:
        """Save auditable per-image predictions for paired and error analyses."""
        if self._all_labels is None:
            self.evaluate()
        save_path = Path(save_path or self.output_dir / "predictions.csv")
        samples = getattr(self.data_loader.dataset, "samples", None)
        if samples is not None and len(samples) != len(self._all_labels):
            raise ValueError("Dataset samples do not align with evaluated predictions")
        headers = [
            "sample_index",
            "sample_path",
            "group_id",
            "true_index",
            "true_label",
            "predicted_index",
            "predicted_label",
            "correct",
            "calibrated_confidence",
            "uncalibrated_confidence",
            *[f"probability_{name}" for name in self.class_names],
        ]
        with save_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for index, (true_index, predicted_index) in enumerate(
                zip(self._all_labels, self._all_preds)
            ):
                calibrated = self._all_probs[index]
                uncalibrated = self._all_uncalibrated_probs[index]
                row = {
                    "sample_index": index,
                    "sample_path": str(samples[index][0]) if samples is not None else "",
                    "group_id": (
                        str(self._all_group_ids[index])
                        if self._all_group_ids is not None
                        else f"image_{index}"
                    ),
                    "true_index": int(true_index),
                    "true_label": self.class_names[int(true_index)],
                    "predicted_index": int(predicted_index),
                    "predicted_label": self.class_names[int(predicted_index)],
                    "correct": int(true_index == predicted_index),
                    "calibrated_confidence": float(calibrated.max()),
                    "uncalibrated_confidence": float(uncalibrated.max()),
                }
                row.update(
                    {
                        f"probability_{name}": float(calibrated[class_index])
                        for class_index, name in enumerate(self.class_names)
                    }
                )
                writer.writerow(row)
        print(f"  ✓ Per-image predictions saved → {save_path.name}")
        return save_path

    # ────────────────────── full evaluation ──────────────────────

    def full_evaluation(self, history: dict | None = None) -> dict:
        """
        Run every evaluation step and print a summary.

        Parameters
        ----------
        history : dict, optional
            Training history dict (from ``Trainer.fit``).

        Returns
        -------
        results : dict
            Aggregate metrics from ``evaluate()``.
        """
        print(f"\n{'=' * 60}")
        print("  Full Evaluation — Brain Tumor Classification")
        print(f"{'=' * 60}\n")

        results = self.evaluate()

        # ── print summary ──
        print(f"\n  {'Metric':<22} {'Value':>10}")
        print(f"  {'─' * 34}")
        print(f"  {'Accuracy':<22} {results['accuracy']:>10.4f}")
        print(f"  {'Precision (macro)':<22} {results['precision_macro']:>10.4f}")
        print(f"  {'Recall (macro)':<22} {results['recall_macro']:>10.4f}")
        print(f"  {'F1 (macro)':<22} {results['f1_macro']:>10.4f}")
        print(f"  {'MCC':<22} {results['mcc']:>10.4f}")
        print(f"  {'ECE':<22} {results['expected_calibration_error']:>10.4f}")
        print(f"  {'Brier score':<22} {results['multiclass_brier_score']:>10.4f}")
        print()
        print(f"  {'Class':<16} {'Prec':>8} {'Rec':>8} {'F1':>8}")
        print(f"  {'─' * 42}")
        for name, m in results["per_class"].items():
            print(
                f"  {name:<16} {m['precision']:>8.4f} "
                f"{m['recall']:>8.4f} {m['f1']:>8.4f}"
            )
        print()

        # ── generate all plots & reports ──
        self.plot_confusion_matrix()
        self.plot_roc_curves()
        self.plot_calibration_curve()
        self.plot_risk_coverage()
        self.generate_classification_report()
        self.save_prediction_table()

        metrics_path = self.output_dir / "research_metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    **results,
                    "temperature": self.temperature,
                    "evaluation_split": self.evaluation_split,
                    "claim_scope": "research-only 2D public-data proof-of-concept",
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"  ✓ Research metrics saved → {metrics_path.name}")

        if history is not None:
            self.plot_training_history(history)

        print(f"\n  All outputs saved to: {self.output_dir}\n")
        return results


# ──────────────────────────────────────────────────────────────
# CLI Entry-point
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Brain Tumor Classification model.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved model checkpoint (.pth).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=config.DEFAULT_MODEL,
        choices=["efficientnet", "custom_cnn"],
        help="Model architecture (default: %(default)s).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Validation-fitted temperature to apply (default: %(default)s).",
    )
    args = parser.parse_args()

    # ── load model ──
    from src.models.model_factory import get_model
    model = get_model(
        model_name=args.model_name,
        num_classes=config.NUM_CLASSES,
        pretrained=False,  # weights come from the checkpoint
    )
    ckpt_path = Path(args.model_path)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict)
    model.to(config.DEVICE)
    print(f"  ✓ Loaded checkpoint: {ckpt_path.name}")

    # ── data ──
    from src.dataset import get_dataloaders
    from src.data_integrity import group_ids_for_materialized_samples

    _, _, test_loader = get_dataloaders(
        train_dir=config.TRAIN_DIR,
        val_dir=config.VAL_DIR,
        test_dir=config.TEST_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    # ── evaluate ──
    group_ids = group_ids_for_materialized_samples(
        test_loader.dataset.samples,
        config.MANIFEST_DIR / "dataset_manifest.csv",
        config.PROCESSED_DATA_DIR,
    )
    evaluator = Evaluator(
        model=model,
        data_loader=test_loader,
        device=config.DEVICE,
        temperature=args.temperature,
        group_ids=group_ids,
        output_dir=config.METRICS_DIR / args.model_name,
        evaluation_split="internal_locked_test",
    )
    evaluator.full_evaluation()


if __name__ == "__main__":
    main()
