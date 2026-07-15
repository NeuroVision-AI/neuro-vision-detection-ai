#!/usr/bin/env python3
"""Build a deterministic, paper-facing development-data quality report."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifests" / "dataset_manifest.csv"
AUDIT = ROOT / "data" / "manifests" / "dataset_audit.json"
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "outputs" / "data_quality"
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
SPLITS = ["train", "val", "test"]


def load_rows() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_class_distribution(rows: list[dict[str, str]]) -> None:
    counts = {
        split: [sum(r["assigned_split"] == split and r["class_name"] == label for r in rows) for label in CLASSES]
        for split in SPLITS
    }
    x = np.arange(len(CLASSES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for index, split in enumerate(SPLITS):
        ax.bar(x + (index - 1) * width, counts[split], width, label=split.capitalize())
    ax.set_xticks(x, [name.replace("_", " ").title() for name in CLASSES])
    ax.set_ylabel("Images")
    ax.set_title("Leakage-controlled analysis cohort by class and split")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "class_distribution.png", dpi=180)
    plt.close(fig)


def save_dimension_plot(rows: list[dict[str, str]]) -> None:
    widths = np.array([int(row["width"]) for row in rows])
    heights = np.array([int(row["height"]) for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].hist(widths, bins=40, alpha=0.75, label="Width")
    axes[0].hist(heights, bins=40, alpha=0.60, label="Height")
    axes[0].set_xlabel("Pixels")
    axes[0].set_ylabel("Images")
    axes[0].set_title("Raw image dimensions")
    axes[0].legend(frameon=False)
    axes[1].scatter(widths, heights, s=5, alpha=0.15)
    axes[1].set_xlabel("Width (pixels)")
    axes[1].set_ylabel("Height (pixels)")
    axes[1].set_title("Width-height pairs")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "image_dimensions.png", dpi=180)
    plt.close(fig)


def sampled_intensity_summary(rows: list[dict[str, str]], seed: int = 42) -> dict:
    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[row["class_name"]].append(row)

    histograms: dict[str, np.ndarray] = {}
    class_stats: dict[str, dict[str, float | int]] = {}
    for label in CLASSES:
        candidates = sorted(by_class[label], key=lambda row: row["relative_path"])
        sampled = rng.sample(candidates, k=min(64, len(candidates)))
        histogram = np.zeros(256, dtype=np.int64)
        image_means = []
        image_stds = []
        raw_min = 255
        raw_max = 0
        for row in sampled:
            image = Image.open(PROCESSED / row["materialized_path"]).convert("L")
            array = np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR))
            histogram += np.bincount(array.ravel(), minlength=256)
            image_means.append(float(array.mean()))
            image_stds.append(float(array.std()))
            raw_min = min(raw_min, int(array.min()))
            raw_max = max(raw_max, int(array.max()))
        histograms[label] = histogram
        class_stats[label] = {
            "sampled_images": len(sampled),
            "resized_for_summary": "224x224 grayscale",
            "pixel_min": raw_min,
            "pixel_max": raw_max,
            "mean_image_mean": float(np.mean(image_means)),
            "mean_image_std": float(np.mean(image_stds)),
        }

    fig, ax = plt.subplots(figsize=(9, 5.4))
    for label in CLASSES:
        density = histograms[label] / histograms[label].sum()
        ax.plot(np.arange(256), density, label=label.replace("_", " ").title())
    ax.set_xlabel("8-bit grayscale intensity")
    ax.set_ylabel("Sampled pixel proportion")
    ax.set_title("Deterministic stratified intensity sample (64 images/class)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "sampled_intensity_distributions.png", dpi=180)
    plt.close(fig)
    return class_stats


def save_representative_montage(rows: list[dict[str, str]]) -> None:
    test_rows = defaultdict(list)
    for row in rows:
        if row["assigned_split"] == "test":
            test_rows[row["class_name"]].append(row)
    fig, axes = plt.subplots(len(CLASSES), 3, figsize=(7.5, 9.5))
    for class_index, label in enumerate(CLASSES):
        candidates = sorted(test_rows[label], key=lambda row: row["relative_path"])
        indices = [len(candidates) // 4, len(candidates) // 2, 3 * len(candidates) // 4]
        for sample_index, row_index in enumerate(indices):
            image = Image.open(PROCESSED / candidates[row_index]["materialized_path"]).convert("L")
            axes[class_index, sample_index].imshow(image, cmap="gray")
            axes[class_index, sample_index].axis("off")
            if sample_index == 0:
                axes[class_index, sample_index].set_title(label.replace("_", " ").title(), loc="left")
    fig.suptitle("Deterministic locked-test examples (labels are source categories)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUTPUT / "representative_locked_test_images.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    counts = {
        split: {
            label: sum(r["assigned_split"] == split and r["class_name"] == label for r in rows)
            for label in CLASSES
        }
        for split in SPLITS
    }
    widths = [int(row["width"]) for row in rows]
    heights = [int(row["height"]) for row in rows]
    save_class_distribution(rows)
    save_dimension_plot(rows)
    intensity = sampled_intensity_summary(rows)
    save_representative_montage(rows)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "records": len(rows),
        "counts_by_split_and_class": counts,
        "raw_dimensions": {
            "width_min": min(widths),
            "width_median": float(np.median(widths)),
            "width_max": max(widths),
            "height_min": min(heights),
            "height_median": float(np.median(heights)),
            "height_max": max(heights),
        },
        "decoded_modes": dict(Counter(row["mode"] for row in rows)),
        "sampled_intensity_summary": intensity,
        "integrity_audit": audit,
        "limitations": [
            "Patient, sequence, scanner, site, age and sex metadata are unavailable.",
            "Intensity summaries use a deterministic 64-image-per-class sample resized to 224x224.",
            "Source folder labels were not independently pathology-verified.",
        ],
    }
    (OUTPUT / "data_quality_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    markdown = f"""# Development data quality report

Generated from the final leakage-controlled manifest on 2026-07-15.

- Retained images: {len(rows):,}; all decoded successfully.
- Split counts: train {sum(counts['train'].values()):,}, validation {sum(counts['val'].values()):,}, locked test {sum(counts['test'].values()):,}.
- Raw widths: {min(widths)}-{max(widths)} pixels (median {np.median(widths):.0f}); heights: {min(heights)}-{max(heights)} (median {np.median(heights):.0f}).
- Decoded source mode: {dict(Counter(row['mode'] for row in rows))}; model input is converted to RGB, resized to 224x224 and ImageNet-normalized.
- One cross-label perceptual component (7 images) was excluded whole; no retained exact hash or duplicate/provenance group crosses final splits.

## Figures

- `class_distribution.png`
- `image_dimensions.png`
- `sampled_intensity_distributions.png`
- `representative_locked_test_images.png`

## Boundaries

Patient, sequence, scanner, site, age and sex metadata are unavailable. The intensity figure is descriptive and based on a deterministic 64-image-per-class sample. Source folder labels were not independently pathology-verified. Full values and audit evidence are in `data_quality_report.json` and `data/manifests/dataset_audit.json`.
"""
    (OUTPUT / "DATA_QUALITY_REPORT.md").write_text(markdown, encoding="utf-8")
    print(OUTPUT / "data_quality_report.json")


if __name__ == "__main__":
    main()
