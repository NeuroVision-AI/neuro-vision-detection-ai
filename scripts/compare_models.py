#!/usr/bin/env python3
"""Paired locked-test comparison from auditable per-image prediction tables."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parent.parent


def read_predictions(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(not row.get("sample_path") for row in rows):
        raise ValueError(f"Prediction table is empty or lacks sample paths: {path}")
    indexed = {row["sample_path"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Prediction table contains duplicate sample paths: {path}")
    return indexed


def paired_group_bootstrap(
    y_true: np.ndarray,
    primary_pred: np.ndarray,
    comparator_pred: np.ndarray,
    group_ids: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    groups = np.unique(group_ids)
    group_rows = {group: np.flatnonzero(group_ids == group) for group in groups}
    rng = np.random.default_rng(seed)
    accuracy_differences = np.empty(samples, dtype=float)
    f1_differences = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        rows = np.concatenate([group_rows[group] for group in sampled_groups])
        accuracy_differences[index] = accuracy_score(
            y_true[rows], primary_pred[rows]
        ) - accuracy_score(y_true[rows], comparator_pred[rows])
        f1_differences[index] = f1_score(
            y_true[rows], primary_pred[rows], average="macro", zero_division=0
        ) - f1_score(
            y_true[rows], comparator_pred[rows], average="macro", zero_division=0
        )

    def interval(values: np.ndarray) -> dict[str, float]:
        return {
            "lower_95": float(np.percentile(values, 2.5)),
            "upper_95": float(np.percentile(values, 97.5)),
        }

    return {
        "accuracy_difference": interval(accuracy_differences),
        "macro_f1_difference": interval(f1_differences),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary",
        type=Path,
        default=ROOT / "outputs" / "metrics" / "efficientnet" / "predictions.csv",
    )
    parser.add_argument(
        "--comparator",
        type=Path,
        default=ROOT / "outputs" / "metrics" / "custom_cnn" / "predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "model_comparison",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    primary = read_predictions(args.primary)
    comparator = read_predictions(args.comparator)
    if set(primary) != set(comparator):
        raise ValueError("Prediction tables do not contain the same locked-test samples")
    paths = sorted(primary)
    for path in paths:
        if primary[path]["true_index"] != comparator[path]["true_index"]:
            raise ValueError(f"True-label mismatch for {path}")
        if primary[path]["group_id"] != comparator[path]["group_id"]:
            raise ValueError(f"Bootstrap-group mismatch for {path}")

    y_true = np.array([int(primary[path]["true_index"]) for path in paths])
    primary_pred = np.array([int(primary[path]["predicted_index"]) for path in paths])
    comparator_pred = np.array([int(comparator[path]["predicted_index"]) for path in paths])
    groups = np.array([primary[path]["group_id"] for path in paths])

    primary_accuracy = float(accuracy_score(y_true, primary_pred))
    comparator_accuracy = float(accuracy_score(y_true, comparator_pred))
    primary_f1 = float(f1_score(y_true, primary_pred, average="macro", zero_division=0))
    comparator_f1 = float(
        f1_score(y_true, comparator_pred, average="macro", zero_division=0)
    )
    primary_only_correct = int(
        np.sum((primary_pred == y_true) & (comparator_pred != y_true))
    )
    comparator_only_correct = int(
        np.sum((primary_pred != y_true) & (comparator_pred == y_true))
    )
    discordant = primary_only_correct + comparator_only_correct
    mcnemar_p = (
        float(
            binomtest(
                min(primary_only_correct, comparator_only_correct),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    intervals = paired_group_bootstrap(
        y_true,
        primary_pred,
        comparator_pred,
        groups,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_split": "internal_locked_test",
        "primary_model": "efficientnet",
        "comparator_model": "custom_cnn",
        "n_images": len(paths),
        "n_duplicate_provenance_groups": int(len(np.unique(groups))),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "primary": {"accuracy": primary_accuracy, "macro_f1": primary_f1},
        "comparator": {"accuracy": comparator_accuracy, "macro_f1": comparator_f1},
        "paired_differences_primary_minus_comparator": {
            "accuracy": primary_accuracy - comparator_accuracy,
            "macro_f1": primary_f1 - comparator_f1,
            **intervals,
        },
        "mcnemar_exact": {
            "unit": "image",
            "interpretation": "exploratory because duplicate/provenance components can contain multiple images",
            "primary_only_correct": primary_only_correct,
            "comparator_only_correct": comparator_only_correct,
            "discordant_predictions": discordant,
            "two_sided_p_value": mcnemar_p,
        },
        "claim_scope": (
            "Paired internal source-test comparison; groups are duplicate/provenance "
            "components, not verified patients. This is not external validation."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "model_comparison.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = (
        "# Paired internal model comparison\n\n"
        f"- Images / duplicate-provenance groups: {len(paths):,} / {len(np.unique(groups)):,}\n"
        f"- EfficientNet minus custom-CNN accuracy: {primary_accuracy - comparator_accuracy:.3f} "
        f"(group-bootstrap 95% CI {intervals['accuracy_difference']['lower_95']:.3f} to "
        f"{intervals['accuracy_difference']['upper_95']:.3f})\n"
        f"- EfficientNet minus custom-CNN macro-F1: {primary_f1 - comparator_f1:.3f} "
        f"(group-bootstrap 95% CI {intervals['macro_f1_difference']['lower_95']:.3f} to "
        f"{intervals['macro_f1_difference']['upper_95']:.3f})\n"
        f"- Exploratory image-level exact McNemar p-value: {mcnemar_p:.4g} "
        f"({primary_only_correct} EfficientNet-only correct; "
        f"{comparator_only_correct} custom-CNN-only correct)\n\n"
        "These are paired internal source-test results. Bootstrap groups are observable "
        "duplicate/provenance components, not verified patients. The image-level McNemar "
        "result is exploratory because a component can contain multiple images.\n"
    )
    (args.output_dir / "MODEL_COMPARISON.md").write_text(markdown, encoding="utf-8")
    print(json_path)


if __name__ == "__main__":
    main()
