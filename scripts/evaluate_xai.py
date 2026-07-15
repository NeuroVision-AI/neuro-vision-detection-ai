#!/usr/bin/env python3
"""Score saved Grad-CAM arrays for repeatability, randomization, and localization."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.xai_evaluation import evaluate_explanation


def load_optional(directory: Path, name: str):
    path = directory / name
    return np.load(path) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Grad-CAM evidence arrays")
    parser.add_argument("root", type=Path, help="Contains original/, repeat/, randomized/, masks/")
    parser.add_argument("--quantile", type=float, default=0.8)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "xai_evaluation" / "xai_metrics.json",
    )
    args = parser.parse_args()

    original_dir = args.root / "original"
    if not original_dir.exists():
        raise FileNotFoundError(f"Missing {original_dir}")
    cases = []
    for original_path in sorted(original_dir.glob("*.npy")):
        name = original_path.name
        metrics = evaluate_explanation(
            np.load(original_path),
            repeat=load_optional(args.root / "repeat", name),
            randomized=load_optional(args.root / "randomized", name),
            mask=load_optional(args.root / "masks", name),
            quantile=args.quantile,
        )
        cases.append({"case": original_path.stem, **metrics})
    if not cases:
        raise ValueError("No original .npy heatmaps found")
    metric_names = sorted({key for case in cases for key in case if key != "case"})
    aggregate = {
        key: float(np.mean([case[key] for case in cases if key in case]))
        for key in metric_names
    }

    rng = np.random.default_rng(42)
    aggregate_ci = {}
    for key in metric_names:
        values = np.asarray([case[key] for case in cases if key in case], dtype=float)
        estimates = [
            float(np.mean(rng.choice(values, size=len(values), replace=True)))
            for _ in range(max(1, args.bootstrap_samples))
        ]
        lower, upper = np.percentile(estimates, [2.5, 97.5])
        aggregate_ci[key] = {
            "estimate": float(values.mean()),
            "lower_95": float(lower),
            "upper_95": float(upper),
            "bootstrap_samples": max(1, args.bootstrap_samples),
            "bootstrap_unit": "image",
        }

    metadata_path = args.root / "cases.csv"
    strata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            metadata = {row["case"]: row for row in csv.DictReader(handle)}
        for field in ("true_label", "correct"):
            field_groups = {}
            values = sorted({row[field] for row in metadata.values() if row.get(field)})
            for value in values:
                selected = [case for case in cases if metadata.get(case["case"], {}).get(field) == value]
                field_groups[value] = {
                    "cases": len(selected),
                    **{
                        key: float(np.mean([case[key] for case in selected if key in case]))
                        for key in metric_names
                        if any(key in case for case in selected)
                    },
                }
            strata[field] = field_groups
    result = {
        "cases": len(cases),
        "saliency_quantile": args.quantile,
        "aggregate": aggregate,
        "aggregate_ci": aggregate_ci,
        "stratified": strata,
        "per_case": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
