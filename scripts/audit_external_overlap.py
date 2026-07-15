#!/usr/bin/env python3
"""Audit exact cross-corpus reuse before calling a dataset external validation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def normalize_path(value: str) -> str:
    return value.replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-corpus exact-overlap audit")
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.development_manifest.open("r", encoding="utf-8", newline="") as handle:
        development = list(csv.DictReader(handle))
    with args.candidate_manifest.open("r", encoding="utf-8", newline="") as handle:
        candidate_all = list(csv.DictReader(handle))

    development_by_hash = defaultdict(list)
    for record in development:
        if record.get("exact_sha256"):
            development_by_hash[record["exact_sha256"]].append(record)

    candidates = [record for record in candidate_all if record.get("task") == "classification"]
    masks_by_stem = {
        Path(record["filename"]).stem: normalize_path(record["relative_path"])
        for record in candidate_all
        if record.get("task") == "segmentation" and record.get("is_mask", "").lower() == "true"
    }
    overlaps = []
    xai_rows = []
    for candidate in candidates:
        matches = development_by_hash.get(candidate.get("sha256", ""), [])
        if not matches:
            continue
        for development_record in matches:
            row = {
                "sha256": candidate["sha256"],
                "candidate_path": normalize_path(candidate["relative_path"]),
                "candidate_split": candidate.get("split", ""),
                "candidate_label": candidate.get("tumor_label", ""),
                "development_path": development_record.get("relative_path", ""),
                "development_materialized_path": development_record.get("materialized_path", ""),
                "development_split": development_record.get("assigned_split", ""),
                "development_label": development_record.get("class_name", ""),
                "label_agreement": candidate.get("tumor_label", "") == development_record.get("class_name", ""),
            }
            overlaps.append(row)
            mask_path = masks_by_stem.get(Path(candidate["filename"]).stem)
            if (
                mask_path
                and development_record.get("assigned_split") == "test"
                and development_record.get("class_name") != "no_tumor"
            ):
                xai_rows.append({**row, "mask_path": mask_path})

    unique_overlap_hashes = {row["sha256"] for row in overlaps}
    candidate_overlap_records = {row["candidate_path"] for row in overlaps}
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_records": len(development),
        "candidate_classification_records": len(candidates),
        "exact_overlap_candidate_records": len(candidate_overlap_records),
        "exact_overlap_unique_hashes": len(unique_overlap_hashes),
        "candidate_exact_overlap_fraction": (
            len(candidate_overlap_records) / len(candidates) if candidates else 0.0
        ),
        "label_disagreements": sum(not row["label_agreement"] for row in overlaps),
        "overlap_by_candidate_split": dict(Counter(row["candidate_split"] for row in overlaps)),
        "overlap_by_development_split": dict(Counter(row["development_split"] for row in overlaps)),
        "independent_external_validation_eligible": not unique_overlap_hashes,
        "decision": (
            "candidate rejected as independent external validation because exact image reuse was detected"
            if unique_overlap_hashes
            else "no exact reuse detected; near-duplicate and provenance review still required"
        ),
        "xai_mask_mapping_records": len(xai_rows),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "overlap_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    fields = list(overlaps[0].keys()) if overlaps else []
    with (args.output_dir / "exact_overlaps.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(overlaps)
    xai_fields = list(xai_rows[0].keys()) if xai_rows else []
    with (args.output_dir / "xai_mask_mapping.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=xai_fields)
        if xai_fields:
            writer.writeheader()
            writer.writerows(xai_rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
