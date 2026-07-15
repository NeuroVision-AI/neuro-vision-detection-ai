#!/usr/bin/env python3
"""Build and cross-audit a folder-labelled external MRI candidate.

The audit is deliberately independent of the candidate's own integrity claims.
It compares every candidate image with the complete development manifest by
raw SHA-256 and 64-bit difference-hash Hamming distance.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_integrity import audit_manifest, build_manifest, read_manifest, write_manifest


def hamming(left: int, right: int) -> int:
    # ``int.bit_count`` is unavailable in the project's legacy Python 3.9
    # environment; this corpus is small enough for the portable equivalent.
    return bin(left ^ right).count("1")


class BKTree:
    """Small metric tree for efficient 64-bit perceptual-hash lookup."""

    def __init__(self) -> None:
        self.root: tuple[int, dict] | None = None

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = (value, {})
            return
        node = self.root
        while True:
            current, children = node
            distance = hamming(value, current)
            if distance == 0:
                return
            if distance not in children:
                children[distance] = (value, {})
                return
            node = children[distance]

    def query(self, value: int, radius: int) -> list[tuple[int, int]]:
        if self.root is None:
            return []
        matches: list[tuple[int, int]] = []
        pending = [self.root]
        while pending:
            current, children = pending.pop()
            distance = hamming(value, current)
            if distance <= radius:
                matches.append((distance, current))
            lower, upper = distance - radius, distance + radius
            pending.extend(child for edge, child in children.items() if lower <= edge <= upper)
        return matches


def write_rows(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a folder-labelled external candidate")
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--near-hamming-radius", type=int, default=5)
    args = parser.parse_args()
    if not 0 <= args.near_hamming_radius <= 64:
        raise ValueError("near-hamming-radius must be between 0 and 64")

    development = read_manifest(args.development_manifest)
    candidate = build_manifest(args.candidate_root, source_name=args.source_name)
    if not candidate:
        raise ValueError("No candidate images were discovered")
    for record in candidate:
        record.assigned_split = record.source_split
        record.materialized_path = record.relative_path

    development_by_sha = defaultdict(list)
    development_by_dhash = defaultdict(list)
    tree = BKTree()
    for record in development:
        development_by_sha[record.exact_sha256].append(record)
        if record.perceptual_hash:
            value = int(record.perceptual_hash, 16)
            if value not in development_by_dhash:
                tree.add(value)
            development_by_dhash[value].append(record)

    exact_rows: list[dict] = []
    near_rows: list[dict] = []
    for record in candidate:
        exact_matches = development_by_sha.get(record.exact_sha256, [])
        for match in exact_matches:
            exact_rows.append(
                {
                    "candidate_path": record.relative_path,
                    "candidate_split": record.source_split,
                    "candidate_label": record.class_name,
                    "development_path": match.relative_path,
                    "development_split": match.assigned_split,
                    "development_label": match.class_name,
                    "sha256": record.exact_sha256,
                    "label_agreement": record.class_name == match.class_name,
                }
            )
        if record.perceptual_hash:
            matches = tree.query(int(record.perceptual_hash, 16), args.near_hamming_radius)
            if matches:
                minimum = min(distance for distance, _ in matches)
                for distance, value in matches:
                    if distance != minimum:
                        continue
                    for match in development_by_dhash[value]:
                        near_rows.append(
                            {
                                "candidate_path": record.relative_path,
                                "candidate_split": record.source_split,
                                "candidate_label": record.class_name,
                                "development_path": match.relative_path,
                                "development_split": match.assigned_split,
                                "development_label": match.class_name,
                                "dhash_hamming_distance": distance,
                                "label_agreement": record.class_name == match.class_name,
                            }
                        )

    candidate_audit = audit_manifest(candidate)
    exact_candidate_paths = {row["candidate_path"] for row in exact_rows}
    near_candidate_paths = {row["candidate_path"] for row in near_rows}
    split_by_candidate_path = {record.relative_path: record.source_split for record in candidate}
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_name": args.source_name,
        "candidate_root": str(args.candidate_root.resolve()),
        "development_manifest": str(args.development_manifest.resolve()),
        "candidate_records": len(candidate),
        "candidate_split_counts": dict(Counter(record.source_split for record in candidate)),
        "candidate_audit": candidate_audit,
        "exact_overlap_candidate_records": len(exact_candidate_paths),
        "exact_overlap_fraction": len(exact_candidate_paths) / len(candidate),
        "exact_overlap_by_candidate_split": dict(
            Counter(split_by_candidate_path[path] for path in exact_candidate_paths)
        ),
        "near_overlap_hamming_radius": args.near_hamming_radius,
        "near_overlap_candidate_records": len(near_candidate_paths),
        "near_overlap_fraction": len(near_candidate_paths) / len(candidate),
        "near_overlap_by_candidate_split": dict(
            Counter(split_by_candidate_path[path] for path in near_candidate_paths)
        ),
        "exact_label_disagreements": sum(not row["label_agreement"] for row in exact_rows),
        "near_label_disagreements": sum(not row["label_agreement"] for row in near_rows),
        "independent_external_validation_eligible": (
            not exact_candidate_paths
            and not near_candidate_paths
            and candidate_audit["label_consistent"]
            and candidate_audit["all_images_decodable"]
        ),
        "decision_rule": "zero exact and zero dHash-near overlap with development, label consistency, and full decodability",
        "limitations": [
            "No patient identifiers are distributed, so patient-level independence cannot be directly verified.",
            "Perceptual-hash screening can miss transformed reuse and can flag visually similar but distinct images.",
            "Dataset ethics, consent, and license provenance require human/institutional confirmation before submission.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(candidate, args.output_dir / "candidate_manifest.csv")
    write_rows(args.output_dir / "exact_overlaps.csv", exact_rows)
    write_rows(args.output_dir / "near_overlaps.csv", near_rows)
    (args.output_dir / "overlap_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
