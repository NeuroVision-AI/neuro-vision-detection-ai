#!/usr/bin/env python3
"""Prepare a leakage-resistant dataset and provenance manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data_integrity import prepare_research_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build provenance manifests and patient/duplicate-grouped splits."
    )
    parser.add_argument("source", type=Path, help="Dataset root")
    parser.add_argument(
        "--output",
        type=Path,
        default=config.PROCESSED_DATA_DIR,
        help="Prepared split directory",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=config.MANIFEST_DIR,
        help="Manifest and audit output directory",
    )
    parser.add_argument("--source-name", default="public_mri", help="Dataset provenance label")
    parser.add_argument(
        "--patient-pattern",
        default=None,
        help="Regex with a named 'patient' group or first capture group",
    )
    parser.add_argument(
        "--do-not-preserve-source-test",
        action="store_true",
        help="Ignore any existing Testing/test folder (not recommended)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing processed dataset after it has been archived",
    )
    parser.add_argument(
        "--exclude-conflicting-groups",
        action="store_true",
        help=(
            "Explicitly exclude every record in a patient/duplicate group with conflicting labels; "
            "the excluded records and pre-exclusion audit are retained"
        ),
    )
    args = parser.parse_args()

    audit = prepare_research_dataset(
        source_root=args.source,
        output_root=args.output,
        manifest_dir=args.manifest_dir,
        source_name=args.source_name,
        patient_pattern=args.patient_pattern,
        seed=config.RANDOM_SEED,
        preserve_source_test=not args.do_not_preserve_source_test,
        overwrite=args.overwrite,
        exclude_conflicting_groups=args.exclude_conflicting_groups,
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
