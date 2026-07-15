#!/usr/bin/env python3
"""Capture a machine-readable execution environment without secrets."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def main() -> None:
    output_dir = ROOT / "outputs" / "environment"
    output_dir.mkdir(parents=True, exist_ok=True)
    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "requirements_sha256": sha256(ROOT / "requirements.txt"),
        "experiment_config_sha256": sha256(ROOT / "configs" / "experiment.yaml"),
        "dataset_manifest_sha256": sha256(
            ROOT / "data" / "manifests" / "dataset_manifest.csv"
        ),
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
        "note": "Environment snapshot contains no API keys or environment-variable values.",
    }
    path = output_dir / "environment_snapshot.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
