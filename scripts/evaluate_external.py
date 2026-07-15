#!/usr/bin/env python3
"""Evaluate a locked external dataset with grouped bootstrap confidence intervals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.dataset import BrainTumorDataset, get_val_transforms
from src.data_integrity import group_ids_for_materialized_samples
from src.evaluate import Evaluator
from src.models.model_factory import get_model
from torch.utils.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked external evaluation")
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["efficientnet", "custom_cnn"], default="efficientnet")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "external_evaluation",
    )
    args = parser.parse_args()

    dataset = BrainTumorDataset(
        root_dir=args.test_dir,
        transform=get_val_transforms(),
        class_names=config.CLASS_NAMES,
    )
    if not len(dataset):
        raise ValueError("External test directory contains no images")
    group_ids = group_ids_for_materialized_samples(
        dataset.samples, args.manifest, args.test_dir.parent
    )
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    model = get_model(args.model, num_classes=config.NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    evaluator = Evaluator(
        model=model,
        data_loader=loader,
        device=config.DEVICE,
        temperature=args.temperature,
        group_ids=group_ids,
        output_dir=args.output_dir,
        evaluation_split="locked_external_test",
    )
    results = evaluator.full_evaluation()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
