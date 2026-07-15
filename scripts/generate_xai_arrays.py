#!/usr/bin/env python3
"""Generate reproducible Grad-CAM, repeat, randomization, and mask arrays."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.dataset import get_val_transforms
from src.explainability import GradCAM
from src.models.model_factory import get_model
from src.utils import seed_everything


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # Multiple candidate files may point to the same development image. Evaluate
    # each locked development case once.
    unique = {}
    for row in rows:
        unique.setdefault(row["development_materialized_path"], row)
    return list(unique.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate XAI evidence arrays")
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["efficientnet", "custom_cnn"], required=True)
    parser.add_argument("--development-root", type=Path, default=config.PROCESSED_DATA_DIR)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "xai_arrays")
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    rows = load_rows(args.mapping)
    if args.max_cases is not None:
        rows = rows[: args.max_cases]
    if not rows:
        raise ValueError("XAI mapping is empty")

    seed_everything(config.RANDOM_SEED)
    model = get_model(args.model, num_classes=config.NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.to(config.DEVICE).eval()

    seed_everything(config.RANDOM_SEED + 1)
    randomized_model = get_model(args.model, num_classes=config.NUM_CLASSES, pretrained=False)
    randomized_model.to(config.DEVICE).eval()

    trained_cam = GradCAM(model, model.get_gradcam_target_layer())
    randomized_cam = GradCAM(randomized_model, randomized_model.get_gradcam_target_layer())
    transform = get_val_transforms()
    for directory in ("original", "repeat", "randomized", "masks"):
        (args.output_root / directory).mkdir(parents=True, exist_ok=True)

    case_rows = []
    for index, row in enumerate(rows, start=1):
        image_path = args.development_root / row["development_materialized_path"]
        mask_path = args.candidate_root / row["mask_path"]
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor.to(config.DEVICE))
            predicted_index = int(logits.argmax(dim=1).item())

        original = trained_cam.generate(tensor, target_class=predicted_index)
        repeat = trained_cam.generate(tensor, target_class=predicted_index)
        randomized = randomized_cam.generate(tensor, target_class=predicted_index)
        mask = np.asarray(
            Image.open(mask_path).convert("L").resize(
                (config.IMG_SIZE, config.IMG_SIZE), Image.Resampling.NEAREST
            )
        ) > 0

        case_name = f"case_{index:05d}"
        np.save(args.output_root / "original" / f"{case_name}.npy", original)
        np.save(args.output_root / "repeat" / f"{case_name}.npy", repeat)
        np.save(args.output_root / "randomized" / f"{case_name}.npy", randomized)
        np.save(args.output_root / "masks" / f"{case_name}.npy", mask.astype(np.uint8))
        true_label = row["development_label"]
        case_rows.append(
            {
                "case": case_name,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "true_label": true_label,
                "predicted_label": config.IDX_TO_CLASS[predicted_index],
                "correct": config.IDX_TO_CLASS[predicted_index] == true_label,
                "explained_class": config.IDX_TO_CLASS[predicted_index],
            }
        )
        if index % 50 == 0 or index == len(rows):
            print(f"Generated {index}/{len(rows)} XAI cases")

    with (args.output_root / "cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(case_rows)
    (args.output_root / "generation_metadata.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "model": args.model,
                "checkpoint": str(args.checkpoint),
                "cases": len(case_rows),
                "target_policy": "model_predicted_class",
                "randomization": "independently initialized full model",
                "localization_scope": "overlapping locked internal-test images with BRISC masks; not external validation",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
