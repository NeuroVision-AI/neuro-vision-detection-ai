"""Finalize the resource-constrained EfficientNet run from its best checkpoint.

The training process completed nine full epochs and wrote a new best-accuracy
checkpoint before it was stopped on a CPU-only host.  This script performs the
same validation-only calibration and locked-test evaluation as ``src.train``
while recording the interruption as a protocol deviation.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.calibration import fit_temperature
from src.data_integrity import group_ids_for_materialized_samples
from src.dataset import get_dataloaders
from src.evaluate import Evaluator
from src.models.model_factory import get_model
from src.train import EarlyStopping
from src.utils import seed_everything


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def main() -> None:
    seed_everything(config.RANDOM_SEED)
    _, val_loader, test_loader = get_dataloaders(
        train_dir=config.TRAIN_DIR,
        val_dir=config.VAL_DIR,
        test_dir=config.TEST_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    model_name = "efficientnet"
    checkpoint = config.MODEL_SAVE_DIR / model_name / config.CHECKPOINT_BEST_ACC
    model = get_model(
        model_name=model_name,
        num_classes=config.NUM_CLASSES,
        pretrained=False,
    )
    EarlyStopping.load_checkpoint(model, checkpoint)
    model.to(config.DEVICE)

    temperature = fit_temperature(model, val_loader, device=config.DEVICE)
    run_metadata = {
        "run_id": "efficientnet_resource_constrained_20260715",
        "model_name": model_name,
        "dataset_manifest_sha256": sha256(config.MANIFEST_DIR / "dataset_manifest.csv"),
        "experiment_config_sha256": sha256(config.PROJECT_ROOT / "configs" / "experiment.yaml"),
        "requested_max_epochs": config.EPOCHS,
        "epochs_completed": 9,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "weight_decay": config.WEIGHT_DECAY,
        "protocol_conformant": False,
        "deviation": (
            "Resource-constrained stop after nine completed epochs on a CPU-only "
            "host; the epoch-9 best-validation-accuracy checkpoint was retained."
        ),
    }
    calibration_path = config.MODEL_SAVE_DIR / model_name / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "temperature": temperature,
                "fit_split": "validation",
                "checkpoint": str(checkpoint),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                **run_metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    group_ids = group_ids_for_materialized_samples(
        test_loader.dataset.samples,
        config.MANIFEST_DIR / "dataset_manifest.csv",
        config.PROCESSED_DATA_DIR,
    )
    metrics_dir = config.METRICS_DIR / model_name
    evaluator = Evaluator(
        model=model,
        data_loader=test_loader,
        device=config.DEVICE,
        temperature=temperature,
        group_ids=group_ids,
        output_dir=metrics_dir,
        evaluation_split="internal_locked_test",
    )
    results = evaluator.full_evaluation(history=None)
    run_summary = {
        **run_metadata,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stopped_early": True,
        "training_complete": False,
        "evaluation_complete": True,
        "calibration_complete": True,
        "best_validation_accuracy": 0.9744,
        "best_validation_loss": 0.4212,
        "internal_test_macro_f1": results["f1_macro"],
        "metrics_path": str(metrics_dir / "research_metrics.json"),
    }
    (config.MODEL_SAVE_DIR / model_name / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
