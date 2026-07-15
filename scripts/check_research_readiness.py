#!/usr/bin/env python3
"""Create an honest machine-readable readiness report for the research paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config


def count_images(root: Path) -> int:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".dcm"}
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions) if root.exists() else 0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit research execution readiness")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "research_readiness.json",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero until paper prerequisites are complete")
    args = parser.parse_args()

    audit_path = config.MANIFEST_DIR / "dataset_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    split_counts = {name: count_images(config.PROCESSED_DATA_DIR / name) for name in ("train", "val", "test")}
    checkpoints = {}
    completed_runs = {}
    evaluated_runs = {}
    for model_name in ("efficientnet", "custom_cnn"):
        model_dir = config.MODEL_SAVE_DIR / model_name
        run_summary_path = model_dir / "run_summary.json"
        run_summary = (
            json.loads(run_summary_path.read_text(encoding="utf-8"))
            if run_summary_path.exists()
            else {}
        )
        checkpoints[model_name] = {
            "best_accuracy": (model_dir / config.CHECKPOINT_BEST_ACC).exists(),
            "best_loss": (model_dir / config.CHECKPOINT_BEST_LOSS).exists(),
            "calibration": (model_dir / "calibration.json").exists(),
        }
        completed_runs[model_name] = bool(
            run_summary.get("protocol_conformant")
            and run_summary.get("training_complete")
            and run_summary.get("evaluation_complete")
            and run_summary.get("calibration_complete")
            and (config.METRICS_DIR / model_name / "research_metrics.json").exists()
        )
        evaluated_runs[model_name] = bool(
            run_summary.get("evaluation_complete")
            and run_summary.get("calibration_complete")
            and (config.METRICS_DIR / model_name / "research_metrics.json").exists()
        )

    implementation_requirements = {
        "literature_tracker_present": (PROJECT_ROOT / "AI_NeuroOnco_Literature_Tracker.xlsx").exists(),
        "literature_audit_present": (
            PROJECT_ROOT
            / "outputs"
            / "literature_audit_2026-07-15"
            / "AI_NeuroOnco_Literature_Audit_and_Index.xlsx"
        ).exists(),
        "literature_synthesis_present": (
            PROJECT_ROOT / "research" / "LITERATURE_SYNTHESIS.md"
        ).exists(),
        "full_text_extraction_present": (
            PROJECT_ROOT / "research" / "FULL_TEXT_EXTRACTION.md"
        ).exists(),
        "track_handoff_present": (
            PROJECT_ROOT / "research" / "TRACK1_TRACK2_HANDOFF.md"
        ).exists(),
        "data_quality_report_present": (
            PROJECT_ROOT / "outputs" / "data_quality" / "data_quality_report.json"
        ).exists(),
        "protocol_present": (PROJECT_ROOT / "research" / "PROTOCOL.md").exists(),
        "experiment_config_present": (PROJECT_ROOT / "configs" / "experiment.yaml").exists(),
        "environment_snapshot_present": (
            PROJECT_ROOT / "outputs" / "environment" / "environment_snapshot.json"
        ).exists(),
        "manuscript_shell_present": (PROJECT_ROOT / "research" / "MANUSCRIPT_DRAFT.md").exists(),
        "data_card_present": (PROJECT_ROOT / "research" / "DATA_CARD.md").exists(),
        "model_card_present": (PROJECT_ROOT / "research" / "MODEL_CARD.md").exists(),
        "reporting_checklist_present": (PROJECT_ROOT / "research" / "REPORTING_CHECKLIST.md").exists(),
        "leakage_pipeline_present": (PROJECT_ROOT / "src" / "data_integrity.py").exists(),
        "calibration_and_statistics_present": (PROJECT_ROOT / "src" / "calibration.py").exists()
        and (PROJECT_ROOT / "src" / "evaluate.py").exists(),
        "external_evaluation_runner_present": (PROJECT_ROOT / "scripts" / "evaluate_external.py").exists(),
        "external_overlap_audit_runner_present": (PROJECT_ROOT / "scripts" / "audit_external_candidate.py").exists(),
        "xai_evaluation_runner_present": (PROJECT_ROOT / "scripts" / "evaluate_xai.py").exists(),
        "rag_evaluation_runner_present": (PROJECT_ROOT / "scripts" / "evaluate_rag.py").exists(),
        "inference_fail_closed_present": (PROJECT_ROOT / "tests" / "test_model_service_safety.py").exists(),
    }
    external_audits = list((PROJECT_ROOT / "data" / "external_manifests").glob("*/overlap_audit.json"))
    external_audit_reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in external_audits
    ]
    xai_expert_metadata_path = PROJECT_ROOT / "research" / "xai_expert_review_metadata.json"
    xai_expert_metadata = (
        json.loads(xai_expert_metadata_path.read_text(encoding="utf-8"))
        if xai_expert_metadata_path.exists()
        else {}
    )
    core_empirical_requirements = {
        "manifest_present": (config.MANIFEST_DIR / "dataset_manifest.csv").exists(),
        "source_provenance_present": (config.MANIFEST_DIR / "source_provenance.json").exists(),
        "dataset_audit_passed": bool(audit.get("leakage_free")) and bool(audit.get("label_consistent")) and bool(audit.get("all_images_decodable")),
        "train_split_present": split_counts["train"] > 0,
        "validation_split_present": split_counts["val"] > 0,
        "locked_test_split_present": split_counts["test"] > 0,
        "efficientnet_evaluated": checkpoints["efficientnet"]["best_accuracy"] and evaluated_runs["efficientnet"],
        "efficientnet_protocol_conformant_complete": completed_runs["efficientnet"],
        "custom_cnn_trained": checkpoints["custom_cnn"]["best_accuracy"] and completed_runs["custom_cnn"],
        "both_models_calibrated": all(
            checkpoints[name]["calibration"] and evaluated_runs[name]
            for name in ("efficientnet", "custom_cnn")
        ),
        "paired_internal_model_comparison_present": (
            PROJECT_ROOT / "outputs" / "model_comparison" / "model_comparison.json"
        ).exists(),
        "external_candidates_audited": bool(external_audit_reports),
        "qualifying_external_candidate_present": any(
            report.get("independent_external_validation_eligible")
            for report in external_audit_reports
        ),
        "external_evaluation_present": any(
            (PROJECT_ROOT / "outputs" / "external_evaluation").rglob("research_metrics.json")
        ),
        "xai_evaluation_present": (PROJECT_ROOT / "outputs" / "xai_evaluation" / "xai_metrics.json").exists(),
        "xai_blinded_expert_review_complete": bool(xai_expert_metadata.get("complete")),
    }
    ancillary_requirements = {
        "rag_retrieval_evaluation_present": (PROJECT_ROOT / "outputs" / "rag_evaluation" / "retrieval_metrics.json").exists(),
        "rag_benchmark_expert_reviewed": bool(
            json.loads(
                (PROJECT_ROOT / "research" / "rag_benchmark_metadata.json").read_text(encoding="utf-8")
            ).get("domain_expert_reviewed")
        ) if (PROJECT_ROOT / "research" / "rag_benchmark_metadata.json").exists() else False,
    }
    implementation_blockers = [name for name, value in implementation_requirements.items() if not value]
    empirical_blockers = [name for name, value in core_empirical_requirements.items() if not value]
    ancillary_blockers = [name for name, value in ancillary_requirements.items() if not value]
    requirements = {**implementation_requirements, **core_empirical_requirements, **ancillary_requirements}
    implementation_fraction = sum(implementation_requirements.values()) / len(implementation_requirements)
    empirical_fraction = sum(core_empirical_requirements.values()) / len(core_empirical_requirements)
    ancillary_fraction = sum(ancillary_requirements.values()) / len(ancillary_requirements)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "paper_ready": not implementation_blockers and not empirical_blockers,
        "implementation_complete": not implementation_blockers,
        "implementation_readiness_fraction": implementation_fraction,
        "empirical_readiness_fraction": empirical_fraction,
        "ancillary_readiness_fraction": ancillary_fraction,
        "ancillary_ready": not ancillary_blockers,
        "readiness_fraction": sum(bool(value) for value in requirements.values()) / len(requirements),
        "requirements": requirements,
        "implementation_requirements": implementation_requirements,
        "empirical_requirements": core_empirical_requirements,
        "ancillary_requirements": ancillary_requirements,
        "blocking_prerequisites": {
            "implementation": implementation_blockers,
            "empirical": empirical_blockers,
            "ancillary": ancillary_blockers,
        },
        "processed_image_counts": split_counts,
        "checkpoints": checkpoints,
        "protocol_conformant_completed_runs": completed_runs,
        "evaluated_runs": evaluated_runs,
        "dataset_audit": audit,
        "external_candidate_audits": external_audit_reports,
        "requirements_sha256": sha256(PROJECT_ROOT / "requirements.txt"),
        "source_literature_tracker_sha256": sha256(
            PROJECT_ROOT / "AI_NeuroOnco_Literature_Tracker.xlsx"
        ),
        "claim_scope": "research-only 2D public-data proof-of-concept",
        "note": "No empirical result is inferred from missing artifacts.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.strict and (implementation_blockers or empirical_blockers):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
