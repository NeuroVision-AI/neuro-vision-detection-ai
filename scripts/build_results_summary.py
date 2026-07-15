#!/usr/bin/env python3
"""Consolidate generated study artifacts into honest paper-facing result tables."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fmt(value, digits: int = 3) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def model_row(model: str, metrics: dict, run_summary: dict) -> dict:
    if not metrics:
        return {
            "dataset": "development",
            "split": "internal_locked_test",
            "model": model,
            "status": "not_complete",
            "notes": "No protocol-conformant completed-run artifact",
        }
    accuracy_ci = metrics["confidence_intervals"]["accuracy"]
    f1_ci = metrics["confidence_intervals"]["f1_macro"]
    comparison = metrics.get("calibration_comparison", {})
    protocol_conformant = bool(run_summary.get("protocol_conformant"))
    notes = "Internal source test; bootstrap groups are duplicate/provenance components, not verified patients"
    if not protocol_conformant:
        notes += "; resource-constrained run, not protocol-conformant"
    return {
        "dataset": "development",
        "split": metrics.get("evaluation_split", "internal_locked_test"),
        "model": model,
        "status": "complete" if protocol_conformant else "evaluated_resource_constrained",
        "n_patients_or_groups": metrics.get("n_bootstrap_groups"),
        "n_images": metrics.get("n_images"),
        "accuracy": metrics.get("accuracy"),
        "accuracy_ci_low": accuracy_ci.get("lower_95"),
        "accuracy_ci_high": accuracy_ci.get("upper_95"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "macro_f1": metrics.get("f1_macro"),
        "macro_f1_ci_low": f1_ci.get("lower_95"),
        "macro_f1_ci_high": f1_ci.get("upper_95"),
        "mcc": metrics.get("mcc"),
        "macro_roc_auc_ovr": metrics.get("roc_auc_macro_ovr"),
        "macro_pr_auc_ovr": metrics.get("pr_auc_macro_ovr"),
        "ece_uncalibrated": comparison.get("uncalibrated", {}).get("expected_calibration_error"),
        "ece_calibrated": metrics.get("expected_calibration_error"),
        "brier_uncalibrated": comparison.get("uncalibrated", {}).get("multiclass_brier_score"),
        "brier_calibrated": metrics.get("multiclass_brier_score"),
        "nll_uncalibrated": comparison.get("uncalibrated", {}).get("negative_log_likelihood"),
        "nll_calibrated": metrics.get("negative_log_likelihood"),
        "temperature": metrics.get("temperature"),
        "notes": notes,
    }


def main() -> None:
    metrics = {
        model: read_json(ROOT / "outputs" / "metrics" / model / "research_metrics.json")
        for model in ("efficientnet", "custom_cnn")
    }
    run_summaries = {
        model: read_json(ROOT / "outputs" / "models" / model / "run_summary.json")
        for model in ("efficientnet", "custom_cnn")
    }
    rows = [
        model_row(model, metrics[model], run_summaries[model])
        for model in ("efficientnet", "custom_cnn")
    ]
    fields = [
        "dataset", "split", "model", "status", "n_patients_or_groups", "n_images",
        "accuracy", "accuracy_ci_low", "accuracy_ci_high", "balanced_accuracy",
        "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "mcc",
        "macro_roc_auc_ovr", "macro_pr_auc_ovr", "ece_uncalibrated", "ece_calibrated",
        "brier_uncalibrated", "brier_calibrated", "nll_uncalibrated", "nll_calibrated",
        "temperature", "notes",
    ]
    csv_path = ROOT / "research" / "results_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    xai = read_json(ROOT / "outputs" / "xai_evaluation" / "xai_metrics.json")
    rag = read_json(ROOT / "outputs" / "rag_evaluation" / "retrieval_metrics.json")
    readiness = read_json(ROOT / "outputs" / "research_readiness.json")
    model_comparison = read_json(
        ROOT / "outputs" / "model_comparison" / "model_comparison.json"
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_results": rows,
        "xai": xai,
        "rag": rag,
        "paired_internal_model_comparison": model_comparison,
        "readiness": readiness,
        "external_primary_endpoint": {
            "status": "not_evaluated",
            "reason": "BRISC and BDNeuro-MRI v7 failed independent cross-corpus reuse audits",
        },
    }
    output_json = ROOT / "outputs" / "research_results_summary.json"
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Generated research results summary",
        "",
        "> Values below are copied from generated artifacts. The external primary endpoint remains unevaluated.",
        "> EfficientNet was evaluated from the frozen epoch-9 best-validation-accuracy checkpoint after a CPU-resource stop and is not a protocol-conformant completed primary run.",
        "",
        "## Internal locked-test results",
        "",
        "| Model | Images / groups | Accuracy (95% CI) | Macro-F1 (95% CI) | MCC | ECE before → after | Temperature |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("status") not in {"complete", "evaluated_resource_constrained"}:
            lines.append(f"| {row['model']} | NA | NA | NA | NA | NA | NA |")
            continue
        lines.append(
            f"| {row['model']} | {row['n_images']} / {row['n_patients_or_groups']} | "
            f"{fmt(row['accuracy'])} ({fmt(row['accuracy_ci_low'])}–{fmt(row['accuracy_ci_high'])}) | "
            f"{fmt(row['macro_f1'])} ({fmt(row['macro_f1_ci_low'])}–{fmt(row['macro_f1_ci_high'])}) | "
            f"{fmt(row['mcc'])} | {fmt(row['ece_uncalibrated'])} → {fmt(row['ece_calibrated'])} | "
            f"{fmt(row['temperature'])} |"
        )
    if model_comparison:
        differences = model_comparison["paired_differences_primary_minus_comparator"]
        accuracy_ci = differences["accuracy_difference"]
        f1_ci = differences["macro_f1_difference"]
        mcnemar = model_comparison["mcnemar_exact"]
        paired_summary = (
            f"EfficientNet minus custom-CNN accuracy: {fmt(differences['accuracy'])} "
            f"(group-bootstrap 95% CI {fmt(accuracy_ci['lower_95'])} to "
            f"{fmt(accuracy_ci['upper_95'])}); macro-F1 difference: "
            f"{fmt(differences['macro_f1'])} (95% CI {fmt(f1_ci['lower_95'])} to "
            f"{fmt(f1_ci['upper_95'])}). Exploratory image-level exact McNemar p="
            f"{mcnemar['two_sided_p_value']:.4g}."
        )
    else:
        paired_summary = "Not complete."
    lines.extend(
        [
            "",
            "Duplicate/provenance components—not verified patients—were the bootstrap unit because patient identifiers were unavailable.",
            "",
            "## Paired internal comparison",
            "",
            paired_summary,
            "",
            "## External validation",
            "",
            "Not evaluated. BRISC and BDNeuro-MRI v7 were rejected before performance evaluation because material exact cross-corpus reuse was detected.",
            "",
            "## Explainability and RAG",
            "",
            f"- Quantitative Grad-CAM cases: {xai.get('cases', 'not complete')}.",
            f"- Preliminary RAG recall@5: {fmt(rag.get('aggregate', {}).get('recall_at_k'))}; hit@5: {fmt(rag.get('aggregate', {}).get('hit_at_k'))}; MRR: {fmt(rag.get('aggregate', {}).get('reciprocal_rank'))}.",
            "- XAI expert review and RAG benchmark expert review remain incomplete.",
        ]
    )
    (ROOT / "research" / "RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(csv_path)
    print(output_json)


if __name__ == "__main__":
    main()
