from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Optional
import traceback

from api.services.model_service import ModelService, ModelUnavailableError
from api.services.report_service import ReportService

router = APIRouter(prefix="/predict", tags=["Prediction"])

# Singletons
_service = ModelService()
_report_service = ReportService()


class ReportRequest(BaseModel):
    predicted_class: str
    confidence: float
    predictions: Dict[str, float]
    model_used: str
    heatmap_base64: Optional[str] = None
    patient_name: Optional[str] = "Anonymous Patient"
    patient_id: Optional[str] = "N/A"
    comments: Optional[str] = ""
    calibrated: bool = False
    confidence_threshold: float = 0.7


@router.post("/")
async def predict_image(
    file: UploadFile = File(..., description="Brain MRI image (JPEG/PNG)"),
    model_name: str = Form("efficientnet", description="Model architecture to use"),
    threshold: float = Form(0.7, description="Confidence threshold for uncertainty flagging"),
):
    """
    Classify a brain MRI image and return prediction + Grad-CAM heatmap.

    Returns:
    - **predicted_class**: Top predicted tumor class
    - **confidence**: Probability of predicted class
    - **is_uncertain**: True if confidence < threshold
    - **predictions**: Full probability distribution over all classes
    - **heatmap_base64**: Base64-encoded Grad-CAM overlay PNG (or null)
    - **model_used**: Which model architecture was used
    """
    filename = file.filename or ""
    is_dcm = filename.lower().endswith(".dcm")
    if not is_dcm and (not file.content_type or not file.content_type.startswith("image/")):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image or DICOM file. Received: {file.content_type}"
        )

    try:
        image_bytes = await file.read()
        result = _service.predict(image_bytes, model_name, threshold)
        result["model_used"] = model_name
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/report")
async def generate_report(request: ReportRequest):
    """
    Generates a research-only PDF summary for the classification output.
    and streams it back for download.
    """
    try:
        pdf_buffer = _report_service.generate_pdf_report(
            predicted_class=request.predicted_class,
            confidence=request.confidence,
            predictions=request.predictions,
            model_used=request.model_used,
            heatmap_base64=request.heatmap_base64,
            patient_name=request.patient_name,
            patient_id=request.patient_id,
            comments=request.comments,
            calibrated=request.calibrated,
            confidence_threshold=request.confidence_threshold,
        )
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=AI_NeuroOnco_Report.pdf",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@router.get("/models")
async def list_models():
    """Return available model architectures and their descriptions."""
    from src import config

    def trained(model_name: str) -> bool:
        model_dir = config.MODEL_SAVE_DIR / model_name
        return any((model_dir / name).exists() for name in [config.CHECKPOINT_BEST_ACC, config.CHECKPOINT_BEST_LOSS])

    return {
        "models": [
            {
                "id": "efficientnet",
                "name": "EfficientNet-B0",
                "description": "ImageNet transfer-learning baseline; performance is unavailable until evaluated.",
                "params": "~5.3M",
                "recommended": True,
                "trained": trained("efficientnet"),
            },
            {
                "id": "custom_cnn",
                "name": "Custom CNN",
                "description": "Lightweight 5-block CNN. ~2M parameters. Faster inference.",
                "params": "~2M",
                "recommended": False,
                "trained": trained("custom_cnn"),
            },
        ]
    }


@router.get("/health")
async def model_health():
    """Check if the model service and dependencies are healthy."""
    try:
        import torch
        from src import config
        trained_models = {
            model_name: any(
                (config.MODEL_SAVE_DIR / model_name / checkpoint).exists()
                for checkpoint in [config.CHECKPOINT_BEST_ACC, config.CHECKPOINT_BEST_LOSS]
            )
            for model_name in ["efficientnet", "custom_cnn"]
        }
        return {
            "status": "ready" if any(trained_models.values()) else "not_ready",
            "device": str(config.DEVICE),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
            "num_classes": config.NUM_CLASSES,
            "class_names": config.CLASS_NAMES,
            "cached_models": list(_service.models_cache.keys()),
            "trained_models": trained_models,
            "intended_use": "research-only 2D public-data proof-of-concept",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_model_metrics(model_name: str = "efficientnet"):
    """
    Retrieve classification model evaluation metrics, reports,
    and performance plots (ROC, confusion matrix) as base64 strings.
    """
    import base64
    import json
    from src import config

    if model_name not in {"efficientnet", "custom_cnn"}:
        raise HTTPException(status_code=400, detail="Unknown model_name")
    metrics_dir = config.METRICS_DIR / model_name

    # Check if metrics exist
    report_json_path = metrics_dir / "classification_report.json"
    conf_matrix_path = metrics_dir / "confusion_matrix.png"
    roc_curves_path = metrics_dir / "roc_curves.png"
    history_path = metrics_dir / "training_history.png"
    calibration_path = metrics_dir / "calibration_curve.png"
    risk_coverage_path = metrics_dir / "risk_coverage.png"
    research_metrics_path = metrics_dir / "research_metrics.json"
    report_txt_path = metrics_dir / "classification_report.txt"

    has_metrics = report_json_path.exists() or conf_matrix_path.exists()

    response = {
        "trained": has_metrics,
        "metrics_directory": str(metrics_dir),
        "confusion_matrix_base64": None,
        "roc_curves_base64": None,
        "training_history_base64": None,
        "calibration_curve_base64": None,
        "risk_coverage_base64": None,
        "research_metrics_json": None,
        "classification_report_text": None,
        "classification_report_json": None,
        "metadata": {
            "device": str(config.DEVICE),
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "epochs_max": config.EPOCHS,
            "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
            "classes": config.CLASS_NAMES,
            "default_model": config.DEFAULT_MODEL,
            "model_name": model_name,
        }
    }

    if has_metrics:
        try:
            # Load images
            if conf_matrix_path.exists():
                with open(conf_matrix_path, "rb") as f:
                    response["confusion_matrix_base64"] = base64.b64encode(f.read()).decode("utf-8")
            if roc_curves_path.exists():
                with open(roc_curves_path, "rb") as f:
                    response["roc_curves_base64"] = base64.b64encode(f.read()).decode("utf-8")
            if history_path.exists():
                with open(history_path, "rb") as f:
                    response["training_history_base64"] = base64.b64encode(f.read()).decode("utf-8")
            if calibration_path.exists():
                with open(calibration_path, "rb") as f:
                    response["calibration_curve_base64"] = base64.b64encode(f.read()).decode("utf-8")
            if risk_coverage_path.exists():
                with open(risk_coverage_path, "rb") as f:
                    response["risk_coverage_base64"] = base64.b64encode(f.read()).decode("utf-8")

            # Load text report
            if report_txt_path.exists():
                with open(report_txt_path, "r", encoding="utf-8") as f:
                    response["classification_report_text"] = f.read()

            # Load JSON report
            if report_json_path.exists():
                with open(report_json_path, "r", encoding="utf-8") as f:
                    response["classification_report_json"] = json.load(f)
            if research_metrics_path.exists():
                with open(research_metrics_path, "r", encoding="utf-8") as f:
                    response["research_metrics_json"] = json.load(f)
        except Exception as e:
            print(f"[predict.py] Failed to load metrics data: {e}")

    return response
