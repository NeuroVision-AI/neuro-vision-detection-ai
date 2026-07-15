"""
ModelService — handles model loading, inference, and Grad-CAM generation.

Uses PIL for all image encoding to avoid dependency on opencv-python.
Models are cached in-memory after first load.
"""

import sys
import io
import base64
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

# ── sys.path fix so we can import src.* from the api/ subdirectory ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.model_factory import get_model
from src import config
from src.explainability import GradCAM, overlay_heatmap


class ModelUnavailableError(RuntimeError):
    """Raised when research inference is requested without validated weights."""


class ModelService:
    """
    Singleton-safe service for PyTorch model inference + Grad-CAM.

    Features
    --------
    - In-memory model cache (loaded once, reused across requests)
    - Automatic checkpoint discovery (best_accuracy → best_loss)
    - PIL-only image encoding (no opencv dependency)
    - Graceful Grad-CAM failure (returns null heatmap instead of 500 error)
    """

    def __init__(self):
        self.models_cache: Dict[str, torch.nn.Module] = {}
        self.model_metadata_cache: Dict[str, Dict[str, Any]] = {}
        self.device = config.DEVICE
        self.transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.PIXEL_MEAN, std=config.PIXEL_STD),
        ])

    # ── Model Loading ──────────────────────────────────────────────────────────

    def load_model(self, model_name: str) -> torch.nn.Module:
        """Load and cache a model. Tries checkpoints in priority order."""
        if model_name in self.models_cache:
            return self.models_cache[model_name]

        model = get_model(model_name, pretrained=False)

        # Checkpoint search priority
        checkpoint_candidates = [
            config.MODEL_SAVE_DIR / model_name / config.CHECKPOINT_BEST_ACC,
            config.MODEL_SAVE_DIR / model_name / config.CHECKPOINT_BEST_LOSS,
            config.MODEL_SAVE_DIR / f"{model_name}_{config.CHECKPOINT_BEST_ACC}",
            config.MODEL_SAVE_DIR / f"{model_name}_{config.CHECKPOINT_BEST_LOSS}",
            config.MODEL_SAVE_DIR / config.CHECKPOINT_BEST_ACC,
            config.MODEL_SAVE_DIR / config.CHECKPOINT_BEST_LOSS,
            config.MODEL_SAVE_DIR / config.CHECKPOINT_LAST,
        ]

        loaded = False
        for ckpt_path in checkpoint_candidates:
            if ckpt_path.exists():
                try:
                    checkpoint = torch.load(
                        ckpt_path, map_location=self.device, weights_only=True
                    )
                    state = (
                        checkpoint["model_state_dict"]
                        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
                        else checkpoint
                    )
                    if isinstance(checkpoint, dict) and checkpoint.get("class_names"):
                        if list(checkpoint["class_names"]) != list(config.CLASS_NAMES):
                            raise ValueError("Checkpoint class order does not match configured classes")
                    model.load_state_dict(state)
                    print(f"[ModelService] Loaded checkpoint: {ckpt_path.name}")
                    calibration_path = ckpt_path.parent / "calibration.json"
                    temperature = 1.0
                    calibrated = False
                    if calibration_path.exists():
                        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
                        temperature = float(calibration.get("temperature", 1.0))
                        if temperature <= 0:
                            raise ValueError("Invalid non-positive calibration temperature")
                        calibrated = True
                    self.model_metadata_cache[model_name] = {
                        "checkpoint": str(ckpt_path),
                        "temperature": temperature,
                        "calibrated": calibrated,
                        "intended_use": "research-only 2D public-data proof-of-concept",
                    }
                    loaded = True
                    break
                except Exception as e:
                    print(f"[ModelService] Failed to load {ckpt_path}: {e}")

        if not loaded:
            raise ModelUnavailableError(
                f"No trained checkpoint is available for '{model_name}'. "
                "Inference is disabled until training and evaluation are completed."
            )

        model.to(self.device)
        model.eval()
        self.models_cache[model_name] = model
        return model

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        image_bytes: bytes,
        model_name: str,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Run full inference pipeline: preprocess → forward → Grad-CAM.

        Returns
        -------
        dict with keys:
            predictions       : {class_name: probability, ...}
            predicted_class   : str
            confidence        : float
            is_uncertain      : bool
            heatmap_base64    : str | None  (base64 PNG)
        """
        model = self.load_model(model_name)
        if not 0.0 < confidence_threshold < 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        metadata = self.model_metadata_cache[model_name]

        # ── Preprocess ──
        try:
            # Check for DICOM magic bytes 'DICM' at byte offset 128
            is_dicom = False
            if len(image_bytes) > 132 and image_bytes[128:132] == b"DICM":
                is_dicom = True

            if is_dicom:
                try:
                    import pydicom
                    from pydicom.pixel_data_handlers.util import apply_voi_lut

                    ds = pydicom.dcmread(io.BytesIO(image_bytes))
                    pixel_array = ds.pixel_array

                    # Apply VOI LUT (Window Center / Width) if available
                    try:
                        pixel_array = apply_voi_lut(pixel_array, ds)
                    except Exception:
                        pass

                    # Normalize to 0-255 uint8 range
                    px_min, px_max = pixel_array.min(), pixel_array.max()
                    if px_max > px_min:
                        normalized = (pixel_array - px_min) / (px_max - px_min)
                    else:
                        normalized = np.zeros_like(pixel_array)

                    uint8_img = (normalized * 255).astype(np.uint8)
                    image = Image.fromarray(uint8_img).convert("RGB")
                    print("[ModelService] Successfully parsed DICOM image")
                except Exception as dcm_err:
                    print(f"[ModelService] DICOM parsing failed: {dcm_err}")
                    raise ValueError(f"Failed to parse DICOM: {dcm_err}")
            else:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Could not decode image: {str(e)}. Ensure you upload a valid JPEG, PNG, or DICOM (.dcm) image.")

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # ── Forward pass ──
        with torch.no_grad():
            logits = model(input_tensor) / float(metadata["temperature"])
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])
        predicted_class = config.IDX_TO_CLASS[predicted_idx]

        predictions = {
            config.IDX_TO_CLASS[i]: float(probs[i])
            for i in range(config.NUM_CLASSES)
        }
        is_uncertain = confidence < confidence_threshold

        # ── Grad-CAM ──
        heatmap_base64: Optional[str] = None
        try:
            target_layer = model.get_gradcam_target_layer()
            grad_cam = GradCAM(model, target_layer)
            cam = grad_cam.generate(input_tensor, target_class=predicted_idx)

            # Build overlay: image in [0,1] float RGB
            orig_np = np.array(
                image.resize((config.IMG_SIZE, config.IMG_SIZE))
            ).astype(np.float32) / 255.0

            overlay = overlay_heatmap(orig_np, cam)

            # Convert to uint8 PIL image → encode PNG with PIL (no cv2 needed)
            overlay_uint8 = (overlay * 255).clip(0, 255).astype(np.uint8)
            pil_overlay = Image.fromarray(overlay_uint8, mode="RGB")

            buf = io.BytesIO()
            pil_overlay.save(buf, format="PNG", optimize=True)
            heatmap_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        except Exception:
            print("[ModelService] Grad-CAM generation failed:")
            traceback.print_exc()

        # Calculate Shannon Entropy: H(P) = -sum(p * log2(p))
        # Max entropy for 4 classes is log2(4) = 2.0. Scale to [0, 1].
        epsilon = 1e-9
        entropy = -float(np.sum(probs * np.log2(probs + epsilon))) / 2.0
        entropy = max(0.0, min(1.0, entropy))

        # Research-only FHIR-shaped Observation. This deliberately avoids
        # diagnostic codes and final-report status.
        now_str = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        fhir_metadata = {
            "resourceType": "Observation",
            "status": "preliminary",
            "code": {
                "coding": [
                    {
                        "system": "https://example.org/ai-neuroonco/research-codes",
                        "code": "2D-MRI-CLASSIFIER-OUTPUT",
                        "display": "Research 2D MRI classifier output"
                    }
                ],
                "text": "Non-diagnostic research model output"
            },
            "subject": {
                "reference": "Patient/anonymous",
                "display": "Anonymous Patient Study"
            },
            "effectiveDateTime": now_str,
            "issued": now_str,
            "note": [{"text": "Not for diagnosis, prognosis, triage, or treatment decisions."}],
            "valueString": f"Dataset label: {predicted_class}; calibrated confidence: {confidence:.4f}; normalized entropy: {entropy:.4f}",
            "extension": [
                {
                    "url": "http://neuroonco.ai/fhir/StructureDefinition/prediction-entropy",
                    "valueDecimal": float(entropy)
                },
                {
                    "url": "http://neuroonco.ai/fhir/StructureDefinition/model-used",
                    "valueString": model_name
                },
                {
                    "url": "http://neuroonco.ai/fhir/StructureDefinition/calibrated",
                    "valueBoolean": bool(metadata["calibrated"])
                }
            ]
        }

        return {
            "predictions": predictions,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "is_uncertain": is_uncertain,
            "heatmap_base64": heatmap_base64,
            "entropy": entropy,
            "fhir_metadata": fhir_metadata,
            "calibrated": bool(metadata["calibrated"]),
            "temperature": float(metadata["temperature"]),
            "checkpoint": Path(metadata["checkpoint"]).name,
            "intended_use": metadata["intended_use"],
            "input_scope": "single 2D image; DICOM series/volumes are not supported",
        }
