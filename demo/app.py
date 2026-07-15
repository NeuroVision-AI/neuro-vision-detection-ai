"""
Gradio Demo Application — Brain Tumor Classification.

Provides an interactive web interface for:
    1. Uploading an MRI image
    2. Selecting a model (EfficientNet-B0 or Custom CNN)
    3. Getting real-time predictions with confidence scores
    4. Viewing Grad-CAM heatmap explanations
    5. Flagging uncertain predictions for human review

Launch:
    python -m demo.app
    # or
    python demo/app.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from PIL import Image
from torchvision import transforms

from src import config
from src.models.model_factory import get_model
from src.explainability import GradCAM, overlay_heatmap


# ──────────────────────────────────────────────
# Global State
# ──────────────────────────────────────────────
LOADED_MODELS = {}


def load_model(model_name: str) -> torch.nn.Module:
    """
    Load a trained model from checkpoint, with caching.

    Falls back to an untrained model if no checkpoint is found
    (useful for demo/testing before training).
    """
    if model_name in LOADED_MODELS:
        return LOADED_MODELS[model_name]

    model = get_model(model_name)

    # Try to load best checkpoint
    checkpoint_path = config.MODEL_SAVE_DIR / f"{model_name}_{config.CHECKPOINT_BEST_ACC}"
    if not checkpoint_path.exists():
        # Try alternative naming
        checkpoint_path = config.MODEL_SAVE_DIR / config.CHECKPOINT_BEST_ACC

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE,
                                weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Loaded checkpoint: {checkpoint_path.name}")
    else:
        print(f"⚠ No checkpoint found for '{model_name}'. Using untrained model.")

    model.eval()
    LOADED_MODELS[model_name] = model
    return model


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """Preprocess an uploaded image for model input."""
    if image is None:
        return None

    pil_image = Image.fromarray(image).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.PIXEL_MEAN, std=config.PIXEL_STD),
    ])
    return transform(pil_image).unsqueeze(0)


def predict(image: np.ndarray, model_name: str, show_gradcam: bool = True,
            confidence_threshold: float = config.CONFIDENCE_THRESHOLD):
    """
    Run prediction on an uploaded MRI image.

    Returns:
        - Confidence scores (dict for Gradio label output)
        - Grad-CAM overlay image (or None)
        - Status text with prediction details
    """
    if image is None:
        return {}, None, "⚠️ Please upload an MRI image."

    # Load model
    model = load_model(model_name)

    # Preprocess
    input_tensor = preprocess_image(image)
    input_tensor = input_tensor.to(config.DEVICE)

    # Prediction
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]

    # Build confidence dict
    confidences = {
        config.IDX_TO_CLASS[i]: float(probs[i])
        for i in range(config.NUM_CLASSES)
    }

    predicted_idx = probs.argmax().item()
    predicted_class = config.IDX_TO_CLASS[predicted_idx]
    confidence = probs[predicted_idx].item()

    # Uncertainty check
    is_uncertain = confidence < confidence_threshold

    # Status text
    status_lines = [
        f"## {'⚠️ UNCERTAIN — Refer to Specialist' if is_uncertain else '✅ Prediction Complete'}",
        f"",
        f"**Predicted Class:** {predicted_class.replace('_', ' ').title()}",
        f"",
        f"**Confidence:** {confidence:.1%}",
        f"",
        f"**Model:** {model_name}",
        f"",
        "### All Probabilities",
    ]
    for cls_name in config.CLASS_NAMES:
        prob = confidences[cls_name]
        bar = "█" * int(prob * 20)
        status_lines.append(
            f"- **{cls_name.replace('_', ' ').title()}**: {prob:.1%} {bar}"
        )

    if is_uncertain:
        status_lines.extend([
            "",
            f"> ⚠️ Confidence ({confidence:.1%}) is below the threshold "
            f"({confidence_threshold:.0%}). This case should be reviewed "
            f"by a medical professional.",
        ])

    status_text = "\n".join(status_lines)

    # Grad-CAM
    gradcam_image = None
    if show_gradcam:
        try:
            target_layer = model.get_gradcam_target_layer()
            grad_cam = GradCAM(model, target_layer)
            heatmap = grad_cam.generate(input_tensor, target_class=predicted_idx)

            # Create overlay on original image
            original_resized = np.array(
                Image.fromarray(image).convert("RGB").resize(
                    (config.IMG_SIZE, config.IMG_SIZE)
                )
            ).astype(np.float32) / 255.0

            overlay = overlay_heatmap(original_resized, heatmap)

            # Create side-by-side figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(original_resized)
            axes[0].set_title("Original MRI", fontsize=12, fontweight="bold")
            axes[0].axis("off")

            axes[1].imshow(heatmap, cmap=config.GRADCAM_COLORMAP)
            axes[1].set_title("Grad-CAM Heatmap", fontsize=12, fontweight="bold")
            axes[1].axis("off")

            axes[2].imshow(overlay)
            axes[2].set_title("Overlay", fontsize=12, fontweight="bold")
            axes[2].axis("off")

            color = "#E74C3C" if is_uncertain else "#2ECC71"
            fig.suptitle(
                f"Prediction: {predicted_class.upper()} ({confidence:.1%})",
                fontsize=14, fontweight="bold", color=color,
            )
            plt.tight_layout()

            # Convert figure to numpy array
            fig.canvas.draw()
            gradcam_image = np.frombuffer(
                fig.canvas.tostring_rgb(), dtype=np.uint8
            ).reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)

        except Exception as e:
            status_text += f"\n\n> ⚠️ Grad-CAM generation failed: {e}"

    return confidences, gradcam_image, status_text


def build_demo() -> gr.Blocks:
    """Build the Gradio demo interface."""

    with gr.Blocks(
        title="🧠 AI NeuroOnco — Brain Tumor Classifier",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
        ),
        css="""
        .gradio-container { max-width: 1200px; margin: auto; }
        .header { text-align: center; margin-bottom: 1em; }
        .footer { text-align: center; font-size: 0.85em; color: #666;
                   margin-top: 2em; padding-top: 1em;
                   border-top: 1px solid #ddd; }
        """,
    ) as demo:

        # Header
        gr.Markdown(
            """
            <div class="header">
            <h1>🧠 AI NeuroOnco — Brain Tumor Classifier</h1>
            <p><em>Track 3 — AI Integration & Development</em></p>
            <p>Upload a brain MRI image to classify tumor type with
            Explainable AI (Grad-CAM) visualisation.</p>
            </div>
            """,
        )

        with gr.Row():
            # Left column — Inputs
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Brain MRI",
                    type="numpy",
                    height=300,
                )
                model_dropdown = gr.Dropdown(
                    choices=["efficientnet", "custom_cnn"],
                    value="efficientnet",
                    label="Select Model",
                )
                gradcam_toggle = gr.Checkbox(
                    value=True,
                    label="Show Grad-CAM Explanation",
                )
                threshold_slider = gr.Slider(
                    minimum=0.3,
                    maximum=0.95,
                    value=config.CONFIDENCE_THRESHOLD,
                    step=0.05,
                    label="Confidence Threshold (flag uncertain below this)",
                )
                predict_btn = gr.Button(
                    "🔍 Classify Tumor",
                    variant="primary",
                    size="lg",
                )

            # Right column — Outputs
            with gr.Column(scale=2):
                confidence_output = gr.Label(
                    label="Classification Confidence",
                    num_top_classes=config.NUM_CLASSES,
                )
                gradcam_output = gr.Image(
                    label="Grad-CAM Explanation",
                    height=350,
                )
                status_output = gr.Markdown(
                    label="Prediction Details",
                )

        # Wire up the prediction
        predict_btn.click(
            fn=predict,
            inputs=[image_input, model_dropdown, gradcam_toggle, threshold_slider],
            outputs=[confidence_output, gradcam_output, status_output],
        )

        # Also trigger on image upload
        image_input.change(
            fn=predict,
            inputs=[image_input, model_dropdown, gradcam_toggle, threshold_slider],
            outputs=[confidence_output, gradcam_output, status_output],
        )

        # Information section
        with gr.Accordion("ℹ️ About This System", open=False):
            gr.Markdown(
                f"""
                ### Model Information
                - **EfficientNet-B0**: Transfer learning from ImageNet,
                  fine-tuned for brain tumor classification
                - **Custom CNN**: Lightweight 5-block CNN (~2M parameters)
                  designed for resource-constrained environments

                ### Classes
                | Class | Description |
                |---|---|
                | Glioma | Most common primary brain tumor |
                | Meningioma | Tumor arising from the meninges |
                | Pituitary | Tumor of the pituitary gland |
                | No Tumor | Normal brain MRI (no tumor detected) |

                ### Explainability
                - **Grad-CAM** highlights the regions the model focuses on
                - Red/warm regions = high importance for the prediction
                - Blue/cool regions = low importance

                ### ⚠️ Disclaimer
                This is a research prototype. It is **NOT** a medical device
                and should **NOT** be used for clinical diagnosis. Always
                consult a qualified medical professional.

                ### Technical Details
                - Image size: {config.IMG_SIZE}×{config.IMG_SIZE}
                - Device: {config.DEVICE}
                - Confidence threshold: {config.CONFIDENCE_THRESHOLD:.0%}
                """
            )

        # Footer
        gr.Markdown(
            """
            <div class="footer">
            AI NeuroOnco Project — Track 3: AI Integration & Development<br>
            Built with PyTorch, Grad-CAM, and Gradio
            </div>
            """,
        )

    return demo


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n🧠 AI NeuroOnco — Brain Tumor Classifier")
    print(f"   Device: {config.DEVICE}")
    print(f"   Models: {', '.join(['efficientnet', 'custom_cnn'])}")
    print(f"   Port:   {config.DEMO_PORT}\n")

    demo = build_demo()
    demo.launch(
        server_port=config.DEMO_PORT,
        share=config.DEMO_SHARE,
        show_error=True,
    )
