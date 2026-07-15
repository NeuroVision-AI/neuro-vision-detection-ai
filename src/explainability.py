"""
Explainability module — Grad-CAM heatmap generation.

Generates visual explanations for model predictions using
Gradient-weighted Class Activation Mapping (Grad-CAM).
Supports research analysis of which image regions influence a model output.
It is not a lesion detector or clinical explanation.

References:
    - Paper #10 (Kaggle-based Explainable CNN)
    - Paper #57 (Interpretable Meningioma Grading with RCAM)
"""

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms

from src import config
from src.utils import denormalize


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Computes a heatmap highlighting the regions of an input image
    most important for a specific class prediction.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Args:
            model: Trained classification model.
            target_layer: The convolutional layer to compute Grad-CAM for
                          (typically the last conv layer).
        """
        self.model = model
        self.target_layer = target_layer

        # Storage for hooks
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for a single image.

        Args:
            input_tensor: Preprocessed image tensor (1, C, H, W).
            target_class: Class index to generate heatmap for.
                          If None, uses the predicted class.

        Returns:
            Heatmap as a numpy array of shape (H, W) with values in [0, 1].
        """
        self.model.eval()
        input_tensor = input_tensor.to(config.DEVICE)

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for the target class
        target_score = output[0, target_class]
        target_score.backward()

        # Get gradients and activations
        gradients = self.gradients[0]    # (C, h, w)
        activations = self.activations[0]  # (C, h, w)

        # Global average pooling of gradients → weights
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted sum of activations
        heatmap = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        # ReLU (keep only positive influence)
        heatmap = F.relu(heatmap)

        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Resize to input image size
        heatmap = heatmap.cpu().numpy()
        heatmap = np.uint8(255 * heatmap)

        # Resize heatmap to match input image
        heatmap_pil = Image.fromarray(heatmap).resize(
            (config.IMG_SIZE, config.IMG_SIZE), Image.BILINEAR
        )
        heatmap = np.array(heatmap_pil).astype(np.float32) / 255.0

        return heatmap


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = config.GRADCAM_ALPHA,
    colormap: str = config.GRADCAM_COLORMAP,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        image: Original image as numpy array (H, W, 3) in [0, 1].
        heatmap: Heatmap as numpy array (H, W) in [0, 1].
        alpha: Transparency of the heatmap overlay.
        colormap: Matplotlib colormap name.

    Returns:
        Overlay image as numpy array (H, W, 3) in [0, 1].
    """
    # Apply colormap to heatmap
    try:
        import matplotlib
        if hasattr(matplotlib, "colormaps"):
            cmap = matplotlib.colormaps[colormap]
        else:
            cmap = cm.get_cmap(colormap)
    except Exception:
        cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Drop alpha channel

    # Blend
    overlay = (1 - alpha) * image + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)

    return overlay


def generate_explanation(
    model: torch.nn.Module,
    image_path: str,
    target_class: Optional[int] = None,
    save_path: Optional[Path] = None,
) -> dict:
    """
    Generate a complete Grad-CAM explanation for a single image.

    Args:
        model: Trained model with a `get_gradcam_target_layer()` method.
        image_path: Path to the input MRI image.
        target_class: Class to explain. If None, uses predicted class.
        save_path: Optional path to save the explanation figure.

    Returns:
        Dictionary with prediction, confidence, heatmap, and overlay.
    """
    # Get target layer
    if hasattr(model, "get_gradcam_target_layer"):
        target_layer = model.get_gradcam_target_layer()
    else:
        raise ValueError(
            "Model must implement get_gradcam_target_layer() method."
        )

    # Load and preprocess image
    original_image = Image.open(image_path).convert("RGB")
    original_np = np.array(original_image.resize(
        (config.IMG_SIZE, config.IMG_SIZE)
    )).astype(np.float32) / 255.0

    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.PIXEL_MEAN, std=config.PIXEL_STD),
    ])
    input_tensor = transform(original_image).unsqueeze(0)

    # Get prediction
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor.to(config.DEVICE))
        probs = F.softmax(logits, dim=1)[0]
        predicted_class = probs.argmax().item()
        confidence = probs[predicted_class].item()

    # Generate Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    explain_class = target_class if target_class is not None else predicted_class
    heatmap = grad_cam.generate(input_tensor, target_class=explain_class)

    # Create overlay
    overlay = overlay_heatmap(original_np, heatmap)

    # Build result
    result = {
        "predicted_class": config.IDX_TO_CLASS[predicted_class],
        "predicted_idx": predicted_class,
        "confidence": confidence,
        "probabilities": {
            config.IDX_TO_CLASS[i]: probs[i].item()
            for i in range(config.NUM_CLASSES)
        },
        "explained_class": config.IDX_TO_CLASS[explain_class],
        "heatmap": heatmap,
        "overlay": overlay,
        "original": original_np,
        "is_uncertain": confidence < config.CONFIDENCE_THRESHOLD,
    }

    # Save visualization
    if save_path:
        _save_explanation_figure(result, save_path)

    return result


def _save_explanation_figure(result: dict, save_path: Path) -> None:
    """
    Create and save a publication-quality explanation figure.

    Layout: [Original | Heatmap | Overlay] with prediction info.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(result["original"])
    axes[0].set_title("Original MRI", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    # Heatmap
    axes[1].imshow(result["heatmap"], cmap=config.GRADCAM_COLORMAP)
    axes[1].set_title(
        f"Grad-CAM: {result['explained_class']}",
        fontsize=13, fontweight="bold"
    )
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(result["overlay"])
    axes[2].set_title("Overlay", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    # Prediction banner
    pred = result["predicted_class"]
    conf = result["confidence"]
    uncertain = " ⚠️ UNCERTAIN" if result["is_uncertain"] else ""

    fig.suptitle(
        f"Prediction: {pred.upper()} ({conf:.1%} confidence){uncertain}",
        fontsize=15, fontweight="bold",
        color="#E74C3C" if result["is_uncertain"] else "#2ECC71",
        y=1.02,
    )

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def batch_generate_explanations(
    model: torch.nn.Module,
    image_dir: Path,
    output_dir: Path = config.HEATMAP_DIR,
    max_images: int = 20,
) -> List[dict]:
    """
    Generate Grad-CAM explanations for a batch of images.

    Args:
        model: Trained model.
        image_dir: Directory containing images.
        output_dir: Directory to save explanation figures.
        max_images: Maximum number of images to process.

    Returns:
        List of result dictionaries.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted([
        p for p in Path(image_dir).rglob("*")
        if p.suffix.lower() in valid_extensions
    ])[:max_images]

    results = []
    for img_path in image_paths:
        save_name = f"gradcam_{img_path.stem}.png"
        save_path = output_dir / save_name

        try:
            result = generate_explanation(
                model=model,
                image_path=str(img_path),
                save_path=save_path,
            )
            result["image_path"] = str(img_path)
            results.append(result)
            print(f"  ✓ {img_path.name} → {result['predicted_class']} "
                  f"({result['confidence']:.1%})")
        except Exception as e:
            print(f"  ✗ {img_path.name} → Error: {e}")

    print(f"\nGenerated {len(results)} explanations → {output_dir}")
    return results
