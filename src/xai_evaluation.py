"""Quantitative checks for Grad-CAM stability, sanity, and localization."""

from __future__ import annotations

from typing import Optional

import numpy as np


def _as_2d(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=float).squeeze()
    if value.ndim != 2:
        raise ValueError("Expected a 2D heatmap or mask")
    if not np.all(np.isfinite(value)):
        raise ValueError("Heatmap contains non-finite values")
    return value


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman-style correlation used for repeatability and sanity checks."""
    left = _as_2d(left).ravel()
    right = _as_2d(right).ravel()
    if left.shape != right.shape:
        raise ValueError("Heatmaps must have the same shape")
    left_rank = np.argsort(np.argsort(left)).astype(float)
    right_rank = np.argsort(np.argsort(right)).astype(float)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def threshold_heatmap(heatmap: np.ndarray, quantile: float = 0.8) -> np.ndarray:
    """Convert a heatmap to its most salient region."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1")
    heatmap = _as_2d(heatmap)
    return heatmap >= np.quantile(heatmap, quantile)


def localization_iou(heatmap: np.ndarray, mask: np.ndarray, quantile: float = 0.8) -> float:
    """Intersection-over-union between salient heatmap pixels and a reference mask."""
    salient = threshold_heatmap(heatmap, quantile)
    reference = _as_2d(mask) > 0
    if salient.shape != reference.shape:
        raise ValueError("Heatmap and mask must have the same shape")
    union = np.logical_or(salient, reference).sum()
    return float(np.logical_and(salient, reference).sum() / union) if union else 0.0


def pointing_game(heatmap: np.ndarray, mask: np.ndarray) -> float:
    """Return 1 when the maximum-saliency pixel falls inside the reference mask."""
    heatmap = _as_2d(heatmap)
    reference = _as_2d(mask) > 0
    if heatmap.shape != reference.shape:
        raise ValueError("Heatmap and mask must have the same shape")
    maximum = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return float(reference[maximum])


def evaluate_explanation(
    original: np.ndarray,
    repeat: Optional[np.ndarray] = None,
    randomized: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    quantile: float = 0.8,
) -> dict:
    """Evaluate one explanation with only the evidence arrays that are available."""
    result = {}
    if repeat is not None:
        result["repeatability_rank_correlation"] = rank_correlation(original, repeat)
    if randomized is not None:
        correlation = rank_correlation(original, randomized)
        result["randomization_rank_correlation"] = correlation
        result["randomization_sensitivity"] = float(1.0 - abs(correlation))
    if mask is not None:
        result["localization_iou"] = localization_iou(original, mask, quantile=quantile)
        result["pointing_game"] = pointing_game(original, mask)
    return result
