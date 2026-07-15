"""Post-hoc temperature calibration fitted on validation data only."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


@torch.no_grad()
def collect_logits(model, data_loader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect unscaled logits and labels without touching the test set."""
    model.eval()
    logits, labels = [], []
    for batch in data_loader:
        images, target = batch[0], batch[1]
        logits.append(model(images.to(device, non_blocking=True)).detach().cpu())
        labels.append(target.detach().cpu())
    if not logits:
        raise ValueError("Calibration loader is empty")
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(
    model,
    validation_loader,
    device: torch.device,
    max_iter: int = 50,
) -> float:
    """Fit a positive scalar temperature by minimizing validation NLL."""
    logits, labels = collect_logits(model, validation_loader, device)
    # Optimize the scalar on CPU for consistent LBFGS support across CUDA/MPS/CPU.
    calibration_device = torch.device("cpu")
    logits = logits.to(calibration_device)
    labels = labels.to(calibration_device)

    log_temperature = nn.Parameter(torch.zeros(1, device=calibration_device))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.05, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 10.0)
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(0.05, 10.0).item())
    return temperature
