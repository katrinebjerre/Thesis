# training/evaluate.py

import torch
from training.metrics import mean_angular_error_deg


@torch.no_grad()
def evaluate_model(model, criterion, loader, device):
    """Evaluate model and return average loss and angular error."""

    model.eval()

    total_loss = 0.0
    total_ang = 0.0
    total_batches = 0

    for images, targets in loader:
        # Move batch to device
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float()

        # Forward pass
        preds = model(images)

        # Compute loss and metric
        loss = criterion(preds, targets)
        ang = mean_angular_error_deg(preds, targets)

        # Accumulate batch results
        total_loss += loss.item()
        total_ang += ang
        total_batches += 1

    # Average over all batches
    avg_loss = total_loss / max(1, total_batches)
    avg_ang = total_ang / max(1, total_batches)

    return avg_loss, avg_ang