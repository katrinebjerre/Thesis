# training/metrics.py

from training.losses import angular_error
import torch


@torch.no_grad()
def mean_angular_error_deg(preds, targets):
    """Return mean angular error in degrees."""
    return angular_error(preds, targets).mean().item()