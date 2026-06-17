# training/losses.py

import math
import torch
import torch.nn as nn


# Coordinate space used for gaze targets during the experiment
COORD_WIDTH_PX = 1440
COORD_HEIGHT_PX = 900

# Physical monitor setup
MONITOR_DIAGONAL_CM = 68.6
NATIVE_WIDTH_PX = 1920
NATIVE_HEIGHT_PX = 1080
VIEWING_DISTANCE_CM = 76.0

# Convert monitor diagonal from pixels to cm scale
diag_px = math.sqrt(NATIVE_WIDTH_PX**2 + NATIVE_HEIGHT_PX**2)  # diagonal in pixels ≈ 2202.9
cm_per_pixel = MONITOR_DIAGONAL_CM / diag_px                   # ≈ 0.0311 cm per pixel
SCREEN_WIDTH_CM = NATIVE_WIDTH_PX * cm_per_pixel               # ≈ 59.8 cm
SCREEN_HEIGHT_CM = NATIVE_HEIGHT_PX * cm_per_pixel             # ≈ 33.6 cm


def angular_error(preds, targets):
    """Computes angular error (in degrees) for each sample."""
    preds = preds.float()
    targets = targets.float()

    pred_x_px = preds[:, 0]
    pred_y_px = preds[:, 1]

    target_x_px = targets[:, 0]
    target_y_px = targets[:, 1]

    # Pixel difference in stimulus coordinate space
    dx_px = pred_x_px - target_x_px
    dy_px = pred_y_px - target_y_px

    # Convert pixel error to cm using pixel-to-cm ratio
    dx_cm = dx_px * (SCREEN_WIDTH_CM / COORD_WIDTH_PX)
    dy_cm = dy_px * (SCREEN_HEIGHT_CM / COORD_HEIGHT_PX)

    dist_cm = torch.sqrt(dx_cm**2 + dy_cm**2 + 1e-8)

    angle_rad = torch.atan(dist_cm / VIEWING_DISTANCE_CM)
    angle_deg = angle_rad * (180.0 / math.pi)

    return angle_deg


class AngularErrorLoss(nn.Module):
    """Mean angular error loss over a batch."""

    def forward(self, preds, targets):
        """Return mean angular error in degrees."""
        return angular_error(preds, targets).mean()


class AngularErrorSquaredLoss(nn.Module):
    """Mean squared angular error loss over a batch."""

    def forward(self, preds, targets):
        """Return mean squared angular error."""
        angle = angular_error(preds, targets)
        return (angle ** 2).mean()


def get_loss_function(name):
    """Return loss function from name."""
    name = name.lower()

    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    if name == "angular_error":
        return AngularErrorLoss()
    if name == "angular_error_squared":
        return AngularErrorSquaredLoss()

    raise ValueError(f"Unknown loss function: {name}")