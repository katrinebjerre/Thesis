# data_utils/dataset.py

import pandas as pd
from torch.utils.data import Dataset
import torch


def load_metadata(csv_path):
    """Load CSV metadata and zero-pad subject IDs."""
    df = pd.read_csv(csv_path)
    df["subject"] = df["subject"].astype(str).str.zfill(3)
    return df


class GazeDataset(Dataset):
    """PyTorch dataset returning (image, (x, y) pixel target)."""

    def __init__(self, df):
        """Initialize paths and targets from DataFrame."""
        self.paths = df["path"].tolist()
        self.targets = torch.tensor(
            df[["x_pixel", "y_pixel"]].to_numpy(),
            dtype=torch.float32
        )

    def __len__(self):
        """Return number of samples."""
        return len(self.paths)

    def __getitem__(self, idx):
        """Load and return (image, target) at index."""
        image = torch.load(self.paths[idx], weights_only=False)
        target = self.targets[idx]
        return image, target