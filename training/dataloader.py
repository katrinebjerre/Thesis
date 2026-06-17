# training/dataloader.py

import torch
from torch.utils.data import DataLoader

from data_utils.dataset import GazeDataset


def make_dataloader(df, batch_size, num_workers, shuffle=False):
    """Create DataLoader from dataframe using GazeDataset."""

    ds = GazeDataset(df)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # Enable faster GPU transfer if CUDA is available
        pin_memory=torch.cuda.is_available(),
        # Keep workers alive between epochs (if using multiprocessing)
        persistent_workers=(num_workers > 0),
    )