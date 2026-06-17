# training/activations.py

import torch.nn as nn


def get_activation(name: str):
    """Return activation class from name (e.g. 'relu')."""
    name = name.lower()

    if name == "relu":
        return nn.ReLU
    if name == "leakyrelu":
        return nn.LeakyReLU

    raise ValueError(f"Unknown activation: {name}")