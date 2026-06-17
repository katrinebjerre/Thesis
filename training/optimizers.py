# training/optimizers

import torch.optim as optim


def get_optimizer(name, model_params, lr):
    """Return optimizer instance from name."""
    name = name.lower()

    if name == "adam":
        return optim.Adam(model_params, lr=lr)
    if name == "sgd":
        return optim.SGD(model_params, lr=lr)

    raise ValueError(f"Unknown optimizer: {name}")