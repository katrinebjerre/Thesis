# models/cnn.py

import torch.nn as nn

from training.activations import get_activation


def make_channel_list(num_blocks: int, start_channels: int) -> list[int]:
    """Return channel sizes per block (doubling each step)."""
    channel_list = []
    channels = start_channels

    for _ in range(num_blocks):
        channel_list.append(channels)
        channels *= 2

    return channel_list


def make_cnn_blocks(
    channel_list: list[int],
    activation_name: str,
    pool_output_size: tuple[int, int],
) -> nn.Sequential:
    """Build feature extractor (Conv2d + activation + pooling, adaptive pooled output)."""
    layers = []

    # Input images are grayscale
    input_channels = 1

    # Activation function
    activation_class = get_activation(activation_name)

    for block_index, output_channels in enumerate(channel_list):
        layers.append(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                padding=1,
            )
        )

        # Activation after convolution
        layers.append(activation_class())

        # The output of this block becomes input to the next block
        input_channels = output_channels

        # Add max pooling after every block except the last one
        if block_index < len(channel_list) - 1:
            layers.append(nn.MaxPool2d(2, 2))

    # Force a fixed output size before the linear layers
    layers.append(nn.AdaptiveAvgPool2d(pool_output_size))

    return nn.Sequential(*layers)


def make_regression_head(
    last_num_channels: int,
    pool_output_size: tuple[int, int],
    hidden_size: int,
    activation_name: str,
) -> nn.Sequential:
    """Build regression head."""
    pooled_height, pooled_width = pool_output_size
    num_input_features = last_num_channels * pooled_height * pooled_width

    activation_class = get_activation(activation_name)

    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_input_features, hidden_size),
        activation_class(),
        nn.Linear(hidden_size, 2),
    )


class CNNRegressor(nn.Module):
    """CNN regressor predicting (x, y) coordinates."""

    def __init__(
        self,
        channel_list: list[int],
        activation_name: str,
        pool_output_size: tuple[int, int],
        hidden_size: int,
    ):
        """Initialize feature extractor and regression head."""
        super().__init__()

        self.feature_extractor = make_cnn_blocks(
            channel_list=channel_list,
            activation_name=activation_name,
            pool_output_size=pool_output_size,
        )

        self.regression_head = make_regression_head(
            last_num_channels=channel_list[-1],
            pool_output_size=pool_output_size,
            hidden_size=hidden_size,
            activation_name=activation_name,
        )

    def forward(self, x):
        """Run input through feature extractor and regression head."""
        x = self.feature_extractor(x)
        x = self.regression_head(x)
        return x


def build_model(
    num_blocks: int,
    base_channel: int,
    adaptive_pool_out: tuple[int, int],
    hidden_dim: int,
    activation: str,
) -> CNNRegressor:
    """Construct full CNN regression model."""

    channel_list = make_channel_list(
        num_blocks=num_blocks,
        start_channels=base_channel,
    )

    model = CNNRegressor(
        channel_list=channel_list,
        activation_name=activation,
        pool_output_size=adaptive_pool_out,
        hidden_size=hidden_dim,
    )

    return model