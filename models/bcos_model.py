# models/bcos_model.py

import torch.nn as nn

from bcos.modules import BcosConv2d, BcosLinear

BCOS_B = 2.0
BCOS_MAX_OUT = 1
USE_BCOS_LINEAR = False


def make_channel_list(num_blocks: int, start_channels: int) -> list[int]:
    """Return channel sizes per block (doubling each step)."""
    channel_list = []
    channels = start_channels

    for _ in range(num_blocks):
        channel_list.append(channels)
        channels *= 2

    return channel_list


def make_bcos_blocks(
    channel_list: list[int],
    activation_name: str,
    pool_output_size: tuple[int, int],
    b: float = BCOS_B,
    max_out: int = BCOS_MAX_OUT,
) -> nn.Sequential:
    """Build B-cos feature extractor (BcosConv2d + pooling, adaptive pooled output)."""
    layers = []

    # Input images are grayscale
    input_channels = 1

    # Activation unused (kept for config compatibility)
    _ = activation_name

    for block_index, output_channels in enumerate(channel_list):
        layers.append(
            BcosConv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                padding=1,
                b=b,
                max_out=max_out,
            )
        )

        # The output of this block becomes input to the next block
        input_channels = output_channels

        # Add max pooling after every block except the last one
        if block_index < len(channel_list) - 1:
            layers.append(nn.MaxPool2d(2, 2))

    # Force a fixed output size before the linear layers
    layers.append(nn.AdaptiveAvgPool2d(pool_output_size))

    return nn.Sequential(*layers)


def make_bcos_regression_head(
    last_num_channels: int,
    pool_output_size: tuple[int, int],
    hidden_size: int,
    activation_name: str,
    use_bcos_linear: bool = USE_BCOS_LINEAR,
    b: float = BCOS_B,
) -> nn.Sequential:
    """Build regression head (optionally using BcosLinear if use_bcos_linear=True)."""
    pooled_height, pooled_width = pool_output_size
    num_input_features = last_num_channels * pooled_height * pooled_width

    _ = activation_name
    
    first_layer = (
        BcosLinear(num_input_features, hidden_size, b=b)
        if use_bcos_linear
        else nn.Linear(num_input_features, hidden_size)
    )

    return nn.Sequential(
        nn.Flatten(),
        first_layer,
        nn.Linear(hidden_size, 2),
    )


class BcosRegressor(nn.Module):
    """B-cos regressor predicting (x, y) coordinates."""

    def __init__(
        self,
        channel_list: list[int],
        activation_name: str,
        pool_output_size: tuple[int, int],
        hidden_size: int,
        b: float = BCOS_B,
        max_out: int = BCOS_MAX_OUT,
        use_bcos_linear: bool = USE_BCOS_LINEAR,
    ):
        """Initialize feature extractor and regression head."""
        super().__init__()

        self.feature_extractor = make_bcos_blocks(
            channel_list=channel_list,
            activation_name=activation_name,
            pool_output_size=pool_output_size,
            b=b,
            max_out=max_out,
        )

        self.regression_head = make_bcos_regression_head(
            last_num_channels=channel_list[-1],
            pool_output_size=pool_output_size,
            hidden_size=hidden_size,
            activation_name=activation_name,
            use_bcos_linear=use_bcos_linear,
            b=b,
        )

    def forward(self, x):
        """Run input through feature extractor and regression head."""
        x = self.feature_extractor(x)
        x = self.regression_head(x)
        return x


def build_bcos_model(
    num_blocks: int,
    base_channel: int,
    adaptive_pool_out: tuple[int, int],
    hidden_dim: int,
    activation: str,
    b: float = BCOS_B,
    max_out: int = BCOS_MAX_OUT,
    use_bcos_linear: bool = USE_BCOS_LINEAR,
) -> BcosRegressor:
    """Construct full B-cos regression model."""

    channel_list = make_channel_list(
        num_blocks=num_blocks,
        start_channels=base_channel,
    )

    model = BcosRegressor(
        channel_list=channel_list,
        activation_name=activation,
        pool_output_size=adaptive_pool_out,
        hidden_size=hidden_dim,
        b=b,
        max_out=max_out,
        use_bcos_linear=use_bcos_linear,
    )

    return model