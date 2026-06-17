# experiments/common.py

from pathlib import Path
import time

import torch

from data_utils.splits import print_split_info
from training.dataloader import make_dataloader
from training.evaluate import evaluate_model
from training.losses import get_loss_function
from training.optimizers import get_optimizer
from training.train import train_one_model

# ============ Toggle between standard CNN and B-cos model ============

USE_BCOS = True

if USE_BCOS:
    from models.bcos_model import build_bcos_model as build_model
else:
    from models.cnn import build_model

# ====================================================================


# Set seed for reproducibility
torch.manual_seed(42)

# Enable faster GPU performance for fixed input sizes
torch.backends.cudnn.benchmark = True


# Select device (GPU / MPS / CPU)
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def split_test_into_val_and_test(test_df, val_ratio, random_state):
    """
    Shuffle test data and split into:
    - validation (for early stopping)
    - final test (for evaluation)
    """
    test_df = test_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    val_size = int(len(test_df) * val_ratio)

    val_df = test_df.iloc[:val_size].copy()
    final_test_df = test_df.iloc[val_size:].copy()

    return val_df, final_test_df


def make_loader(df, settings, shuffle):
    """Create DataLoader from dataframe."""
    return make_dataloader(
        df=df,
        batch_size=settings["batch_size"],
        num_workers=settings["num_workers"],
        shuffle=shuffle,
    )


def make_model(settings, pretrained_weights=None, transfer_mode="head_only"):
    """
    Build model and optionally load pretrained weights for transfer learning.
    If pretrained_weights is None, model is trained from scratch.
    """
    model = build_model(
        num_blocks=settings["num_blocks"],
        base_channel=settings["base_channel"],
        adaptive_pool_out=settings["adaptive_pool_out"],
        hidden_dim=settings["hidden_dim"],
        activation=settings["activation"],
    ).to(DEVICE)

    print("\n=== Model info ===")
    print(f"USE_BCOS: {USE_BCOS}")
    print(f"Builder: {build_model.__module__}.{build_model.__name__}")
    print(f"Model class: {model.__class__.__name__}")
    print("==================\n")

    # Scratch: do nothing, keep random initialization
    if pretrained_weights is None:
        return model

    # Load pretrained weights
    state_dict = torch.load(pretrained_weights, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)

    # =========================
    # Transfer learning for B-cos
    # =========================
    if USE_BCOS:

        # Last Two Conv Blocks:
        if transfer_mode == "last_two_conv":
            for param in model.feature_extractor.parameters():
                param.requires_grad = False

            layers = list(model.feature_extractor.children())

            # Unfreeze last two conv blocks
            for layer in layers[-4:]:
                for param in layer.parameters():
                    param.requires_grad = True

        else:
            raise ValueError(
                f"Unknown transfer_mode for B-cos: {transfer_mode}"
            )

        return model

    # =========================
    # Transfer learning for CNN
    # =========================

    # Full Fine-Tune: Train all layers
    if transfer_mode == "full_finetune":
        pass

    # Head Only: Freeze all conv blocks
    elif transfer_mode == "head_only":
        for param in model.feature_extractor.parameters():
            param.requires_grad = False

    # Last Conv Block:
    elif transfer_mode == "last_conv":
        for param in model.feature_extractor.parameters():
            param.requires_grad = False

        layers = list(model.feature_extractor.children())

        # Unfreeze last conv block
        for layer in layers[-3:]:
            for param in layer.parameters():
                param.requires_grad = True

    # Last Two Conv Blocks:
    elif transfer_mode == "last_two_conv":
        for param in model.feature_extractor.parameters():
            param.requires_grad = False

        layers = list(model.feature_extractor.children())

        # Unfreeze last two conv blocks
        for layer in layers[-6:]:
            for param in layer.parameters():
                param.requires_grad = True

    # Last Three Conv Blocks:
    elif transfer_mode == "last_three_conv":
        for param in model.feature_extractor.parameters():
            param.requires_grad = False

        layers = list(model.feature_extractor.children())

        # Unfreeze last three conv blocks
        for layer in layers[-9:]:
            for param in layer.parameters():
                param.requires_grad = True

    else:
        raise ValueError(f"Unknown transfer_mode for CNN: {transfer_mode}")

    return model


def make_model_path(model_name):
    """Create path for saving a model."""
    models_dir = Path("results/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    return models_dir / f"{model_name}.pth"


def train_and_evaluate(
    settings,
    model_name,
    train_df,
    test_df,
    row_info,
    pretrained_weights=None,
    transfer_mode="full_finetune",
):
    """
    Full pipeline for one model:
    - split test into val/test
    - create loaders
    - train model
    - evaluate model
    - save model
    - return results
    """

    model_path = make_model_path(model_name)

    # Split held-out data into val/test
    val_df, final_test_df = split_test_into_val_and_test(
        test_df=test_df,
        val_ratio=settings["val_ratio"],
        random_state=settings["split_random_state"],
    )

    # Print run info
    print(f"Using device: {DEVICE}")
    print_split_info("Train vs. Val/Test", train_df, test_df, "Train", "Val/Test")
    print("\n")

    # Create dataloaders
    train_loader = make_loader(train_df, settings, shuffle=True)
    val_loader = make_loader(val_df, settings, shuffle=False)
    test_loader = make_loader(final_test_df, settings, shuffle=False)

    # Build model
    model = make_model(
        settings,
        pretrained_weights=pretrained_weights,
        transfer_mode=transfer_mode,
    )

    # Loss function
    criterion = get_loss_function(settings["loss_function"])

    # Optimizer
    # Only include parameters that are not frozen (requires_grad=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(
        settings["optimizer"],
        trainable_params,
        settings["learning_rate"],
    )

    start_time = time.time()

    # Train model
    best_state, train_losses, val_losses, best_val_loss, best_epoch, stopped_early = train_one_model(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        DEVICE,
        settings["num_epochs"],
        settings["patience"],
    )

    # Load best model (early stopping)
    model.load_state_dict(best_state)

    # Evaluate on final test set
    test_loss, test_ang_error = evaluate_model(
        model,
        criterion,
        test_loader,
        DEVICE,
    )

    runtime_sec = time.time() - start_time
    
    # Print results
    print("\n")
    print(f"\nFinished training: {model_name}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test angular error: {test_ang_error:.4f}°")
    print("\n")
    print("\n")

    # Save model weights
    torch.save(model.state_dict(), model_path)

    # Summary row (one per model)
    result_row = {
        "model_name": model_name,
        "model_path": str(model_path),
        "pretrained_weights": str(pretrained_weights) if pretrained_weights else None,
        **row_info,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(final_test_df),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_ang_error": test_ang_error,
        "runtime_sec": runtime_sec,
        "stopped_early": stopped_early,
    }

    # Per-epoch history (for plotting/debugging)
    history_rows = []
    for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1):
        history_rows.append({
            "model_name": model_name,
            **row_info,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

    return result_row, history_rows