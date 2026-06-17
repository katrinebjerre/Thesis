# experiments/grid_search.py

from itertools import product
from pathlib import Path
import json
import time

import pandas as pd
import torch
from tqdm import tqdm

from data_utils.splits import print_split_info
from data_utils.dataset import load_metadata
from models.cnn import build_model
from training.dataloader import make_dataloader
from training.losses import get_loss_function
from training.optimizers import get_optimizer
from training.train import train_one_model
from training.evaluate import evaluate_model

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True

# ---------------------------
# Setup
# ---------------------------

ROOT_DIR = "."
CSV_PATH = "data_utils/metadata_tensors.csv"

RESULTS_DIR = Path("results/grid_search")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Targets reserved for validation/test
TEST_TARGETS = [0, 4, 12, 20, 24]

VAL_RATIO = 0.4
SPLIT_RANDOM_STATE = 42

# Data settings
IMAGE_SIZE = (96, 128)
BATCH_SIZE = 128
NUM_WORKERS = 0

# Training settings
NUM_EPOCHS = 30
PATIENCE = 4

# Device selection
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ---------------------------
# Hyperparameter grid
# ---------------------------

num_blocks_options = [2, 3, 4]
base_channel_options = [16, 32]
adaptive_pool_options = [(1, 1), (3, 4)]
hidden_dim_options = [32, 64, 128]

activations = ["relu", "leakyrelu"]
loss_functions = ["mse", "angular_error", "angular_error_squared"]
optimizers_list = ["adam", "sgd"]
learning_rates = [0.001, 0.01]


# ---------------------------
# Helpers
# ---------------------------

def create_loader(df, shuffle):
    """Create dataloader with predefined settings."""
    return make_dataloader(
        df=df,
        root_dir=ROOT_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=shuffle,
    )


def print_config_header(run_idx, total_runs, config):
    """Print header for a run configuration."""
    print("\n" + "=" * 80)
    print(f"Run {run_idx}/{total_runs}")
    print(
        f"blocks={config['num_blocks']} | "
        f"ch={config['base_channel']} | "
        f"pool={config['adaptive_pool_out']} | "
        f"h={config['hidden_dim']} | "
        f"act={config['activation']} | "
        f"loss={config['loss_function']} | "
        f"opt={config['optimizer']} | "
        f"lr={config['learning_rate']}"
    )


def print_run_summary(stopped_early, avg_test_ang, runtime, is_new_best):
    """Print summary of run results."""
    status = "EARLY STOPPED" if stopped_early else "FINISHED"
    print(f"{status} | TEST ANG={avg_test_ang:.4f}° | {runtime:.1f}s")

    if is_new_best:
        print(f"NEW BEST -> {avg_test_ang:.4f}°")


# ---------------------------
# Data split
# ---------------------------

df = load_metadata(CSV_PATH)

print(f"Using device: {DEVICE}")
print(f"Total samples: {len(df)}")
print(f"Targets: {sorted(df['target'].unique().tolist())}")

# Split into train and reserved targets
train_df_full = df[~df["target"].isin(TEST_TARGETS)].copy()
reserved_df_full = df[df["target"].isin(TEST_TARGETS)].copy()

# Shuffle reserved and split into val/test
reserved_df_full = reserved_df_full.sample(
    frac=1.0,
    random_state=SPLIT_RANDOM_STATE,
).reset_index(drop=True)

val_size = int(len(reserved_df_full) * VAL_RATIO)
val_df_full = reserved_df_full.iloc[:val_size].copy()
test_df_full = reserved_df_full.iloc[val_size:].copy()

print_split_info("Train / Reserved", train_df_full, reserved_df_full, "Train", "Reserved")
print_split_info("Val / Test", val_df_full, test_df_full, "Val", "Test")


# ---------------------------
# Grid search
# ---------------------------

all_results = []
best_test_ang_overall = float("inf")
best_config = None

# Create all combinations
grid = list(product(
    num_blocks_options,
    base_channel_options,
    adaptive_pool_options,
    hidden_dim_options,
    activations,
    loss_functions,
    optimizers_list,
    learning_rates,
))

print(f"\nConfigs: {len(grid)}")

total_start_time = time.time()

#print("Creating train loader...")
train_loader = create_loader(train_df_full, True)
#print("Train loader created.")

#print("Creating val loader...")
val_loader = create_loader(val_df_full, False)
#print("Val loader created.")

#print("Creating test loader...")
test_loader = create_loader(test_df_full, False)
#print("Test loader created.")


for run_idx, (
    num_blocks,
    base_channel,
    adaptive_pool_out,
    hidden_dim,
    activation,
    loss_name,
    optimizer_name,
    learning_rate,
) in enumerate(tqdm(grid, desc="Grid Search", unit="config"), start=1):

    config_start_time = time.time()

    # Store config
    config = {
        "num_blocks": num_blocks,
        "base_channel": base_channel,
        "adaptive_pool_out": adaptive_pool_out,
        "hidden_dim": hidden_dim,
        "activation": activation,
        "loss_function": loss_name,
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
    }

    print_config_header(run_idx, len(grid), config)

    # Build model
    model = build_model(
        num_blocks=num_blocks,
        base_channel=base_channel,
        adaptive_pool_out=adaptive_pool_out,
        hidden_dim=hidden_dim,
        activation=activation,
    ).to(DEVICE)

    # Loss and optimizer
    criterion = get_loss_function(loss_name)
    optimizer = get_optimizer(
        name=optimizer_name,
        model_params=model.parameters(),
        lr=learning_rate,
    )

    # Train model
    best_state, train_losses, val_losses, best_val_loss, best_epoch, stopped_early = train_one_model(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        DEVICE,
        NUM_EPOCHS,
        PATIENCE,
    )

    model.load_state_dict(best_state)

    # Evaluate on test set
    avg_test_loss, avg_test_ang = evaluate_model(
        model=model,
        criterion=criterion,
        loader=test_loader,
        device=DEVICE,
    )

    runtime = time.time() - config_start_time

    # Store results
    result_row = {
        **config,
        "adaptive_pool_out": str(adaptive_pool_out),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": avg_test_loss,
        "mean_test_angular_error": avg_test_ang,
        "runtime_sec": round(runtime, 2),
        "stopped_early": stopped_early,
    }

    all_results.append(result_row)

    # Track best config
    is_new_best = avg_test_ang < best_test_ang_overall
    if is_new_best:
        best_test_ang_overall = avg_test_ang
        best_config = config

        # Save best model
        best_model_path = RESULTS_DIR / "grid_search_best.pth"
        torch.save(model.state_dict(), best_model_path)

    print_run_summary(stopped_early, avg_test_ang, runtime, is_new_best)


# ---------------------------
# Save results
# ---------------------------

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values("mean_test_angular_error").reset_index(drop=True)

results_csv_path = RESULTS_DIR / "grid_search_results.csv"
results_df.to_csv(results_csv_path, index=False)

best_config_payload = {
    "targets": TEST_TARGETS,
    "val_ratio": VAL_RATIO,
    "best_ang": best_test_ang_overall,
    "best_config": {
        **best_config,
        "adaptive_pool_out": list(best_config["adaptive_pool_out"]),
    },
}

best_config_json_path = RESULTS_DIR / "best_config.json"
with open(best_config_json_path, "w") as f:
    json.dump(best_config_payload, f, indent=2)

total_runtime = time.time() - total_start_time

print("\n" + "=" * 80)
print("Finished grid search")
print(f"Saved: {results_csv_path}")
print(f"Best ang: {best_test_ang_overall:.6f}°")
print(best_config)