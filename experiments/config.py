# experiments/config.py

from pathlib import Path
import json


BASE_SETTINGS = {
    "csv_path": "data_utils/metadata_tensors.csv",
    "batch_size": 128,
    "num_workers": 4,
    "num_epochs": 40,
    "patience": 6,
    "val_ratio": 0.4,
    "split_random_state": 42,
}


def get_base_settings(json_path="results/grid_search/best_config.json"):
    """Load the best configuration found during grid search."""
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Best config file not found: {json_path}")

    print(f"\nLoading config from: {json_path}")

    with open(json_path, "r") as f:
        payload = json.load(f)

    best_config = payload["best_config"].copy()

    if "adaptive_pool_out" in best_config:
        best_config["adaptive_pool_out"] = tuple(best_config["adaptive_pool_out"])

    config = {
        **BASE_SETTINGS,
        **best_config,
    }

    print("\nUsing config:")
    for k, v in config.items():
        print(f"{k:<20}: {v}")

    print("\n")

    return config