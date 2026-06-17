# experiments/gen_loto_bcos.py

from pathlib import Path

import pandas as pd

from data_utils.splits import get_general_pool, make_general_loto_split
from experiments.common import train_and_evaluate
from experiments.config import get_base_settings


def main():

    # Settings for this experiment
    SETTINGS = {
        **get_base_settings(),
        "results_dir": "results/generalized_bcos",
        "results_file": "gen_loto_bcos.csv",
        "history_file": "gen_loto_bcos_history.csv",
    }

    # Get all targets from general dataset
    targets = sorted(get_general_pool(SETTINGS["csv_path"])["target"].unique())

    all_results = []
    all_history = []

    # Loop over targets (Leave-One-Target-Out)
    for target in targets:
        print(f"Running gen_loto_bcos | held_out_target={target}")

        # Create model name
        model_name = f"gen_loto_bcos_{target}"

        # Create train/test split
        train_df, test_df = make_general_loto_split(
            target=target,
            csv_path=SETTINGS["csv_path"],
        )

        # Metadata for CSV
        row_info = {
            "experiment_type": "general_loto_bcos",
            "subject_spec_id": None,
            "held_out_subject": None,
            "held_out_target": target,
            "init_mode": "scratch",
        }

        # Run training pipeline
        result_row, history_rows = train_and_evaluate(
            settings=SETTINGS,
            model_name=model_name,
            train_df=train_df,
            test_df=test_df,
            row_info=row_info,
            pretrained_weights=None,
        )

        all_results.append(result_row)
        all_history.extend(history_rows)

    # Save results
    results_dir = Path(SETTINGS["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(all_results).to_csv(results_dir / SETTINGS["results_file"], index=False)
    pd.DataFrame(all_history).to_csv(results_dir / SETTINGS["history_file"], index=False)

    print("Finished gen_loto_bcos")


if __name__ == "__main__":
    main()