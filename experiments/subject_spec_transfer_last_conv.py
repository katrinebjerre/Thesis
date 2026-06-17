# experiments/subject_spec_transfer_last_conv.py

from pathlib import Path

import pandas as pd

from data_utils.dataset import load_metadata
from data_utils.splits import make_person_loto_split
from data_utils.subject_groups import get_person_specific_subjects
from experiments.common import train_and_evaluate
from experiments.config import get_base_settings


# Transfer learning modes:
# - head_only: train head only (freeze all conv blocks)
# - last_conv: train last conv block + head
# - last_two_conv: train last 2 conv blocks + head
# - last_three_conv: train last 3 conv blocks + head
# - full_finetune: train entire model

def main():

    # Settings for this experiment
    TRANSFER_MODE = "last_conv"
    SETTINGS = {
        **get_base_settings(),
        "results_dir": "results/subject_spec",
        "results_file": f"subject_spec_transfer_{TRANSFER_MODE}.csv",
        "history_file": f"subject_spec_transfer_{TRANSFER_MODE}_history.csv",
    }

    # Get reserved subjects and all targets
    subjects = get_person_specific_subjects()
    df = load_metadata(SETTINGS["csv_path"])
    df = df[df["subject"].isin(subjects)]
    targets = sorted(df["target"].unique())

    all_results = []
    all_history = []

    # Loop over subjects and targets
    for subject in subjects:
        for target in targets:
            print(
                f"Running subject_spec_transfer_{TRANSFER_MODE} | "
                f"subject={subject} | held_out_target={target}"
            )

            subject_str = str(subject).zfill(3)

            # Create model name
            model_name = f"subject_spec_transfer_{TRANSFER_MODE}_{subject_str}_{target}"

            # Create train/test split (single subject)
            train_df, test_df = make_person_loto_split(
                subject_id=subject_str,
                target=target,
                csv_path=SETTINGS["csv_path"],
            )

            # Load pretrained generalized model (same target removed)
            pretrained_weights = Path("results/models") / f"gen_loto_{target}.pth"
            if not pretrained_weights.exists():
                raise FileNotFoundError(f"Missing {pretrained_weights}")

            # Metadata for CSV
            row_info = {
                "experiment_type": "subject_spec_transfer",
                "subject_spec_id": subject_str,
                "held_out_subject": None,
                "held_out_target": target,
                "init_mode": "transfer",
                "transfer_mode": TRANSFER_MODE,
            }

            # Run training pipeline
            result_row, history_rows = train_and_evaluate(
                settings=SETTINGS,
                model_name=model_name,
                train_df=train_df,
                test_df=test_df,
                row_info=row_info,
                pretrained_weights=pretrained_weights,
                transfer_mode=TRANSFER_MODE,
            )

            all_results.append(result_row)
            all_history.extend(history_rows)

    # Save results
    results_dir = Path(SETTINGS["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(all_results).to_csv(results_dir / SETTINGS["results_file"], index=False)
    pd.DataFrame(all_history).to_csv(results_dir / SETTINGS["history_file"], index=False)

    print(f"Finished subject_spec_transfer_{TRANSFER_MODE}")


if __name__ == "__main__":
    main()