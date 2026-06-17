# experiments/subject_spec_scratch.py

from pathlib import Path

import pandas as pd

from data_utils.dataset import load_metadata
from data_utils.splits import make_person_loto_split
from data_utils.subject_groups import get_person_specific_subjects
from experiments.common import train_and_evaluate
from experiments.config import get_base_settings


def main():

    # Settings for this experiment
    SETTINGS = {
        **get_base_settings(),
        "results_dir": "results/subject_spec",
        "results_file": "subject_spec_scratch.csv",
        "history_file": "subject_spec_scratch_history.csv",
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
            print(f"Running subject_spec_scratch | subject={subject} | held_out_target={target}")

            subject_str = str(subject).zfill(3)

            # Create model name
            model_name = f"subject_spec_scratch_{subject_str}_{target}"

            # Create train/test split (single subject)
            train_df, test_df = make_person_loto_split(
                subject_id=subject_str,
                target=target,
                csv_path=SETTINGS["csv_path"],
            )

            # Metadata for CSV
            row_info = {
                "experiment_type": "subject_spec_scratch",
                "subject_spec_id": subject_str,
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

    print("Finished subject_spec_scratch")


if __name__ == "__main__":
    main()