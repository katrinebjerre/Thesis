# experiments/gen_loto_reserved_eval.py

from pathlib import Path

import pandas as pd
import torch

from data_utils.splits import get_general_pool, make_reserved_subject_target_split
from data_utils.subject_groups import get_person_specific_subjects
from experiments.common import DEVICE, make_loader, make_model, make_model_path
from experiments.config import get_base_settings
from training.evaluate import evaluate_model
from training.losses import get_loss_function


def evaluate_existing_model(settings, model_name, test_df):
    """Load an already trained model and evaluate it on test_df."""
    model_path = make_model_path(model_name)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Build model
    model = make_model(settings, pretrained_weights=None)

    # Load saved weights
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Make loader
    test_loader = make_loader(test_df, settings, shuffle=False)

    # Loss
    criterion = get_loss_function(settings["loss_function"])

    # Evaluate
    test_loss, test_ang_error = evaluate_model(
        model,
        criterion,
        test_loader,
        DEVICE,
    )

    return test_loss, test_ang_error, model_path


def main():

    # Settings for this experiment
    SETTINGS = {
        **get_base_settings(),
        "results_dir": "results/generalized",
        "results_file": "gen_loto_reserved_eval.csv",
    }

    # Use same targets as the generalized experiment
    targets = sorted(get_general_pool(SETTINGS["csv_path"])["target"].unique())

    # Reserved subjects for person-specific experiments
    reserved_subjects = get_person_specific_subjects()

    all_results = []

    for subject_id in reserved_subjects:
        for target in targets:
            print(f"Running gen_loto_reserved_eval | subject={subject_id} | target={target}")

            model_name = f"gen_loto_{target}"

            # Test set = one reserved subject on one target
            test_df = make_reserved_subject_target_split(
                subject_id=subject_id,
                target=target,
                csv_path=SETTINGS["csv_path"],
            )

            if len(test_df) == 0:
                print(f"Skipping subject={subject_id}, target={target} (no samples found)")
                continue

            test_loss, test_ang_error, model_path = evaluate_existing_model(
                settings=SETTINGS,
                model_name=model_name,
                test_df=test_df,
            )

            print(f"Test loss: {test_loss:.4f} | Test ang error: {test_ang_error:.4f}°\n")

            row_info = {
                "experiment_type": "general_loto_reserved_eval",
                "subject_spec_id": subject_id,
                "held_out_subject": subject_id,
                "held_out_target": target,
                "init_mode": "scratch",
            }

            result_row = {
                "model_name": model_name,
                "model_path": str(model_path),
                "pretrained_weights": None,
                **row_info,
                "n_test": len(test_df),
                "test_loss": test_loss,
                "test_ang_error": test_ang_error,
            }

            all_results.append(result_row)

    # Save results
    results_dir = Path(SETTINGS["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(all_results).to_csv(results_dir / SETTINGS["results_file"], index=False)

    print("Finished gen_loto_reserved_eval")


if __name__ == "__main__":
    main()