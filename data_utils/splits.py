# data_utils/splits.py

from data_utils.dataset import load_metadata
from data_utils.subject_groups import get_person_specific_subjects


def exclude_subjects(df, subjects):
    """Return dataFrame excluding specified subjects."""
    return df[~df["subject"].isin(subjects)].copy()


def keep_only_subject(df, subject_id):
    """Return dataFrame with only the specified subject."""
    return df[df["subject"] == subject_id].copy()


# --------------------------------------------------
# General model splits
# --------------------------------------------------

def get_general_pool(csv_path="data_utils/metadata_tensors.csv"):
    """Load the full dataset and remove the reserved subjects."""
    df = load_metadata(csv_path)
    reserved_subjects = get_person_specific_subjects()
    return exclude_subjects(df, reserved_subjects)


def make_general_loso_split(test_subject, csv_path="data_utils/metadata_tensors.csv"):
    """Leave-One-Subject-Out split for the general model. INCLUDING THE RESERVED SUBJECTS."""
    df = load_metadata(csv_path)

    train_df = df[df["subject"] != test_subject].copy()
    test_df = df[df["subject"] == test_subject].copy()

    return train_df, test_df


def make_general_loto_split(target, csv_path="data_utils/metadata_tensors.csv"):
    """Leave-One-Target-Out split for the general model. EXCLUDING THE RESERVED SUBJECTS."""
    df = get_general_pool(csv_path)

    train_df = df[df["target"] != target].copy()
    test_df = df[df["target"] == target].copy()

    return train_df, test_df


# --------------------------------------------------
# Person-specific model splits
# --------------------------------------------------

def make_person_loto_split(subject_id, target, csv_path="data_utils/metadata_tensors.csv"):
    """Leave-One-Target-Out split for one subject."""
    df = load_metadata(csv_path)
    df = keep_only_subject(df, subject_id)

    train_df = df[df["target"] != target].copy()
    test_df = df[df["target"] == target].copy()

    return train_df, test_df


# --------------------------------------------------
# External evaluation splits
# --------------------------------------------------

def make_reserved_subject_target_split(subject_id, target, csv_path="data_utils/metadata_tensors.csv"):
    """
    Return only rows for one reserved subject and one target.
    Used for external evaluation of a generalized LOTO model.
    """
    df = load_metadata(csv_path)

    test_df = df[
        (df["subject"] == subject_id) &
        (df["target"] == target)
    ].copy()

    return test_df


# --------------------------------------------------
# Helper function
# --------------------------------------------------

def print_split_info(name, df_a, df_b, label_a="A", label_b="B"):
    """Print basic information (size, subjects, targets) for two dataset splits."""
    print("\n" + "=" * 50)
    print(f"{name}")
    print("=" * 50)

    print(f"{label_a + ' samples':<20}: {len(df_a)}")
    print(f"{label_b + ' samples':<20}: {len(df_b)}")

    if len(df_a) > 0:
        print(f"\n{label_a + ' subjects':<20}: {sorted(df_a['subject'].unique().tolist())}")
        print(f"{label_a + ' targets':<20}: {sorted(df_a['target'].unique().tolist())}")

    if len(df_b) > 0:
        print(f"\n{label_b + ' subjects':<20}: {sorted(df_b['subject'].unique().tolist())}")
        print(f"{label_b + ' targets':<20}: {sorted(df_b['target'].unique().tolist())}")