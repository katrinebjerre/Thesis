# preprocessing/make_metadata.py

from pathlib import Path
import pandas as pd
import numpy as np

# Project paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
BAD_PATHS_FILE = ROOT_DIR / "preprocessing" / "pupil_detection" / "failed_pupil_detection.csv"
OUTPUT_FILE = ROOT_DIR / "data_utils" / "metadata.csv"

# Screen resolution
SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 900

# Create normalized grid positions (5x5)
xs = np.linspace(0.15, 0.85, 5)
ys = np.linspace(0.15, 0.85, 5)


def get_norm_coord(target_id):
    """Convert target ID (0-24) to normalized (x, y) grid coordinates."""
    row = target_id // 5
    col = target_id % 5
    return xs[col], ys[row]


def normalize_path(path_str):
    """Normalize file paths to a consistent format for matching."""
    path_str = str(path_str).strip().replace("\\", "/")
    path_str = path_str.replace("../", "")
    return path_str


def make_metadata():
    """Create metadata CSV with subject, target, coordinates, and filtered image paths."""
    paths = list(DATA_DIR.glob("*/*/*.png"))

    rows = []
    for p in paths:
        subject = p.parts[-3]
        filename = p.stem
        target = int(filename.split("_")[0])

        x_norm, y_norm = get_norm_coord(target)
        x_pixel = x_norm * SCREEN_WIDTH
        y_pixel = y_norm * SCREEN_HEIGHT

        rows.append({
            "subject": subject,
            "target": target,
            "x_norm": x_norm,
            "y_norm": y_norm,
            "x_pixel": x_pixel,
            "y_pixel": y_pixel,
            "path": str(p.relative_to(ROOT_DIR)).replace("\\", "/")
        })

    df = pd.DataFrame(rows)

    bad_paths = pd.read_csv(BAD_PATHS_FILE)

    df["path"] = df["path"].apply(normalize_path)
    bad_paths["image_path"] = bad_paths["image_path"].apply(normalize_path)

    # Remove bad images
    df = df[~df["path"].isin(bad_paths["image_path"])]

    df = df.sort_values(["subject", "target", "path"]).reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False, float_format="%.3f")

    return df


if __name__ == "__main__":
    df = make_metadata()
    print(df.head())
    print("Rows in metadata:", len(df))