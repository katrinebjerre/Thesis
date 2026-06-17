# preprocessing/process_tensors.py

from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


def preprocess(
    input_csv="data_utils/metadata.csv",
    output_csv="data_utils/metadata_tensors.csv",
    output_root="data_processed",
):
    """Resize, greyscale, and convert images to tensors, save them, and update metadata paths."""

    df = pd.read_csv(input_csv)

    transform = transforms.Compose([
        transforms.Resize((96, 128)),
        transforms.ToTensor(),
    ])

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    new_paths = []

    print(f"processing {len(df)} images...")

    for i, rel_path in enumerate(df["path"]):
        src = Path(rel_path)

        # Build output path without duplicating "data/" prefix
        rel = src.relative_to("data")
        dst = output_root / rel.with_suffix(".pt")
        dst.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(src).convert("L")
        tensor = transform(img)

        torch.save(tensor, dst)
        new_paths.append(str(dst))

        if i % 20000 == 0:
            print(i)

    df["path"] = new_paths
    df.to_csv(output_csv, index=False)

    print("done")


if __name__ == "__main__":
    preprocess()