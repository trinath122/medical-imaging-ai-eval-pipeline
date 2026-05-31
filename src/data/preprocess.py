"""
Preprocessing pipeline for NIH ChestX-ray14 PNG images.

NIH distributes images as PNG (originally from DICOM). This module:
  1. Reads each PNG (or DICOM if present)
  2. Applies CLAHE histogram equalisation (standard chest X-ray enhancement)
  3. Resizes to target_size x target_size
  4. Saves normalised float16 PNG to processed_dir

Run: python -m src.data.preprocess --config experiments/configs/base_config.yaml
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def read_image(path: Path) -> np.ndarray:
    """Read PNG or DICOM, return uint8 grayscale array."""
    if path.suffix.lower() == ".dcm":
        if not DICOM_AVAILABLE:
            raise RuntimeError("pydicom not installed; cannot read DICOM files.")
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        # Normalise to 0-255
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        return arr.astype(np.uint8)

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid, tile_grid),
    )
    return clahe.apply(img)


def preprocess_image(path: Path, target_size: int) -> np.ndarray:
    img = read_image(path)
    img = apply_clahe(img)
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return img


def build_split_files(cfg: dict) -> None:
    """Generate train/val/test CSV splits from the NIH metadata."""
    raw_dir = Path(cfg["data"]["raw_dir"])
    splits_dir = Path(cfg["data"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    meta_path = raw_dir / cfg["data"]["metadata_file"]
    df = pd.read_csv(meta_path, encoding="latin-1")

    # Keep only images that were actually downloaded
    images_dir = raw_dir / "images"
    downloaded = {p.name for p in images_dir.glob("**/*.png")}
    df = df[df["Image Index"].isin(downloaded)].reset_index(drop=True)
    print(f"Images available after filtering: {len(df):,}")

    # Use NIH official train/test lists if present
    train_list_path = raw_dir / "train_val_list.txt"
    test_list_path = raw_dir / "test_list.txt"

    if train_list_path.exists() and test_list_path.exists():
        train_files = set(train_list_path.read_text().splitlines())
        test_files = set(test_list_path.read_text().splitlines())
        train_df = df[df["Image Index"].isin(train_files)]
        test_df = df[df["Image Index"].isin(test_files)]
        # Split train into train+val
        val_size = int(len(train_df) * cfg["data"]["val_ratio"])
        val_df = train_df.sample(val_size, random_state=cfg["data"]["random_seed"])
        train_df = train_df.drop(val_df.index)
    else:
        # Random split
        rng = np.random.default_rng(cfg["data"]["random_seed"])
        idx = rng.permutation(len(df))
        n = len(df)
        n_train = int(n * cfg["data"]["train_ratio"])
        n_val = int(n * cfg["data"]["val_ratio"])
        train_df = df.iloc[idx[:n_train]]
        val_df = df.iloc[idx[n_train : n_train + n_val]]
        test_df = df.iloc[idx[n_train + n_val :]]

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    print(f"Split sizes — train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")


def preprocess_all(cfg: dict) -> None:
    raw_dir = Path(cfg["data"]["raw_dir"]) / "images"
    proc_dir = Path(cfg["data"]["processed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    target_size = cfg["data"]["image_size"]

    image_paths = list(raw_dir.glob("**/*.png")) + list(raw_dir.glob("**/*.dcm"))
    print(f"Preprocessing {len(image_paths):,} images → {proc_dir}")

    errors = []
    for path in tqdm(image_paths, unit="img"):
        out_path = proc_dir / path.name.replace(".dcm", ".png")
        if out_path.exists():
            continue
        try:
            img = preprocess_image(path, target_size)
            cv2.imwrite(str(out_path), img)
        except Exception as e:
            errors.append((path.name, str(e)))

    if errors:
        print(f"\nFailed to process {len(errors)} images:")
        for name, err in errors[:10]:
            print(f"  {name}: {err}")

    print(f"Done. Processed images saved to {proc_dir}")


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    print("=== Step 1: Building split files ===")
    build_split_files(cfg)
    print("\n=== Step 2: Preprocessing images ===")
    preprocess_all(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/base_config.yaml")
    args = parser.parse_args()
    main(args.config)
