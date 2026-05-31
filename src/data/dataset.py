"""
PyTorch Dataset for NIH ChestX-ray14.

Each sample returns:
  image  : float32 tensor [3, H, W]  (grayscale replicated to 3 channels for ImageNet pretrained weights)
  labels : float32 tensor [14]        (multi-hot encoding, one per pathology)
  meta   : dict with patient_id, image_index, view_position
"""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]


def get_train_transforms(image_size: int) -> A.Compose:
    return A.Compose([
        A.RandomRotate90(p=0.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(5, 30), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int) -> A.Compose:
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def encode_labels(finding_labels: str) -> np.ndarray:
    """Convert pipe-separated label string to multi-hot float32 array."""
    vec = np.zeros(len(LABELS), dtype=np.float32)
    for label in finding_labels.split("|"):
        label = label.strip()
        if label in LABELS:
            vec[LABELS.index(label)] = 1.0
    return vec


class ChestXrayDataset(Dataset):
    def __init__(
        self,
        split_csv: str,
        images_dir: str,
        transform: Optional[A.Compose] = None,
        image_size: int = 224,
    ):
        self.df = pd.read_csv(split_csv)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.image_size = image_size

        # Pre-encode all labels for speed
        self.labels = np.stack(
            self.df["Finding Labels"].apply(encode_labels).values
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.images_dir / row["Image Index"]

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback: black image (avoids crash during training)
            img = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # Replicate grayscale to 3 channels (expected by ImageNet pretrained weights)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.transform:
            augmented = self.transform(image=img)
            img_tensor = augmented["image"]  # [3, H, W] float32
        else:
            img = img.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1))

        label_tensor = torch.from_numpy(self.labels[idx])

        meta = {
            "image_index": row["Image Index"],
            "patient_id": str(row.get("Patient ID", "")),
            "view_position": str(row.get("View Position", "")),
        }

        return img_tensor, label_tensor, meta


def build_dataloaders(cfg: dict, num_workers: int = 4):
    """Return (train_loader, val_loader, test_loader, pos_weights) given config dict."""
    splits_dir = Path(cfg["data"]["splits_dir"])
    images_dir = Path(cfg["data"]["processed_dir"])
    image_size = cfg["data"]["image_size"]
    batch_size = cfg["training"]["batch_size"]

    train_ds = ChestXrayDataset(
        split_csv=str(splits_dir / "train.csv"),
        images_dir=str(images_dir),
        transform=get_train_transforms(image_size),
        image_size=image_size,
    )
    val_ds = ChestXrayDataset(
        split_csv=str(splits_dir / "val.csv"),
        images_dir=str(images_dir),
        transform=get_val_transforms(image_size),
        image_size=image_size,
    )
    test_ds = ChestXrayDataset(
        split_csv=str(splits_dir / "test.csv"),
        images_dir=str(images_dir),
        transform=get_val_transforms(image_size),
        image_size=image_size,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Compute positive class weights for weighted BCE loss
    label_counts = train_ds.labels.sum(axis=0)
    n_samples = len(train_ds)
    pos_weights = torch.tensor(
        (n_samples - label_counts) / (label_counts + 1e-8),
        dtype=torch.float32,
    )

    return train_loader, val_loader, test_loader, pos_weights
