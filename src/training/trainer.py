"""
Training loop with:
  - Mixed precision (torch.cuda.amp)
  - Cosine LR scheduler with warm-up
  - Early stopping on validation AUC
  - MLflow experiment tracking
  - Best-checkpoint saving
"""
import time
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import LABELS
from src.training.losses import WeightedBCELoss


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, labels, _ in tqdm(loader, desc="  train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    for images, labels, _ in tqdm(loader, desc="  eval ", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    avg_loss = total_loss / len(loader.dataset)

    # Per-class AUC-ROC (skip classes with no positive samples)
    aucs = []
    per_class_auc = {}
    for i, name in enumerate(LABELS):
        if all_labels[:, i].sum() > 0:
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            aucs.append(auc)
            per_class_auc[name] = auc

    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    return avg_loss, mean_auc, per_class_auc, all_probs, all_labels


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    checkpoint_dir: str = "checkpoints",
) -> str:
    """Full training loop. Returns path to best checkpoint."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_ckpt = str(Path(checkpoint_dir) / "best_model.pt")

    pos_weights = train_loader.dataset.labels
    pos_weights_tensor = torch.tensor(
        (len(pos_weights) - pos_weights.sum(0)) / (pos_weights.sum(0) + 1e-8),
        dtype=torch.float32,
    ).to(device)

    criterion = WeightedBCELoss(pos_weights_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=1e-6,
    )
    scaler = GradScaler("cuda", enabled=cfg["training"]["mixed_precision"])
    early_stopper = EarlyStopping(patience=cfg["training"]["early_stopping_patience"])

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run():
        mlflow.log_params({
            "architecture": cfg["model"]["architecture"],
            "pretrained": cfg["model"]["pretrained"],
            "dropout_rate": cfg["model"]["dropout_rate"],
            "epochs": cfg["training"]["epochs"],
            "batch_size": cfg["training"]["batch_size"],
            "lr": cfg["training"]["learning_rate"],
        })

        best_auc = 0.0
        for epoch in range(1, cfg["training"]["epochs"] + 1):
            t0 = time.time()

            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
            val_loss, val_auc, per_class_auc, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:>3}/{cfg['training']['epochs']}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_auc={val_auc:.4f}  [{elapsed:.0f}s]"
            )

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mean_auc": val_auc,
                **{f"val_auc/{k}": v for k, v in per_class_auc.items()},
            }, step=epoch)

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": val_auc,
                    "cfg": cfg,
                }, best_ckpt)
                print(f"  ✓ New best AUC {val_auc:.4f} — checkpoint saved")

            if early_stopper(val_auc):
                print(f"Early stopping triggered at epoch {epoch}")
                break

        mlflow.log_metric("best_val_auc", best_auc)
        mlflow.log_artifact(best_ckpt)
        print(f"\nTraining complete. Best val AUC: {best_auc:.4f}")

    return best_ckpt
