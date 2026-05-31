"""
Main training entry point.

Usage:
  python train.py
  python train.py --config experiments/configs/base_config.yaml
  python train.py --epochs 10 --batch_size 16 --lr 0.0002
"""
import argparse

import torch
import yaml

from src.data.dataset import build_dataloaders
from src.models.densenet import build_model
from src.training.trainer import train


def parse_args():
    p = argparse.ArgumentParser(description="Train ChestXray DenseNet-121")
    p.add_argument("--config", default="experiments/configs/base_config.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nBuilding data loaders ...")
    train_loader, val_loader, _, pos_weights = build_dataloaders(cfg, num_workers=args.num_workers)
    print(f"  train={len(train_loader.dataset):,}  val={len(val_loader.dataset):,}")

    print("\nBuilding model ...")
    model = build_model(cfg, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}  Trainable: {trainable_params:,}")

    print("\nStarting training ...")
    best_ckpt = train(model, train_loader, val_loader, cfg, device, args.checkpoint_dir)
    print(f"\nBest checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
