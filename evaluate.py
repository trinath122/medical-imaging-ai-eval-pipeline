"""
Full evaluation pipeline on the held-out test set.

Produces:
  - Per-class AUC-ROC table
  - Overall diagnostic accuracy
  - MC Dropout uncertainty report
  - Failure mode classification CSV
  - All results logged to MLflow

Usage:
  python evaluate.py --checkpoint checkpoints/best_model.pt
  python evaluate.py --checkpoint checkpoints/best_model.pt --config experiments/configs/base_config.yaml
"""
import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from src.data.dataset import build_dataloaders, LABELS
from src.evaluation.failure_modes import classify_failures
from src.evaluation.metrics import compute_metrics, metrics_to_dataframe
from src.models.densenet import build_model
from src.models.uncertainty import mc_predict, uncertainty_summary
from src.utils.mlflow_utils import log_dataframe_artifact, log_failure_report, log_metrics_dict


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ChestXray model on test set")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--config", default="experiments/configs/base_config.yaml")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def run_mc_inference(model, loader, n_passes, device):
    """Run MC Dropout inference over the full loader."""
    all_means, all_stds, all_entropies, all_labels, all_meta = [], [], [], [], []

    for images, labels, meta in tqdm(loader, desc="MC inference"):
        images = images.to(device)
        mean_p, std_p, entropy = mc_predict(model, images, n_passes=n_passes)
        all_means.append(mean_p)
        all_stds.append(std_p)
        all_entropies.append(entropy)
        all_labels.append(labels.numpy())
        all_meta.extend([{k: v[i] for k, v in meta.items()} for i in range(len(images))])

    return (
        np.concatenate(all_means, axis=0),
        np.concatenate(all_stds, axis=0),
        np.concatenate(all_entropies, axis=0),
        np.concatenate(all_labels, axis=0),
        all_meta,
    )


def main():
    args = parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Checkpoint from epoch {ckpt.get('epoch', '?')}, val_auc={ckpt.get('val_auc', '?'):.4f}")

    # Data
    print("\nBuilding test loader ...")
    _, _, test_loader, _ = build_dataloaders(cfg, num_workers=args.num_workers)
    print(f"  Test set: {len(test_loader.dataset):,} images")

    # Compute label prevalence (for rare-class detection)
    label_prevalence = test_loader.dataset.labels.mean(axis=0)

    # MC Dropout inference
    n_passes = cfg["uncertainty"]["mc_dropout_passes"]
    print(f"\nRunning MC Dropout inference ({n_passes} passes) ...")
    probs, stds, entropies, labels, meta = run_mc_inference(model, test_loader, n_passes, device)

    # ── Metrics ────────────────────────────────────────────────────────────────
    print("\nComputing metrics ...")
    metrics = compute_metrics(probs, labels, threshold=args.threshold)
    metrics_df = metrics_to_dataframe(metrics)

    print("\n" + "=" * 60)
    print(metrics_df.to_string())
    print("=" * 60)
    print(f"\nMean AUC:              {metrics['mean_auc']:.4f}")
    print(f"Mean F1:               {metrics['mean_f1']:.4f}")
    print(f"Diagnostic Accuracy:   {metrics['diagnostic_accuracy']:.4f}")

    # ── Uncertainty summary ────────────────────────────────────────────────────
    unc_summary = uncertainty_summary(stds, LABELS)
    print(f"\nMean predictive entropy: {entropies.mean():.4f}")

    # ── Failure mode classification ────────────────────────────────────────────
    print("\nClassifying failure modes ...")
    failure_df = classify_failures(probs, stds, labels, meta, label_prevalence, args.threshold)
    total = len(test_loader.dataset)
    print(f"\nFailures: {len(failure_df):,} / {total:,} samples ({len(failure_df)/total*100:.1f}%)")

    # ── Save results ────────────────────────────────────────────────────────────
    metrics_df.to_csv(Path(args.results_dir) / "per_class_metrics.csv")
    failure_df.to_csv(Path(args.results_dir) / "failure_modes.csv", index=False)
    print(f"\nResults saved to {args.results_dir}/")

    # ── MLflow logging ─────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="evaluation"):
        log_metrics_dict(metrics)
        log_metrics_dict(unc_summary)
        mlflow.log_metric("mean_entropy", float(entropies.mean()))
        mlflow.log_metric("n_failures", len(failure_df))
        log_dataframe_artifact(metrics_df, "per_class_metrics.csv", args.results_dir)
        log_failure_report(failure_df, args.results_dir)
        mlflow.log_artifact(args.checkpoint)

    print("\nEvaluation complete. Results logged to MLflow.")


if __name__ == "__main__":
    main()
