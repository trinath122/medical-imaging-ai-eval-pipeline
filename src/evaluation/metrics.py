"""
Evaluation metrics for multi-label chest X-ray classification.

Computes:
  - Per-class AUC-ROC
  - Mean AUC across all classes
  - Per-class accuracy at 0.5 threshold
  - Overall diagnostic accuracy (any pathology correctly identified)
  - F1 score per class
  - Confusion stats (TP, FP, TN, FN) per class
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
from typing import Dict

from src.data.dataset import LABELS


def compute_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Args:
        probs  : [N, 14] sigmoid probabilities
        labels : [N, 14] ground truth multi-hot
        threshold : decision threshold
    Returns:
        dict of metric_name -> value
    """
    preds = (probs >= threshold).astype(int)
    metrics = {}

    aucs = []
    f1s = []
    for i, name in enumerate(LABELS):
        if labels[:, i].sum() == 0:
            continue  # skip if no positives in split

        auc = roc_auc_score(labels[:, i], probs[:, i])
        f1 = f1_score(labels[:, i], preds[:, i], zero_division=0)
        acc = accuracy_score(labels[:, i], preds[:, i])
        tn, fp, fn, tp = confusion_matrix(labels[:, i], preds[:, i], labels=[0, 1]).ravel()

        metrics[f"auc/{name}"] = float(auc)
        metrics[f"f1/{name}"] = float(f1)
        metrics[f"acc/{name}"] = float(acc)
        metrics[f"tp/{name}"] = int(tp)
        metrics[f"fp/{name}"] = int(fp)
        metrics[f"tn/{name}"] = int(tn)
        metrics[f"fn/{name}"] = int(fn)
        aucs.append(auc)
        f1s.append(f1)

    metrics["mean_auc"] = float(np.mean(aucs)) if aucs else 0.0
    metrics["mean_f1"] = float(np.mean(f1s)) if f1s else 0.0

    # Overall diagnostic accuracy:
    # A prediction is "correct" if for each sample at least one true pathology is detected
    # or the sample is correctly predicted as normal.
    sample_any_correct = (
        ((preds * labels).sum(axis=1) > 0) |         # at least one TP
        ((preds.sum(axis=1) == 0) & (labels.sum(axis=1) == 0))  # both normal
    )
    metrics["diagnostic_accuracy"] = float(sample_any_correct.mean())

    return metrics


def metrics_to_dataframe(metrics: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for name in LABELS:
        if f"auc/{name}" not in metrics:
            continue
        rows.append({
            "pathology": name,
            "AUC": round(metrics[f"auc/{name}"], 4),
            "F1": round(metrics[f"f1/{name}"], 4),
            "Accuracy": round(metrics[f"acc/{name}"], 4),
            "TP": metrics[f"tp/{name}"],
            "FP": metrics[f"fp/{name}"],
            "TN": metrics[f"tn/{name}"],
            "FN": metrics[f"fn/{name}"],
        })
    return pd.DataFrame(rows).set_index("pathology")
