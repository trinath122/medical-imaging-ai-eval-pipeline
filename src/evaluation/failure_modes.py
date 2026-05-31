"""
Failure mode classification for chest X-ray predictions.

Classifies each wrong prediction into one of four modes:
  1. FALSE_POSITIVE_HIGH_CONF  — model predicted pathology confidently, but none present
  2. FALSE_NEGATIVE_HIGH_CONF  — pathology present but model confidently said normal
  3. FALSE_POSITIVE_UNCERTAIN  — wrong positive prediction with high uncertainty
  4. FALSE_NEGATIVE_UNCERTAIN  — missed pathology with high uncertainty

Additionally flags:
  - MULTI_LABEL_CONFUSION  — correct finding count but wrong specific labels
  - RARE_CLASS_MISS         — missed a class with low training prevalence
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from src.data.dataset import LABELS


HIGH_CONF_THRESHOLD = 0.75
UNCERTAIN_STD_THRESHOLD = 0.15
RARE_CLASS_PREVALENCE = 0.02  # classes below 2% prevalence


@dataclass
class FailureSample:
    image_index: str
    patient_id: str
    true_labels: List[str]
    pred_labels: List[str]
    failure_modes: List[str]
    max_confidence: float
    max_uncertainty: float
    probs: np.ndarray = field(repr=False)
    stds: np.ndarray = field(repr=False)


def _active_labels(vec: np.ndarray) -> List[str]:
    return [LABELS[i] for i in range(len(LABELS)) if vec[i] > 0.5]


def classify_failures(
    probs: np.ndarray,
    stds: np.ndarray,
    labels: np.ndarray,
    metadata: List[dict],
    label_prevalence: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Args:
        probs    : [N, 14] mean MC probabilities
        stds     : [N, 14] MC std (uncertainty)
        labels   : [N, 14] ground truth
        metadata : list of dicts with 'image_index', 'patient_id'
        label_prevalence : [14] class frequency in training set
    Returns:
        DataFrame with one row per failure sample
    """
    preds = (probs >= threshold).astype(int)
    failures: List[FailureSample] = []

    for i in range(len(probs)):
        true = labels[i]
        pred = preds[i]

        # Only analyse incorrect predictions
        if np.array_equal(true, pred):
            continue

        modes: List[str] = []

        for c in range(len(LABELS)):
            p = probs[i, c]
            s = stds[i, c]

            fp = pred[c] == 1 and true[c] == 0
            fn = pred[c] == 0 and true[c] == 1

            if fp:
                if p >= HIGH_CONF_THRESHOLD:
                    modes.append("FALSE_POSITIVE_HIGH_CONF")
                elif s >= UNCERTAIN_STD_THRESHOLD:
                    modes.append("FALSE_POSITIVE_UNCERTAIN")
                else:
                    modes.append("FALSE_POSITIVE")

            if fn:
                if (1 - p) >= HIGH_CONF_THRESHOLD:
                    modes.append("FALSE_NEGATIVE_HIGH_CONF")
                elif s >= UNCERTAIN_STD_THRESHOLD:
                    modes.append("FALSE_NEGATIVE_UNCERTAIN")
                else:
                    modes.append("FALSE_NEGATIVE")

                if label_prevalence is not None and label_prevalence[c] < RARE_CLASS_PREVALENCE:
                    modes.append("RARE_CLASS_MISS")

        # Multi-label confusion: correct count, wrong assignments
        if true.sum() > 0 and pred.sum() == true.sum() and not np.array_equal(true, pred):
            modes.append("MULTI_LABEL_CONFUSION")

        modes = list(dict.fromkeys(modes))  # deduplicate, preserve order

        meta = metadata[i] if i < len(metadata) else {}
        failures.append(FailureSample(
            image_index=meta.get("image_index", str(i)),
            patient_id=meta.get("patient_id", ""),
            true_labels=_active_labels(true),
            pred_labels=_active_labels(pred.astype(float)),
            failure_modes=modes,
            max_confidence=float(probs[i].max()),
            max_uncertainty=float(stds[i].max()),
            probs=probs[i],
            stds=stds[i],
        ))

    rows = [
        {
            "image_index": f.image_index,
            "patient_id": f.patient_id,
            "true_labels": "|".join(f.true_labels) or "No Finding",
            "pred_labels": "|".join(f.pred_labels) or "No Finding",
            "failure_modes": "|".join(f.failure_modes),
            "max_confidence": round(f.max_confidence, 4),
            "max_uncertainty": round(f.max_uncertainty, 4),
        }
        for f in failures
    ]
    df = pd.DataFrame(rows)

    if not df.empty:
        # Summary counts
        mode_counts = {}
        for row in rows:
            for mode in row["failure_modes"].split("|"):
                if mode:
                    mode_counts[mode] = mode_counts.get(mode, 0) + 1
        print("\nFailure mode summary:")
        for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
            print(f"  {mode:<35} {count:>5}")

    return df
