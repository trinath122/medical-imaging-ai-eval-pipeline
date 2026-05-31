"""
Monte Carlo Dropout uncertainty quantification.

Given a model with MCDropout layers, run T stochastic forward passes
and compute per-sample statistics:
  - mean prediction   (best point estimate)
  - predictive variance / std  (epistemic uncertainty)
  - predictive entropy          (total uncertainty)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


def mc_predict(
    model: nn.Module,
    images: torch.Tensor,
    n_passes: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        mean_probs  : [N, C]  mean sigmoid probability across T passes
        std_probs   : [N, C]  std deviation (epistemic uncertainty)
        entropy     : [N]     mean predictive entropy per sample
    """
    model.train()  # MCDropout stays active; but we still want BN in eval mode
    # Selectively set BN layers to eval
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()

    all_probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.stack(all_probs, axis=0)  # [T, N, C]
    mean_probs = all_probs.mean(axis=0)       # [N, C]
    std_probs = all_probs.std(axis=0)         # [N, C]

    # Predictive entropy: H = -sum(p * log(p))
    eps = 1e-8
    entropy = -(
        mean_probs * np.log(mean_probs + eps)
        + (1 - mean_probs) * np.log(1 - mean_probs + eps)
    ).mean(axis=1)  # [N]

    return mean_probs, std_probs, entropy


def uncertainty_summary(std_probs: np.ndarray, label_names: list) -> dict:
    """Return per-class mean uncertainty as a dict for logging."""
    return {f"uncertainty/{name}": float(std_probs[:, i].mean())
            for i, name in enumerate(label_names)}
