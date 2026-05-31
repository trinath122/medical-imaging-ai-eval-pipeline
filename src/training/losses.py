"""
Weighted Binary Cross-Entropy loss for multi-label chest X-ray classification.

NIH ChestX-ray14 is heavily imbalanced (e.g., Hernia ~0.2%, Infiltration ~18%).
Positive-class weights computed from training set label frequencies are passed in
at construction time so the loss is always dataset-aware.
"""
import torch
import torch.nn as nn


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weights: torch.Tensor):
        super().__init__()
        # pos_weights: [num_classes] — higher weight for rare classes
        self.register_buffer("pos_weights", pos_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weights,
        )
