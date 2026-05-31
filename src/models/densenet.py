"""
DenseNet-121 chest pathology classifier with MC Dropout for uncertainty estimation.

Architecture follows CheXNet (Rajpurkar et al., 2017):
  - ImageNet-pretrained DenseNet-121 backbone
  - Global Average Pooling
  - Dropout layer (kept active at inference for MC Dropout)
  - Single linear head outputting 14 logits (one per NIH pathology class)
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights


class MCDropout(nn.Dropout):
    """Dropout that stays active even during model.eval() for MC sampling."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.dropout(x, p=self.p, training=True)


class ChestXrayDenseNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 14,
        dropout_rate: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)

        # Keep all feature layers; replace classifier
        self.features = backbone.features
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = MCDropout(p=dropout_rate)
        self.classifier = nn.Linear(1024, num_classes)

        # Initialise classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = torch.relu(features)
        pooled = self.global_avg_pool(features)
        pooled = torch.flatten(pooled, 1)
        dropped = self.dropout(pooled)
        logits = self.classifier(dropped)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def build_model(cfg: dict, device: torch.device) -> ChestXrayDenseNet:
    model = ChestXrayDenseNet(
        num_classes=cfg["model"]["num_classes"],
        dropout_rate=cfg["model"]["dropout_rate"],
        pretrained=cfg["model"]["pretrained"],
    )
    return model.to(device)
