"""
Emotion recognition model based on ResNet.
Transfer learning from ImageNet-pretrained weights, fine-tuned on FER2013.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision import models


def create_emotion_model(
    num_classes: int = 7,
    backbone: str = "resnet18",
    pretrained: bool = True,
    checkpoint: Optional[Union[str, Path]] = None,
) -> nn.Module:
    """
    Create emotion recognition model.

    Args:
        num_classes: Number of emotion classes (7 for FER2013)
        backbone: resnet18 | resnet34 | resnet50
        pretrained: Use ImageNet pretrained weights
        checkpoint: Path to fine-tuned checkpoint (overrides pretrained)

    Returns:
        EmotionResNet model
    """
    model = EmotionResNet(num_classes=num_classes, backbone=backbone, pretrained=pretrained)

    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)

    return model


class EmotionResNet(nn.Module):
    """
    ResNet-based emotion classifier.
    Replaces final FC layer for 7-class emotion prediction.
    """

    BACKBONES = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
    }

    def __init__(
        self,
        num_classes: int = 7,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        if backbone not in self.BACKBONES:
            raise ValueError(f"Unknown backbone: {backbone}")

        resnet_fn = self.BACKBONES[backbone]
        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = resnet_fn(weights=weights)

        # Remove final FC
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        in_features = resnet.fc.in_features

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (predicted class indices, probabilities)."""
        probs = self.predict_proba(x)
        preds = probs.argmax(dim=1)
        confidences = probs.max(dim=1).values
        return preds, confidences
