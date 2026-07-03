from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torchvision import models

BACKBONE_REGISTRY: Dict[str, tuple] = {
    # EfficientNet family (b0 – b7)
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights, 1280),
    "efficientnet_b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights, 1280),
    "efficientnet_b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights, 1408),
    "efficientnet_b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights, 1536),
    "efficientnet_b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights, 1792),
    "efficientnet_b5": (models.efficientnet_b5, models.EfficientNet_B5_Weights, 2048),
    "efficientnet_b6": (models.efficientnet_b6, models.EfficientNet_B6_Weights, 2304),
    "efficientnet_b7": (models.efficientnet_b7, models.EfficientNet_B7_Weights, 2560),
    # EfficientNetV2
    "efficientnet_v2_s": (
        models.efficientnet_v2_s,
        models.EfficientNet_V2_S_Weights,
        1280,
    ),
    # MobileNet
    "mobilenet_v3_large": (
        models.mobilenet_v3_large,
        models.MobileNet_V3_Large_Weights,
        960,
    ),
    # ConvNeXt family
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights, 768),
    "convnext_small": (models.convnext_small, models.ConvNeXt_Small_Weights, 768),
    "convnext_base": (models.convnext_base, models.ConvNeXt_Base_Weights, 1024),
    "convnext_large": (models.convnext_large, models.ConvNeXt_Large_Weights, 1536),
}


class MultiLabelBoxClassifier(nn.Module):
    def __init__(
        self,
        num_labels=4,
        dropout=0.3,
        pretrained=True,
        backbone_name="efficientnet_b0",
    ):
        super().__init__()

        if backbone_name not in BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone {backbone_name!r}. "
                f"Supported: {sorted(BACKBONE_REGISTRY)}"
            )

        self.num_labels = num_labels
        self.backbone_name = backbone_name

        entry = BACKBONE_REGISTRY[backbone_name]
        model_fn, weights_enum, in_features = entry

        weights = weights_enum.DEFAULT if pretrained else None
        base = model_fn(weights=weights)

        self.features = base.features
        self.avgpool = base.avgpool

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(256, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def freeze_backbone(model: MultiLabelBoxClassifier) -> None:
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_backbone(model: MultiLabelBoxClassifier) -> None:
    for param in model.features.parameters():
        param.requires_grad = True


def get_model_info(model: MultiLabelBoxClassifier) -> Dict[str, float]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / 1024 / 1024

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": round(model_size_mb, 2),
    }
