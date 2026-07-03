from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torchvision import models


class _ViTFeatures(nn.Module):
    """Feature extractor wrapper for Vision Transformer.

    Runs patch embedding + class token + positional encoding + encoder,
    then returns the CLS token embedding (bypasses the classification head).
    """

    def __init__(self, vit) -> None:
        super().__init__()
        self.vit = vit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit._process_input(x)
        n = x.shape[0]
        x = torch.cat([self.vit.class_token.expand(n, -1, -1), x], dim=1)
        x = self.vit.encoder(x)
        return x[:, 0]  # CLS token


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
    # RegNetY
    "regnet_y_3_2gf": (
        models.regnet_y_3_2gf,
        models.RegNet_Y_3_2GF_Weights,
        1512,
        "regnet",
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
    # Vision Transformer family
    "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights, 768, "vit"),
    "vit_b_32": (models.vit_b_32, models.ViT_B_32_Weights, 768, "vit"),
    "vit_l_16": (models.vit_l_16, models.ViT_L_16_Weights, 1024, "vit"),
    "vit_l_32": (models.vit_l_32, models.ViT_L_32_Weights, 1024, "vit"),
    "vit_h_14": (models.vit_h_14, models.ViT_H_14_Weights, 1280, "vit"),
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
        model_fn, weights_enum, in_features = entry[:3]
        build_mode = entry[3] if len(entry) > 3 else "features"

        weights = weights_enum.DEFAULT if pretrained else None
        base = model_fn(weights=weights)

        if build_mode == "regnet":
            self.features = nn.Sequential(base.stem, base.trunk_output)
            self.avgpool = base.avgpool
        elif build_mode == "vit":
            self.features = _ViTFeatures(base)
            self.avgpool = nn.Identity()
        else:
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
