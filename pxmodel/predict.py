from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from torchao.quantization import (
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    quantize_,
)

from pxmodel.augmentation import LABEL_NAMES, get_tta_transforms, get_val_transform
from pxmodel.config import *
from pxmodel.model import MultiLabelBoxClassifier

QUANT_METHODS = {
    "int8_wo": ("int8 weight-only", lambda: Int8WeightOnlyConfig(version=2)),
    "int8_dynamic": (
        "int8 dynamic act+wt",
        lambda: Int8DynamicActivationInt8WeightConfig(version=2),
    ),
}


def _apply_method(model, method_key):
    if method_key not in QUANT_METHODS:
        raise ValueError(f"Unknown method {method_key!r}")
    _, config_fn = QUANT_METHODS[method_key]
    quantize_(model, config_fn(), device=next(model.parameters()).device.type)


def load_checkpoint(
    path: str | Path,
    device: torch.device,
    backbone_name: str,
) -> MultiLabelBoxClassifier:
    raw = torch.load(path, map_location=device, weights_only=False)

    if isinstance(raw, MultiLabelBoxClassifier):
        raw.to(device).eval()
        return raw

    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
        backbone = backbone_name or raw.get("backbone", "efficientnet_b0")
        num_labels = raw.get("num_labels", 4)
        model = MultiLabelBoxClassifier(
            num_labels=num_labels,
            backbone_name=backbone,
            pretrained=False,
        )
        model.load_state_dict(state_dict)
        model.to(device).eval()
        return model

    if isinstance(raw, dict) and "state_dict" in raw and "method" in raw:
        backbone = backbone_name or raw.get("backbone", "efficientnet_b0")
        num_labels = raw.get("num_labels", 4)
        model = MultiLabelBoxClassifier(
            num_labels=num_labels,
            backbone_name=backbone,
            pretrained=False,
        )
        _apply_method(model, raw["method"])
        model.load_state_dict(raw["state_dict"])
        model.to(device).eval()
        return model

    if isinstance(raw, dict):
        backbone = backbone_name or "efficientnet_b0"
        model = MultiLabelBoxClassifier(
            num_labels=4,
            backbone_name=backbone,
            pretrained=False,
        )
        model.load_state_dict(raw, strict=False)
        model.to(device).eval()
        return model

    raise ValueError(f"Unrecognised checkpoint format in {path}")


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@torch.no_grad()
def predict_single(
    model: MultiLabelBoxClassifier,
    image: np.ndarray,
    transform,
    device: torch.device,
) -> np.ndarray:
    # Preprocessing and host/device transfer are intentionally done before
    # timing so the printed value is model forward-pass latency only.
    tensor = transform(image=image)["image"].unsqueeze(0).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    logits = model(tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    # Postprocessing is intentionally done after timing.
    result = torch.sigmoid(logits).cpu().numpy().squeeze(0)

    print(f"Model inference time (single forward pass): {end_time - start_time:.4f}s")
    return result


@torch.no_grad()
def predict_tta(
    model: MultiLabelBoxClassifier,
    image: np.ndarray,
    tta_transforms: list,
    device: torch.device,
) -> np.ndarray:
    all_sigmoids: List[np.ndarray] = []
    for tfm in tta_transforms:
        tensor = tfm(image=image)["image"].unsqueeze(0).to(device)
        logits = model(tensor)
        sigmoids = torch.sigmoid(logits).cpu().numpy().squeeze(0)
        all_sigmoids.append(sigmoids)
    return np.mean(all_sigmoids, axis=0)


def format_prediction(
    filename: str,
    confidences: np.ndarray,
    thresholds: List[float],
    label_names: List[str],
) -> str:
    lines = [f"  {filename}"]
    for i, name in enumerate(label_names):
        conf = confidences[i]
        is_positive = conf >= thresholds[i]
        tag = "YES" if is_positive else "NO "
        lines.append(f"    {name:<15s}  {tag}  (conf: {conf:.4f})")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-label inference on a single image."
    )
    parser.add_argument("--image", type=str, help="Path to the input image")
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint file")
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Backbone name (override if checkpoint metadata is missing/wrong)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = load_checkpoint(checkpoint, device, args.backbone)
    num_labels = model.num_labels
    print(f"Model loaded from: {checkpoint}")
    print(f"Backbone: {model.backbone_name}  |  Labels: {num_labels}")

    thresholds = [threshold] * num_labels
    print(f"Thresholds: {dict(zip(LABEL_NAMES, thresholds))}")

    val_transform = get_val_transform(image_size=image_size)
    tta_transforms = get_tta_transforms(image_size=image_size) if use_tta else None

    image = load_image(Path(args.image))

    if tta_transforms is not None:
        confidences = predict_tta(model, image, tta_transforms, device)
    else:
        confidences = predict_single(model, image, val_transform, device)

    print("=" * 60)
    print("  PREDICTION")
    print("=" * 60)
    display_str = format_prediction(
        Path(args.image).name, confidences, thresholds, LABEL_NAMES
    )
    print(display_str)
    print("=" * 60)


if __name__ == "__main__":
    main()
