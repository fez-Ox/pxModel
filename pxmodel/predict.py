from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
import cv2
import numpy as np
import torch

from pxmodel.augmentation import LABEL_NAMES, get_tta_transforms, get_val_transform
from pxmodel.config import *
from pxmodel.model import MultiLabelBoxClassifier


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> MultiLabelBoxClassifier:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = MultiLabelBoxClassifier(
        num_labels=ckpt["num_labels"],
        backbone_name=ckpt.get("backbone", "efficientnet_b0"),
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


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
    """Run inference on a single image and return sigmoid probabilities.

    Returns
    -------
    np.ndarray
        Shape ``(num_labels,)`` with sigmoid probabilities.
    """
    tensor = transform(image=image)["image"].unsqueeze(0).to(device)
    logits = model(tensor)
    return torch.sigmoid(logits).cpu().numpy().squeeze(0)


@torch.no_grad()
def predict_tta(
    model: MultiLabelBoxClassifier,
    image: np.ndarray,
    tta_transforms: list,
    device: torch.device,
) -> np.ndarray:
    """Run TTA inference: average sigmoid outputs over multiple augmented views.

    Returns
    -------
    np.ndarray
        Shape ``(num_labels,)`` with averaged sigmoid probabilities.
    """
    all_sigmoids: List[np.ndarray] = []
    for tfm in tta_transforms:
        tensor = tfm(image=image)["image"].unsqueeze(0).to(device)
        logits = model(tensor)
        sigmoids = torch.sigmoid(logits).cpu().numpy().squeeze(0)
        all_sigmoids.append(sigmoids)
    return np.mean(all_sigmoids, axis=0)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_prediction(
    filename: str,
    confidences: np.ndarray,
    thresholds: List[float],
    label_names: List[str],
) -> str:
    """Format a single prediction for display.

    Returns
    -------
    str
        Human-readable multi-line summary.
    """
    lines = [f"  {filename}"]

    for i, name in enumerate(label_names):
        conf = confidences[i]
        is_positive = conf >= thresholds[i]
        tag = "YES" if is_positive else "NO "
        lines.append(f"    {name:<15s}  {tag}  (conf: {conf:.4f})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-label inference on a single image."
    )
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = load_model_from_checkpoint(checkpoint, device)
    num_labels = model.num_labels
    print(f"Model loaded from: {checkpoint}")
    print(f"Backbone: {model.backbone_name}  |  Labels: {num_labels}")

    # ── Thresholds ────────────────────────────────────────────────────────
    thresholds = [threshold] * num_labels
    print(f"Thresholds: {dict(zip(LABEL_NAMES, thresholds))}")

    # ── Prepare transforms ────────────────────────────────────────────────
    val_transform = get_val_transform(image_size=image_size)
    tta_transforms = (
        get_tta_transforms(image_size=image_size) if use_tta else None
    )
    if use_tta:
        print(f"TTA enabled: {len(tta_transforms)} augmented views per image")

    # ── Load image & run inference ────────────────────────────────────────
    image = load_image(Path(args.image_path))

    if tta_transforms is not None:
        confidences = predict_tta(model, image, tta_transforms, device)
    else:
        confidences = predict_single(model, image, val_transform, device)

    print("=" * 60)
    print("  PREDICTION")
    print("=" * 60)
    display_str = format_prediction(
        Path(args.image_path).name, confidences, thresholds, LABEL_NAMES
    )
    print(display_str)
    print("=" * 60)

if __name__ == "__main__":
    main()
