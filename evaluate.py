"""Standalone evaluation script for the trained multi-label box classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from augmentation import LABEL_NAMES, get_val_transform
from config import *
from dataset_multilabel import MultiLabelBoxDataset
from model import MultiLabelBoxClassifier


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> MultiLabelBoxClassifier:
    """Instantiate the model from a checkpoint dictionary.

    Expected checkpoint keys: ``backbone``, ``num_labels``, ``model_state_dict``.
    """
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


@torch.no_grad()
def collect_predictions(
    model: MultiLabelBoxClassifier,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on the full dataloader and return sigmoid outputs and ground truths.

    Returns
    -------
    all_sigmoids : np.ndarray, shape ``(N, num_labels)``
        Sigmoid probabilities for every sample.
    all_targets : np.ndarray, shape ``(N, num_labels)``
        Ground-truth binary labels for every sample.
    """
    all_sigmoids: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for images, labels in dataloader:
        images = images.to(device)
        logits = model(images)
        sigmoids = torch.sigmoid(logits).cpu().numpy()
        all_sigmoids.append(sigmoids)
        all_targets.append(labels.numpy())

    return np.concatenate(all_sigmoids), np.concatenate(all_targets)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def find_best_thresholds(
    sigmoids: np.ndarray,
    targets: np.ndarray,
    label_names: List[str],
) -> Dict[str, float]:
    """Sweep thresholds per label and return the one that maximises F1.

    Thresholds tested: 0.1, 0.2, … , 0.9.

    Returns
    -------
    dict
        ``{label_name: best_threshold}``
    """
    candidate_thresholds = np.arange(0.1, 1.0, 0.1)
    best: Dict[str, float] = {}

    print("\n" + "=" * 70)
    print("  THRESHOLD SWEEP  (per-label, maximising F1)")
    print("=" * 70)

    for idx, label in enumerate(label_names):
        best_f1 = -1.0
        best_t = 0.5

        for t in candidate_thresholds:
            preds = (sigmoids[:, idx] >= t).astype(int)
            f1 = f1_score(targets[:, idx].astype(int), preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(round(t, 1))

        best[label] = best_t
        print(f"  {label:<15s}  best threshold = {best_t:.1f}  (F1 = {best_f1:.4f})")

    print("=" * 70)
    return best


def print_results(
    sigmoids: np.ndarray,
    targets: np.ndarray,
    thresholds: List[float],
    label_names: List[str],
) -> None:
    """Print a detailed per-label classification report and exact match ratio."""
    predictions = np.zeros_like(sigmoids, dtype=int)
    for i, t in enumerate(thresholds):
        predictions[:, i] = (sigmoids[:, i] >= t).astype(int)

    targets_int = targets.astype(int)

    # --- Per-label classification report ---
    print("\n" + "=" * 70)
    print("  CLASSIFICATION REPORT  (per label)")
    print("=" * 70)
    print(
        classification_report(
            targets_int,
            predictions,
            target_names=label_names,
            zero_division=0,
        )
    )

    # --- Macro-averaged F1 ---
    macro_f1 = f1_score(
        targets_int, predictions, average="macro", zero_division=0
    )
    micro_f1 = f1_score(
        targets_int, predictions, average="micro", zero_division=0
    )
    sample_f1 = f1_score(
        targets_int, predictions, average="samples", zero_division=0
    )

    # --- Exact match ratio ---
    exact_match = np.all(predictions == targets_int, axis=1).mean()

    print("-" * 70)
    print(f"  Macro F1:          {macro_f1:.4f}")
    print(f"  Micro F1:          {micro_f1:.4f}")
    print(f"  Sample-avg F1:     {sample_f1:.4f}")
    print(f"  Exact match ratio: {exact_match:.4f}")
    print("-" * 70)

    # --- Thresholds used ---
    print("\n  Thresholds used:")
    for name, t in zip(label_names, thresholds):
        print(f"    {name:<15s} = {t:.2f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = load_model_from_checkpoint(checkpoint, device)
    print(f"Model loaded from: {checkpoint}")

    # ── Build dataset / dataloader ────────────────────────────────────────
    transform = get_val_transform(image_size=image_size)
    dataset = MultiLabelBoxDataset(
        images_dir=images_dir,
        labels_csv=test_csv,
        transform=transform,
        split="test",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"Test samples: {len(dataset)}")

    # ── Collect predictions ───────────────────────────────────────────────
    sigmoids, targets = collect_predictions(model, dataloader, device)

    # ── Threshold selection ───────────────────────────────────────────────
    if find_best_thresholds:
        best = find_best_thresholds(sigmoids, targets, LABEL_NAMES)
        thresholds = [best[name] for name in LABEL_NAMES]
    else:
        thresholds = [threshold] * len(LABEL_NAMES)

    # ── Print detailed results ────────────────────────────────────────────
    print_results(sigmoids, targets, thresholds, LABEL_NAMES)


if __name__ == "__main__":
    main()
