"""Multi-label training pipeline for box-condition classification."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from augmentation import get_train_transform, get_val_transform
from config import *
from dataset_multilabel import MultiLabelBoxDataset
from model import MultiLabelBoxClassifier, freeze_backbone, get_model_info, unfreeze_backbone

# Label names matching the CSV column order.
LABEL_NAMES: list[str] = ["damaged", "plastic_wrap", "sealed", "open"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute per-label and aggregate multi-label metrics.

    Args:
        predictions: Raw logits or sigmoid probabilities, shape ``(N, L)``.
        targets: Ground-truth binary labels, shape ``(N, L)``.
        threshold: Decision threshold applied to *sigmoid(predictions)*.

    Returns:
        Dictionary with per-label precision / recall / F1, mean-F1, and
        exact-match ratio.
    """
    # Apply sigmoid if predictions look like raw logits (may contain negatives).
    probs = 1.0 / (1.0 + np.exp(-predictions.astype(np.float64)))
    preds_binary = (probs >= threshold).astype(np.int32)
    targets_int = targets.astype(np.int32)

    num_labels = targets.shape[1]
    metrics: dict[str, Any] = {}

    per_label_f1: list[float] = []
    for i in range(num_labels):
        label = LABEL_NAMES[i] if i < len(LABEL_NAMES) else f"label_{i}"
        p = precision_score(targets_int[:, i], preds_binary[:, i], zero_division=0)
        r = recall_score(targets_int[:, i], preds_binary[:, i], zero_division=0)
        f1 = f1_score(targets_int[:, i], preds_binary[:, i], zero_division=0)
        metrics[f"{label}/precision"] = p
        metrics[f"{label}/recall"] = r
        metrics[f"{label}/f1"] = f1
        per_label_f1.append(f1)

    metrics["mean_f1"] = float(np.mean(per_label_f1))

    # Exact-match ratio: fraction of samples where *all* labels are correct.
    exact = np.all(preds_binary == targets_int, axis=1).astype(np.float32)
    metrics["exact_match"] = float(exact.mean())

    return metrics


# ---------------------------------------------------------------------------
# Training & evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:

    model.train()
    running_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    use_amp = device.type == "cuda"

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_preds.append(logits.detach().cpu().numpy())
        all_targets.append(labels.detach().cpu().numpy())

        if (batch_idx + 1) % max(1, len(dataloader) // 5) == 0:
            print(
                f"  batch {batch_idx + 1:>4}/{len(dataloader)}  "
                f"loss={loss.item():.4f}"
            )

    avg_loss = running_loss / len(dataloader.dataset)
    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return avg_loss, predictions, targets


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate the model on a dataloader without gradient computation.

    Returns:
        ``(avg_loss, all_predictions, all_targets)`` — same layout as
        :func:`train_one_epoch`.
    """
    model.eval()
    running_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    use_amp = device.type == "cuda"

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)

        running_loss += loss.item() * images.size(0)
        all_preds.append(logits.cpu().numpy())
        all_targets.append(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)
    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return avg_loss, predictions, targets


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def _print_epoch_summary(
    epoch: int,
    phase: str,
    train_loss: float,
    val_loss: float,
    metrics: dict[str, Any],
) -> None:
    """Print a formatted summary table for one epoch."""
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  {phase}  Epoch {epoch}  Summary")
    print(sep)
    print(f"  {'Train loss':<20s} {train_loss:.5f}")
    print(f"  {'Val   loss':<20s} {val_loss:.5f}")
    print(f"  {'Mean F1':<20s} {metrics['mean_f1']:.4f}")
    print(f"  {'Exact match':<20s} {metrics['exact_match']:.4f}")
    print(f"  {'-' * 46}")
    header = f"  {'Label':<16s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s}"
    print(header)
    print(f"  {'-' * 46}")
    for name in LABEL_NAMES:
        p = metrics.get(f"{name}/precision", 0.0)
        r = metrics.get(f"{name}/recall", 0.0)
        f1 = metrics.get(f"{name}/f1", 0.0)
        print(f"  {name:<16s} {p:>8.4f} {r:>8.4f} {f1:>8.4f}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Pos-weight computation
# ---------------------------------------------------------------------------

def _compute_pos_weight(dataset: MultiLabelBoxDataset, device: torch.device) -> torch.Tensor:
    """Compute ``pos_weight`` for ``BCEWithLogitsLoss``.

    ``pos_weight[j] = num_negatives_j / num_positives_j`` so that the
    minority-positive class is up-weighted.
    """
    weights = dataset.get_pos_weight()
    dist = dataset.get_label_distribution()
    print(f"Label distribution: {dist}")
    print(f"pos_weight per label: {dict(zip(LABEL_NAMES, weights.tolist()))}")
    return weights.to(device)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    model: nn.Module,
    epoch: int,
    best_f1: float,
    output_path: Path,
) -> None:
    """Persist a checkpoint dictionary to *output_path*."""
    ckpt = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_f1": best_f1,
        "backbone": backbone_name,
        "num_labels": len(LABEL_NAMES),
        "label_names": LABEL_NAMES,
        "threshold": 0.5,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output_path)
    print(f"  ✓ Saved checkpoint → {output_path}  (mean-F1={best_f1:.4f})")


def train(backbone_name: str, ckpt_name: str = "best_model.pt") -> dict:
    """Orchestrate the full two-phase training run.

    Parameters
    ----------
    backbone_name : str
        A key from ``model.BACKBONE_REGISTRY``.
    ckpt_name : str
        Filename for the best checkpoint inside ``output_dir``.

    Returns
    -------
    dict
        ``backbone_name``, ``best_f1``, ``total_params``, ``training_time_s``,
        ``model_size_mb``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    img_dir = Path(images_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_transform = get_train_transform(image_size)
    val_transform = get_val_transform(image_size)

    train_dataset = MultiLabelBoxDataset(
        images_dir=img_dir,
        labels_csv=Path(train_csv),
        transform=train_transform,
        split="train",
    )
    val_dataset = MultiLabelBoxDataset(
        images_dir=img_dir,
        labels_csv=Path(val_csv),
        transform=val_transform,
        split="val",
    )

    print(f"Train samples : {len(train_dataset)}")
    print(f"Val   samples : {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    pos_weight = _compute_pos_weight(train_dataset, device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = MultiLabelBoxClassifier(
        num_labels=len(LABEL_NAMES),
        dropout=dropout,
        backbone_name=backbone_name,
    ).to(device)

    info = get_model_info(model)
    print(f"Backbone      : {model.backbone_name}")
    print(f"Parameters    : {info['total_params']:,}  ({info['model_size_mb']} MB)")

    best_f1 = 0.0
    best_metrics = None
    best_ckpt_path = out_dir / ckpt_name
    global_epoch = 0

    training_start = time.perf_counter()

    # ======================= PHASE 1 – frozen backbone =======================
    if not skip_phase1:
        print("\n" + "=" * 68)
        print("  PHASE 1 — Frozen backbone, training classifier head only")
        print("=" * 68)

        freeze_backbone(model)

        optimizer_p1 = torch.optim.Adam(
            model.classifier.parameters(),
            lr=lr_head,
            weight_decay=weight_decay,
        )

        for epoch in range(1, epochs_phase1 + 1):
            global_epoch += 1
            t0 = time.perf_counter()

            train_loss, train_preds, train_targets = train_one_epoch(
                model, train_loader, loss_fn, optimizer_p1, device,
            )
            val_loss, val_preds, val_targets = evaluate(
                model, val_loader, loss_fn, device,
            )

            metrics = compute_metrics(val_preds, val_targets)
            elapsed = time.perf_counter() - t0

            _print_epoch_summary(epoch, "Phase-1", train_loss, val_loss, metrics)
            print(f"  ⏱  {elapsed:.1f}s")

            if metrics["mean_f1"] > best_f1:
                best_f1 = metrics["mean_f1"]
                best_metrics = metrics
                _save_checkpoint(model, global_epoch, best_f1, best_ckpt_path)
    else:
        print("\n⏭  Skipping Phase 1")

    print("\n" + "=" * 68)
    print("  PHASE 2 — Fine-tuning entire network")
    print("=" * 68)

    unfreeze_backbone(model)

    param_groups = [
        {
            "params": model.features.parameters(),
            "lr": lr_backbone,
        },
        {
            "params": model.classifier.parameters(),
            "lr": lr_head / 10,
        },
    ]

    optimizer_p2 = torch.optim.Adam(
        param_groups,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer_p2, T_max=epochs_phase2)

    for epoch in range(1, epochs_phase2 + 1):
        global_epoch += 1
        t0 = time.perf_counter()

        train_loss, train_preds, train_targets = train_one_epoch(
            model, train_loader, loss_fn, optimizer_p2, device,
        )
        val_loss, val_preds, val_targets = evaluate(
            model, val_loader, loss_fn, device,
        )
        scheduler.step()

        metrics = compute_metrics(val_preds, val_targets)
        elapsed = time.perf_counter() - t0

        _print_epoch_summary(epoch, "Phase-2", train_loss, val_loss, metrics)
        print(
            f"  ⏱  {elapsed:.1f}s  |  "
            f"LR backbone={scheduler.get_last_lr()[0]:.2e}  "
            f"head={scheduler.get_last_lr()[1]:.2e}"
        )

        if metrics["mean_f1"] > best_f1:
            best_f1 = metrics["mean_f1"]
            best_metrics = metrics
            _save_checkpoint(model, global_epoch, best_f1, best_ckpt_path)

    training_time_s = time.perf_counter() - training_start

    # ---- Inference benchmark -------------------------------------------------
    model.eval()
    dummy = torch.randn(batch_size, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
        t0 = time.perf_counter()
        for _ in range(100):
            model(dummy)
        elapsed = time.perf_counter() - t0
    inference_time_ms = round(elapsed / 100 / batch_size * 1000, 3)
    print(f"  Inference latency: {inference_time_ms:.3f} ms / sample")

    # ---- Final summary ------------------------------------------------------
    print("\n" + "=" * 68)
    print(f"  Training complete!  Best mean-F1 = {best_f1:.4f}")
    print(f"  Best checkpoint    → {best_ckpt_path}")
    print("=" * 68 + "\n")

    return {
        "backbone_name": backbone_name,
        "best_f1": best_f1,
        "best_metrics": best_metrics or {},
        "total_params": info["total_params"],
        "model_size_mb": info["model_size_mb"],
        "training_time_s": round(training_time_s, 1),
        "inference_time_ms": inference_time_ms,
    }


def main() -> None:
    result = train(backbone_name=backbone_name)
    print(f"Done. Best F1 = {result['best_f1']:.4f}  ({result['training_time_s']:.0f}s)")


if __name__ == "__main__":
    main()
