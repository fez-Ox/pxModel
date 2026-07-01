"""Quantization pipeline for MultiLabelBoxClassifier.

Usage:
    python -m pxmodel.quantize              # dynamic + static (default)
    python -m pxmodel.quantize --dynamic    # dynamic only
    python -m pxmodel.quantize --static     # static only
    python -m pxmodel.quantize --qat        # QAT + fine-tune
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from pxmodel.augmentation import LABEL_NAMES, get_val_transform
from pxmodel.config import *
from pxmodel.dataset_multilabel import MultiLabelBoxDataset
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


def model_size_mb(model: nn.Module) -> float:
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(model.state_dict(), f.name)
        return Path(f.name).stat().st_size / (1024 * 1024)


def _make_loader(
    dataset: torch.utils.data.Dataset,
    shuffle: bool = False,
    use_gpu: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers if use_gpu else 0,
        pin_memory=use_gpu,
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for images, labels in dataloader:
        images = images.to(device)
        logits = model(images)
        sigmoids = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(sigmoids)
        all_targets.append(labels.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    binary = (preds >= 0.5).astype(np.int32)
    targets_int = targets.astype(np.int32)

    exact_match = np.all(binary == targets_int, axis=1).mean()
    macro_f1 = float(f1_score(targets_int, binary, average="macro", zero_division=0))
    sample_f1 = float(f1_score(targets_int, binary, average="samples", zero_division=0))

    return {"exact_match": exact_match, "macro_f1": macro_f1, "sample_f1": sample_f1}


def print_comparison(name: str, metrics: dict, size_mb: float) -> None:
    print(
        f"  {name:<25s}  "
        f"macro-F1={metrics['macro_f1']:.4f}  "
        f"exact={metrics['exact_match']:.4f}  "
        f"size={size_mb:.2f} MB"
    )


# ---------------------------------------------------------------------------
# Dynamic quantization
# ---------------------------------------------------------------------------


def quantize_dynamic(
    model: MultiLabelBoxClassifier,
    dtype: torch.dtype = torch.qint8,
) -> nn.Module:
    """Apply post-training dynamic quantization (weights → int8).

    Only ``nn.Linear`` layers are quantized.  The model is moved to CPU.
    """
    model.cpu()
    model.eval()
    quantized = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=dtype,
    )
    return quantized


# ---------------------------------------------------------------------------
# Static quantization (FX Graph mode)
# ---------------------------------------------------------------------------


def quantize_static(
    model: MultiLabelBoxClassifier,
    calibration_loader: DataLoader,
    backend: str = "x86",
) -> nn.Module:
    """Apply post-training static quantization (weights + activations → int8).

    Requires a calibration dataloader to determine activation ranges.
    Uses FX graph mode (``torch.ao.quantization.quantize_fx``).
    """
    try:
        from torch.ao.quantization import get_default_qconfig
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    except ImportError:
        raise ImportError("FX quantization requires PyTorch >= 1.13")

    model = model.cpu()
    model.eval()

    qconfig = get_default_qconfig(backend)

    example_inputs = next(iter(calibration_loader))[0][:1]
    qconfig_dict = {"": qconfig}

    prepared = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)

    with torch.no_grad():
        for images, _ in calibration_loader:
            prepared(images)

    quantized = convert_fx(prepared)
    return quantized


# ---------------------------------------------------------------------------
# QAT — Quantization-Aware Training
# ---------------------------------------------------------------------------


def quantize_qat(
    model: MultiLabelBoxClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-4,
    backend: str = "x86",
) -> nn.Module:
    """Apply quantization-aware training.

    The model is prepared with fake-quantize nodes, fine-tuned for
    *epochs*, then converted to a statically quantized model.
    """
    try:
        from torch.ao.quantization import get_default_qconfig
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx
    except ImportError:
        raise ImportError("FX quantization requires PyTorch >= 1.13")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    qconfig = get_default_qconfig(backend)
    example_inputs = next(iter(train_loader))[0][:1].to(device)
    qconfig_dict = {"": qconfig}

    qat_model = prepare_qat_fx(model, qconfig_dict, example_inputs=example_inputs)
    qat_model.train()

    optimizer = torch.optim.Adam(qat_model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = qat_model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        val_metrics = evaluate_model(qat_model, val_loader, device)
        print(
            f"  QAT epoch {epoch:>2}/{epochs}  loss={avg_loss:.4f}  "
            f"macro-F1={val_metrics['macro_f1']:.4f}"
        )

    qat_model.cpu()
    qat_model.eval()
    quantized = convert_fx(qat_model)
    return quantized


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantization pipeline for MultiLabelBoxClassifier"
    )
    parser.add_argument("--dynamic", action="store_true", help="Dynamic only")
    parser.add_argument("--static", action="store_true", help="Static only")
    parser.add_argument("--qat", action="store_true", help="QAT + fine-tune")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override checkpoint path "
             "(default: checkpoints/best_model.pt from config)",
    )
    args = parser.parse_args()

    run_all = not any([args.dynamic, args.static, args.qat])
    ckpt_path = Path(args.checkpoint) if args.checkpoint else checkpoint
    has_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if has_gpu else "cpu")

    print(f"Device: {device}")
    print(f"Backend: {backend}")
    print(f"Checkpoint: {ckpt_path}")

    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Place your .pt file at this path or use --checkpoint."
        )

    # ---- Load original model ------------------------------------------------
    model = load_model_from_checkpoint(ckpt_path, device)
    print(f"Model: {model.backbone_name}  |  Labels: {model.num_labels}")

    val_transform = get_val_transform(image_size)
    calib_dataset = MultiLabelBoxDataset(
        images_dir=images_dir,
        labels_csv=val_csv,
        transform=val_transform,
        split="val",
    )
    calib_loader = _make_loader(calib_dataset, use_gpu=has_gpu)

    # ---- Baseline metrics ---------------------------------------------------
    print("\n" + "=" * 60)
    print("  BASELINE (float32)")
    print("=" * 60)
    fp32_metrics = evaluate_model(model, calib_loader, device)
    fp32_size = model_size_mb(model)
    print_comparison("float32", fp32_metrics, fp32_size)

    results: list[tuple[str, nn.Module, dict, float]] = []

    # ---- Dynamic ------------------------------------------------------------
    if run_all or args.dynamic:
        print("\n" + "=" * 60)
        print("  DYNAMIC QUANTIZATION (int8 weights)")
        print("=" * 60)
        dyn = quantize_dynamic(model)
        dyn_metrics = evaluate_model(dyn, calib_loader, torch.device("cpu"))
        dyn_size = model_size_mb(dyn)
        print_comparison("dynamic int8", dyn_metrics, dyn_size)
        results.append(("dynamic_int8", dyn, dyn_metrics, dyn_size))

    # ---- Static -------------------------------------------------------------
    if run_all or args.static:
        print("\n" + "=" * 60)
        print("  STATIC QUANTIZATION (int8 weights + activations)")
        print("=" * 60)

        n_calib = min(calibration_samples, len(calib_dataset))
        calib_subset = torch.utils.data.Subset(
            calib_dataset, list(range(n_calib))
        )
        calib_sub_loader = DataLoader(
            calib_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # calibration runs on CPU, keep workers=0
        )
        print(f"Calibration samples: {n_calib}")

        # Load from disk into CPU for static quantization (int8 kernels only on CPU)
        model_cpu = load_model_from_checkpoint(ckpt_path, torch.device("cpu"))
        stat = quantize_static(model_cpu, calib_sub_loader, backend)
        stat_metrics = evaluate_model(stat, calib_loader, torch.device("cpu"))
        stat_size = model_size_mb(stat)
        print_comparison("static int8", stat_metrics, stat_size)
        results.append(("static_int8", stat, stat_metrics, stat_size))

    # ---- QAT ----------------------------------------------------------------
    if run_all or args.qat:
        print("\n" + "=" * 60)
        print("  QUANTIZATION-AWARE TRAINING")
        print("=" * 60)

        train_dataset = MultiLabelBoxDataset(
            images_dir=images_dir,
            labels_csv=train_csv,
            transform=get_val_transform(image_size),  # no augment for QAT
            split="train",
        )
        train_loader = _make_loader(train_dataset, shuffle=True, use_gpu=has_gpu)

        qat_model = quantize_qat(
            load_model_from_checkpoint(ckpt_path, device),
            train_loader,
            calib_loader,
            epochs=5,
        )
        qat_metrics = evaluate_model(qat_model, calib_loader, torch.device("cpu"))
        qat_size = model_size_mb(qat_model)
        print_comparison("QAT int8", qat_metrics, qat_size)
        results.append(("QAT_int8", qat_model, qat_metrics, qat_size))

    # ---- Summary table ------------------------------------------------------
    print("\n" + "=" * 70)
    print("  QUANTIZATION SUMMARY")
    print("=" * 70)
    print(
        f"  {'Method':<25s} {'Macro-F1':>9s} {'Exact':>7s} {'Size':>7s} {'ΔF1':>7s} {'Ratio':>7s}"
    )
    print("-" * 70)
    for name, _, met, sz in results:
        df1 = met["macro_f1"] - fp32_metrics["macro_f1"]
        ratio = fp32_size / sz if sz > 0 else 0
        print(
            f"  {name:<25s} {met['macro_f1']:>9.4f} {met['exact_match']:>7.4f} "
            f"{sz:>6.2f}MB {df1:>+7.4f} {ratio:>6.1f}x"
        )
    print("=" * 70)

    # ---- Save best quantized model ------------------------------------------
    if results:
        best_idx = int(np.argmax([r[2]["macro_f1"] for r in results]))
        best_name, best_model, best_met, _ = results[best_idx]
        quantized_output.parent.mkdir(parents=True, exist_ok=True)

        ckpt = {
            "quantized_state_dict": best_model.state_dict(),
            "method": best_name,
            "backend": backend,
            "num_labels": LABEL_NAMES,
            "original_backbone": model.backbone_name,
            "baseline_f1": fp32_metrics["macro_f1"],
            "quantized_f1": best_met["macro_f1"],
        }
        torch.save(ckpt, quantized_output)
        print(f"\n  Saved best quantized model → {quantized_output}")
        print(f"  Method: {best_name}  |  Macro-F1: {best_met['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
