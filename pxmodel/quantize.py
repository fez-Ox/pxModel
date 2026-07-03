from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchao.quantization import (
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    quantize_,
)

from pxmodel.augmentation import LABEL_NAMES, get_val_transform
from pxmodel.config import *
from pxmodel.dataset_multilabel import MultiLabelBoxDataset
from pxmodel.model import MultiLabelBoxClassifier

warnings.filterwarnings("ignore", message=".*is deprecated.*")


def load_model(
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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    all_preds, all_targets = [], []
    for images, labels in dataloader:
        images = images.to(device)
        logits = model(images)
        sigmoids = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(sigmoids)
        all_targets.append(labels.numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    binary = (preds >= threshold).astype(np.int32)
    targets_int = targets.astype(np.int32)
    return {
        "exact_match": float(np.all(binary == targets_int, axis=1).mean()),
        "macro_f1": float(
            f1_score(targets_int, binary, average="macro", zero_division=0)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize with torchao")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--qat", action="store_true", help="Run QAT (int4)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else checkpoint
    has_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if has_gpu else "cpu")

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = load_model(ckpt_path, device)
    print(f"Model: {model.backbone_name}  |  Labels: {model.num_labels}")
    print(f"Device: {device}")

    val_tfm = get_val_transform(image_size)
    val_ds = MultiLabelBoxDataset(
        images_dir=images_dir,
        labels_csv=val_csv,
        transform=val_tfm,
        split="val",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if has_gpu else 0,
        pin_memory=has_gpu,
    )

    fp32_met = evaluate(model, val_loader, device)
    fp32_sz = model_size_mb(model)
    results = [("float32", fp32_met, fp32_sz)]

    fmt = "  {:<28s}  {:>8.4f}  {:>7.4f}  {:>7.2f}MB  {:+9.4f}  {:>5.1f}x"

    def add_result(label, m, loader, dev):
        met = evaluate(m, loader, dev)
        sz = model_size_mb(m)
        d = met["macro_f1"] - fp32_met["macro_f1"]
        r = fp32_sz / sz if sz > 0 else 0
        results.append((label, met, sz))
        print(fmt.format(label, met["macro_f1"], met["exact_match"], sz, d, r))

    # --- Int8 weight-only (version 2, no deprecation) ---
    print("\n" + "=" * 70)
    print("  INT8 WEIGHT-ONLY")
    m = load_model(ckpt_path, device)
    quantize_(m, Int8WeightOnlyConfig(version=2), device=device.type)
    add_result("int8 weight-only", m, val_loader, device)

    # --- Int8 dynamic activation + weight (version 2) ---
    print("\n" + "=" * 70)
    print("  INT8 DYNAMIC (activation + weight)")
    m = load_model(ckpt_path, device)
    quantize_(m, Int8DynamicActivationInt8WeightConfig(version=2), device=device.type)
    add_result("int8 dynamic act+wt", m, val_loader, device)

    # --- QAT (int4 weight-only) ---
    if args.qat:
        print("\n" + "=" * 70)
        print("  QAT (int4 weight-only)")
        from torchao.quantization import Int4WeightOnlyConfig
        from torchao.quantization.qat import QATConfig

        train_ds = MultiLabelBoxDataset(
            images_dir=images_dir,
            labels_csv=train_csv,
            transform=val_tfm,
            split="train",
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers if has_gpu else 0,
            pin_memory=has_gpu,
        )

        m = load_model(ckpt_path, device)
        base_cfg = Int4WeightOnlyConfig(group_size=32)
        quantize_(m, QATConfig(base_cfg, step="prepare"))

        loss_fn = nn.BCEWithLogitsLoss()
        optim = torch.optim.Adam(m.parameters(), lr=1e-4)
        for epoch in range(1, 4):
            m.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optim.zero_grad()
                loss_fn(m(images), labels).backward()
                optim.step()
            vm = evaluate(m, val_loader, device)
            print(f"  epoch {epoch}: macro-F1={vm['macro_f1']:.4f}")

        quantize_(m, QATConfig(base_cfg, step="convert"))
        add_result("QAT int4", m, val_loader, device)

    # --- Summary ---
    sep = "=" * 70
    print(f"\n{sep}")
    print(
        f"  {'Method':<28s}  {'Macro-F1':>8s}  {'Exact':>7s}  {'Size':>7s}  {'F1 Δ':>9s}  {'Ratio':>5s}"
    )
    print("-" * 70)
    for label, met, sz in results:
        d0 = met["macro_f1"] - fp32_met["macro_f1"] if label != "float32" else 0.0
        r0 = fp32_sz / sz if sz > 0 and label != "float32" else 1.0
        print(fmt.format(label, met["macro_f1"], met["exact_match"], sz, d0, r0))
    print(sep)


if __name__ == "__main__":
    main()
