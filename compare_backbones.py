from __future__ import annotations
import time
from pathlib import Path
from augmentation import LABEL_NAMES
from config import output_dir
from model import BACKBONE_REGISTRY, get_model_info, MultiLabelBoxClassifier
from train import train


BACKBONES = [
    "efficientnet_b0",
    "efficientnet_b3",
    "efficientnet_b7",
    "mobilenet_v3_large",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
]

def main():
    print("=" * 74)
    print("   BACKBONE COMPARISON BENCHMARK")
    print("=" * 74)
    print(f"   Backbones to train: {', '.join(BACKBONES)}")
    print(f"   Output dir: {output_dir.resolve()}")
    print("=" * 74 + "\n")

    results: list[dict] = []
    total_start = time.perf_counter()

    for i, backbone in enumerate(BACKBONES, 1):
        if backbone not in BACKBONE_REGISTRY:
            print(f"  [SKIP] Unknown backbone {backbone!r}, skipping.")
            continue

        print(f"\n{'#' * 74}")
        print(f"  [{i}/{len(BACKBONES)}]  Training backbone: {backbone}")
        print(f"{'#' * 74}\n")

        ckpt_name = f"best_{backbone}.pt"
        t0 = time.perf_counter()
        result = train(backbone_name=backbone, ckpt_name=ckpt_name)
        elapsed = time.perf_counter() - t0

        result["training_time_s"] = round(elapsed, 1)
        results.append(result)

        ckpt_path = Path(output_dir) / ckpt_name
        if ckpt_path.exists():
            model_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
            result["checkpoint_mb"] = round(model_size_mb, 2)
        else:
            result["checkpoint_mb"] = 0.0

        print(f"\n  ✓ {backbone} done  |  "
              f"F1={result['best_f1']:.4f}  "
              f"params={result['total_params']:,}  "
              f"time={elapsed:.0f}s")

    if not results:
        print("\nNo results to show.")
        return

    total_wall = time.perf_counter() - total_start

    # Build column header dynamically based on label names
    label_f1_headers = [f"{n[:4]}-F1" for n in LABEL_NAMES]

    print("\n\n" + "=" * 80)
    print("   BACKBONE COMPARISON — RESULTS")
    print("=" * 80)

    header = (
        f"  {'Backbone':<22s} {'Params':>8s} {'Size':>6s} "
        + " ".join(f"{h:>7s}" for h in label_f1_headers)
        + f" {'Mean F1':>7s} {'Time(m)':>7s} {'Ckpt':>6s}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print(sep)
    print(header)
    print(sep)

    for r in results:
        param_str = f"{r['total_params'] / 1e6:.2f}M"
        size_str = f"{r['model_size_mb']:.1f}"
        f1_str = f"{r['best_f1']:.4f}"
        time_str = f"{r['training_time_s'] / 60:.1f}"
        ckpt_str = f"{r['checkpoint_mb']:.1f}"

        per_label = " ".join(
            f"{r['best_metrics'].get(f'{n}/f1', 0.0):>7.4f}" for n in LABEL_NAMES
        )

        print(
            f"  {r['backbone_name']:<22s} {param_str:>8s} {size_str:>6s} "
            f"{per_label}"
            f" {f1_str:>7s} {time_str:>7s} {ckpt_str:>6s}"
        )

    print(sep)
    print(f"  Total wall time: {total_wall / 60:.1f} minutes")
    print("=" * 80)


if __name__ == "__main__":
    main()
