from __future__ import annotations
import time
from datetime import datetime
from pathlib import Path

from pxmodel.config import batch_size, dropout, epochs_phase1, epochs_phase2, image_size, output_dir
from pxmodel.labels import LABEL_NAMES
from pxmodel.model import BACKBONE_REGISTRY
from pxmodel.train import train


BACKBONES = [
    "efficientnet_b0",
    "efficientnet_b3",
    "efficientnet_b7",
    "mobilenet_v3_large",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
]


def _format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m"
    return f"{m}m {s}s"


def main() -> None:
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
              f"inf={result['inference_time_ms']:.1f}ms  "
              f"time={elapsed:.0f}s")

    if not results:
        print("\nNo results to show.")
        return

    total_wall = time.perf_counter() - total_start

    label_f1_headers = [f"{n[:4]}-F1" for n in LABEL_NAMES]

    print("\n\n" + "=" * 90)
    print("   BACKBONE COMPARISON — SUMMARY")
    print("=" * 90)

    header = (
        f"  {'Backbone':<22s} {'Params':>8s} {'Size':>6s} "
        + " ".join(f"{h:>7s}" for h in label_f1_headers)
        + f" {'Mean F1':>7s} {'Inf(ms)':>8s} {'Time':>8s} {'Ckpt':>6s}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print(sep)
    print(header)
    print(sep)

    for r in results:
        param_str = f"{r['total_params'] / 1e6:.2f}M"
        size_str = f"{r['model_size_mb']:.1f}"
        f1_str = f"{r['best_f1']:.4f}"
        inf_str = f"{r['inference_time_ms']:.1f}"
        time_str = _format_duration(r['training_time_s'])
        ckpt_str = f"{r['checkpoint_mb']:.1f}"

        per_label = " ".join(
            f"{r['best_metrics'].get(f'{n}/f1', 0.0):>7.4f}" for n in LABEL_NAMES
        )

        print(
            f"  {r['backbone_name']:<22s} {param_str:>8s} {size_str:>6s} "
            f"{per_label}"
            f" {f1_str:>7s} {inf_str:>8s} {time_str:>8s} {ckpt_str:>6s}"
        )

    print(sep)
    print(f"  Total wall time: {_format_duration(total_wall)}")
    print("=" * 90)

    # ------------------------------------------------------------------
    # Console: per-backbone detail (precision / recall / F1)
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("   PER-BACKBONE DETAILS  (Precision / Recall / F1)")
    print("=" * 90)

    for r in results:
        name = r['backbone_name']
        print(f"\n  {name}")
        print(f"  {'Label':<16s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s}")
        print(f"  {'-' * 44}")
        for label in LABEL_NAMES:
            p = r['best_metrics'].get(f"{label}/precision", 0.0)
            r_val = r['best_metrics'].get(f"{label}/recall", 0.0)
            f = r['best_metrics'].get(f"{label}/f1", 0.0)
            print(f"  {label:<16s} {p:>8.4f} {r_val:>8.4f} {f:>8.4f}")

    # ------------------------------------------------------------------
    # Markdown export
    # ------------------------------------------------------------------
    md_path = Path(output_dir) / "comparison_results.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Backbone Comparison Results",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        "",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| Image size | {image_size} |",
        f"| Batch size | {batch_size} |",
        f"| Epochs (phase 1) | {epochs_phase1} |",
        f"| Epochs (phase 2) | {epochs_phase2} |",
        f"| Dropout | {dropout} |",
        "",
        "## Summary",
        "",
    ]

    # Summary table
    md_cols = ["Backbone", "Params", "Size (MB)", "Mean F1", "Inf (ms)", "Time", "Ckpt (MB)"]
    lines.append("| " + " | ".join(md_cols) + " |")
    lines.append("|" + "|".join("---" for _ in md_cols) + "|")

    for r in results:
        param_str = f"{r['total_params'] / 1e6:.2f}M"
        size_str = f"{r['model_size_mb']:.1f}"
        f1_str = f"{r['best_f1']:.4f}"
        inf_str = f"{r['inference_time_ms']:.1f}"
        time_str = _format_duration(r['training_time_s'])
        ckpt_str = f"{r['checkpoint_mb']:.1f}"
        lines.append(
            f"| {r['backbone_name']} | {param_str} | {size_str} | "
            f"{f1_str} | {inf_str} | {time_str} | {ckpt_str} |"
        )

    lines.extend(["", "## Per-backbone Details", ""])

    for r in results:
        lines.append(f"### {r['backbone_name']}")
        lines.append("")
        lines.append("| Label | Precision | Recall | F1 |")
        lines.append("|---|---|---|---|")
        for label in LABEL_NAMES:
            p = r['best_metrics'].get(f"{label}/precision", 0.0)
            r_val = r['best_metrics'].get(f"{label}/recall", 0.0)
            f = r['best_metrics'].get(f"{label}/f1", 0.0)
            lines.append(f"| {label} | {p:.4f} | {r_val:.4f} | {f:.4f} |")
        lines.append("")

    md_path.write_text("\n".join(lines))
    print(f"\n  Results saved → {md_path.resolve()}")
    print()


if __name__ == "__main__":
    main()
