"""Export PyTorch checkpoints to LiteRT (TFLite) format.

This module only exports existing ``.pt`` artifacts. It does not train or
quantize models. Both standard training checkpoints and torchao quantized
checkpoints are supported when the converter can lower the model.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# TFLite export is CPU-only. Disable CUDA before any converter (jax,
# litert-*) is imported so hosts with a CUDA toolkit don't fail GPU probes.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import torch

from pxmodel.config import checkpoint, export_dir, image_size
from pxmodel.model import MultiLabelBoxClassifier
from pxmodel.predict import load_checkpoint


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    backbone_name: str | None = None,
) -> MultiLabelBoxClassifier:
    """Load a five-class training or torchao-quantized checkpoint."""
    return load_checkpoint(checkpoint_path, device, backbone_name)


def file_size_mb(path: Path) -> float:
    """Return file size in MiB."""
    return path.stat().st_size / (1024 * 1024)


def print_summary_table(rows: list[tuple[str, Path]]) -> None:
    """Print a summary table of exported files and their sizes."""
    print("\n" + "=" * 60)
    print("  EXPORT SUMMARY")
    print("=" * 60)
    print(f"  {'Format':<25s} {'Size':>10s}  Path")
    print("-" * 60)
    for label, path in rows:
        size = file_size_mb(path)
        print(f"  {label:<25s} {size:>8.2f} MB  {path}")
    print("=" * 60)


def export_tflite(
    model: MultiLabelBoxClassifier,
    output_dir: Path,
    image_size: int,
    output_name: str | None = None,
) -> Path:
    """Export a loaded model to TFLite via ``litert-torch``.

    Parameters
    ----------
    model:
        Loaded PyTorch model.
    output_dir:
        Destination directory.
    image_size:
        Square image size used by the model.
    output_name:
        Optional filename stem or ``.tflite`` filename. Defaults to
        ``<backbone>_multilabel.tflite``.
    """
    try:
        import ai_edge_torch  # noqa: F401 — deprecated shim for litert-torch
    except ImportError:
        try:
            import litert_torch as ai_edge_torch  # noqa: F811
        except ImportError:
            raise ImportError(
                "litert-torch (or its deprecated alias ai-edge-torch) is "
                "required for TFLite export. Use `uv sync --locked --extra tflite` "
                "or run through ./export_all_backbones.sh."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    if output_name is None:
        output_filename = f"{model.backbone_name}_multilabel.tflite"
    else:
        output_filename = output_name if output_name.endswith(".tflite") else f"{output_name}.tflite"
    output_path = output_dir / output_filename

    model = model.cpu().eval()
    dummy_input = torch.randn(1, 3, image_size, image_size)

    edge_model = ai_edge_torch.convert(model, (dummy_input,))
    edge_model.export(output_path)

    tflite_size = file_size_mb(output_path)
    print(f"\n  LiteRT (TFLite) model saved: {output_path}")
    print(f"    File size: {tflite_size:.2f} MB")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a .pt checkpoint to TFLite")
    parser.add_argument("--checkpoint", type=Path, default=checkpoint)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=export_dir)
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional output filename/stem (default: derived from checkpoint)",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cpu")
    print(f"Device: {device}")

    model = load_model_from_checkpoint(args.checkpoint, device, args.backbone)
    print(f"Model loaded from: {args.checkpoint}")
    print(f"Backbone: {model.backbone_name}  |  Labels: {model.num_labels}")

    output_name = args.output_name or f"{args.checkpoint.stem}_multilabel"

    summary_rows: list[tuple[str, Path]] = [("Source checkpoint", args.checkpoint)]
    tflite_path = export_tflite(model, args.output_dir, image_size, output_name)
    summary_rows.append(("LiteRT (TFLite)", tflite_path))
    print_summary_table(summary_rows)


if __name__ == "__main__":
    main()
