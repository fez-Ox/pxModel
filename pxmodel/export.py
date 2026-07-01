"""Export the trained model to LiteRT (TFLite) format.

Uses ``ai-edge-torch`` for direct PyTorch → TFLite conversion
(no intermediate ONNX file).

Requires:
    pip install ai-edge-torch
"""

from __future__ import annotations

from pathlib import Path

import torch

from pxmodel.config import *
from pxmodel.model import MultiLabelBoxClassifier


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
) -> Path:
    """Export the model to TFLite via ``litert-torch``.

    Returns
    -------
    Path
        Path to the saved ``.tflite`` file.
    """
    try:
        import ai_edge_torch  # noqa: F401 — deprecated shim for litert-torch
    except ImportError:
        try:
            import litert_torch as ai_edge_torch  # noqa: F811
        except ImportError:
            raise ImportError(
                "litert-torch (or its deprecated alias ai-edge-torch) is "
                "required for TFLite export.\n"
                "  pip install litert-torch"
            )

    output_path = output_dir / f"{model.backbone_name}_multilabel.tflite"

    model = model.cpu()
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size)

    edge_model = ai_edge_torch.convert(model, (dummy_input,))
    edge_model.export(output_path)

    tflite_size = file_size_mb(output_path)
    print(f"\n  LiteRT (TFLite) model saved: {output_path}")
    print(f"    File size: {tflite_size:.2f} MB")

    return output_path


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = load_model_from_checkpoint(checkpoint, device)
    print(f"Model loaded from: {checkpoint}")
    print(f"Backbone: {model.backbone_name}  |  Labels: {model.num_labels}")

    # ── Create output directory ───────────────────────────────────────────
    export_dir.mkdir(parents=True, exist_ok=True)

    # ── Export ────────────────────────────────────────────────────────────
    summary_rows: list[tuple[str, Path]] = []
    summary_rows.append(("Original checkpoint", checkpoint))

    tflite_path = export_tflite(model, export_dir, image_size)
    summary_rows.append(("LiteRT (TFLite)", tflite_path))

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary_table(summary_rows)


if __name__ == "__main__":
    main()
