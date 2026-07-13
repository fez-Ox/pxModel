"""ONNX Runtime inference with optional int8 dynamic quantization.

Exports a PyTorch checkpoint to ONNX, optionally quantizes to int8,
and runs inference via ONNX Runtime for faster CPU execution.

Usage:
    python -m pxmodel.predict_onnx --image <path> --checkpoint <path>
    python -m pxmodel.predict_onnx --image <path> --checkpoint <path> --quantize
    python -m pxmodel.predict_onnx --image <path> --onnx-cache model.onnx
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from pxmodel.augmentation import get_val_transform
from pxmodel.config import *
from pxmodel.labels import LABEL_NAMES, NUM_LABELS, require_current_label_count
from pxmodel.model import MultiLabelBoxClassifier


def load_fp32_model(
    checkpoint_path: str | Path,
    backbone_name: str | None = None,
) -> MultiLabelBoxClassifier:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict")
    if state_dict is None:
        raise ValueError(
            "Unsupported checkpoint format — expected a training checkpoint "
            "with 'model_state_dict' key. Quantized checkpoints cannot be "
            "exported to ONNX directly; use the original FP32 checkpoint."
        )
    backbone = backbone_name or ckpt.get("backbone", "efficientnet_b0")
    num_labels = ckpt.get("num_labels", NUM_LABELS)
    require_current_label_count(num_labels, f"Checkpoint {checkpoint_path}")
    model = MultiLabelBoxClassifier(
        num_labels=num_labels,
        backbone_name=backbone,
        pretrained=False,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_to_onnx(
    model: MultiLabelBoxClassifier,
    output_path: Path,
    image_size: int = 224,
) -> Path:
    import warnings
    model.cpu().eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes=None,
            opset_version=18,
            dynamo=False,
        )
    return output_path


def quantize_onnx(onnx_path: Path) -> Path:
    import warnings
    from onnxruntime.quantization import quantize_dynamic, QuantType

    warnings.warn(
        "int8 quantization may reduce model accuracy significantly. "
        "FP32 ONNX Runtime is recommended for best speed without accuracy loss."
    )

    quantized_path = onnx_path.with_stem(onnx_path.stem + "_int8")
    quantize_dynamic(
        onnx_path.as_posix(),
        quantized_path.as_posix(),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv"],
    )
    return quantized_path


def run_ort(
    onnx_path: Path,
    image: np.ndarray,
    transform,
) -> tuple[np.ndarray, float]:
    import onnxruntime

    tensor = transform(image=image)["image"].unsqueeze(0).numpy()

    session = onnxruntime.InferenceSession(
        onnx_path.as_posix(),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    start = time.perf_counter()
    logits = session.run(None, {input_name: tensor})[0]
    elapsed = time.perf_counter() - start

    if logits.ndim != 2 or logits.shape != (1, NUM_LABELS):
        raise ValueError(
            f"ONNX model output shape is {logits.shape}; expected (1, {NUM_LABELS}) "
            "for the current five-class label schema. Re-export the model."
        )

    sigmoids = 1.0 / (1.0 + np.exp(-logits)).squeeze(0)
    return sigmoids, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ONNX Runtime inference with optional int8 quantization",
    )
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="PyTorch checkpoint (FP32, with 'model_state_dict' key)",
    )
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument(
        "--onnx-cache",
        type=str,
        default=None,
        help="Path to cache/load ONNX file (skips export if exists)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize ONNX model to int8 before inference",
    )
    parser.add_argument(
        "--keep-onnx",
        action="store_true",
        help="Keep the ONNX file after inference (default: delete temp files)",
    )
    args = parser.parse_args()

    print(f"Device: CPU (ONNX Runtime{' int8' if args.quantize else ' FP32'})")

    # ── Export or load ONNX ──────────────────────────────────────────────
    if args.onnx_cache:
        onnx_path = Path(args.onnx_cache)
        if not onnx_path.is_file():
            print(f"Exporting checkpoint to ONNX: {onnx_path}")
            model = load_fp32_model(args.checkpoint, args.backbone)
            export_to_onnx(model, onnx_path, image_size)
            print(f"  Backbone: {model.backbone_name}  |  Labels: {model.num_labels}")
            print(f"  Size: {onnx_path.stat().st_size / 1024**2:.2f} MB")
        else:
            print(f"Using cached ONNX: {onnx_path}")
    else:
        model = load_fp32_model(args.checkpoint, args.backbone)
        tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        onnx_path = Path(tmp.name)
        print(f"Exporting to temporary ONNX: {onnx_path}")
        export_to_onnx(model, onnx_path, image_size)
        print(f"  Backbone: {model.backbone_name}  |  Labels: {model.num_labels}")
        print(f"  Size: {onnx_path.stat().st_size / 1024**2:.2f} MB")

    # ── Quantize (optional) ─────────────────────────────────────────────
    if args.quantize:
        print("Quantizing ONNX model to int8 (dynamic)...")
        onnx_path = quantize_onnx(onnx_path)
        print(f"  Quantized size: {onnx_path.stat().st_size / 1024**2:.2f} MB")

    # ── Load image ──────────────────────────────────────────────────────
    image_bgr = cv2.imread(str(args.image))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    val_transform = get_val_transform(image_size=image_size)

    # ── Inference ───────────────────────────────────────────────────────
    confidences, latency = run_ort(onnx_path, image_rgb, val_transform)

    print(f"\nONNX Runtime inference time: {latency:.4f}s")

    # ── Results ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("  PREDICTION")
    print("=" * 60)
    thresholds = [float(threshold)] * len(LABEL_NAMES)
    for i, name in enumerate(LABEL_NAMES):
        conf = confidences[i]
        tag = "YES" if conf >= thresholds[i] else "NO "
        print(f"    {name:<15s}  {tag}  (conf: {conf:.4f})")
    print("=" * 60)

    # ── Cleanup ─────────────────────────────────────────────────────────
    if not args.keep_onnx and not args.onnx_cache:
        if args.quantize:
            # Delete both FP32 and quantized temp files
            fp32_path = onnx_path.with_stem(
                onnx_path.stem.replace("_int8", "")
            )
            if fp32_path.exists():
                fp32_path.unlink()
        onnx_path.unlink()


if __name__ == "__main__":
    main()
