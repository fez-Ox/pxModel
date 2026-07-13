"""Benchmark inference methods across backbones and devices.

Measures latency for PyTorch baseline, ``torch.compile`` (inductor),
``torch.compile`` with TensorRT backend, and ONNX Runtime (FP32 / int8
dynamic) on CPU or CUDA.

Usage:
    # All available checkpoints, all methods (auto-detect from checkpoints/)
    python -m pxmodel.benchmark

    # Specific backbone (searches checkpoints/best_<backbone>.pt)
    python -m pxmodel.benchmark --backbone convnext_base

    # Custom checkpoint with explicit methods
    python -m pxmodel.benchmark \\
        --checkpoint checkpoints/best_efficientnet_b0.pt \\
        --methods baseline,compile,tensorrt,onnx

    # Force CPU even if CUDA is available
    python -m pxmodel.benchmark --cpu

    # More iterations for stable timing
    python -m pxmodel.benchmark --iterations 100

Requirements:
    pip install onnx onnxruntime
    pip install torch_tensorrt onnxruntime-gpu  # for TensorRT
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from pxmodel.config import *
from pxmodel.labels import NUM_LABELS, require_current_label_count
from pxmodel.model import (
    BACKBONE_REGISTRY,
    MultiLabelBoxClassifier,
    get_model_info,
)

HERE = Path(__file__).resolve().parent
CHECKPOINTS_DIR = HERE.parent / "checkpoints"
ONNX_CACHE_DIR = HERE.parent / "exported_models"

METHODS_ALL = ("baseline", "compile", "tensorrt", "onnx", "onnx-int8")


# ── Helpers ────────────────────────────────────────────────────────────


def _infer_backbone(checkpoint_path: Path) -> str | None:
    stem = checkpoint_path.stem
    for name in BACKBONE_REGISTRY:
        if name in stem:
            return name
    return None


def _detect_checkpoints(
    checkpoint: str | None,
    backbone: str | None,
) -> list[tuple[Path, str]]:
    if checkpoint:
        p = Path(checkpoint)
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        inferred = backbone or _infer_backbone(p)
        if not inferred:
            raise ValueError(
                f"Cannot infer backbone from {p.name}. "
                f"Use --backbone to specify."
            )
        return [(p, inferred)]

    if backbone:
        p = CHECKPOINTS_DIR / f"best_{backbone}.pt"
        if not p.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {p}. "
                f"Specify a custom path with --checkpoint."
            )
        return [(p, backbone)]

    results: list[tuple[Path, str]] = []
    for f in sorted(CHECKPOINTS_DIR.glob("best_*.pt")):
        name = _infer_backbone(f)
        if name:
            results.append((f, name))
    if not results:
        raise FileNotFoundError(
            f"No best_*.pt checkpoints found in {CHECKPOINTS_DIR}."
        )
    return results


def _load_model(path: Path, backbone: str, device: torch.device) -> MultiLabelBoxClassifier:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))
    if state_dict is None:
        raise ValueError(f"Unrecognised checkpoint format: {path}")
    num_labels = ckpt.get("num_labels", NUM_LABELS)
    require_current_label_count(num_labels, f"Checkpoint {path}")
    model = MultiLabelBoxClassifier(
        num_labels=num_labels,
        backbone_name=backbone,
        pretrained=False,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        warnings.warn(f"Unexpected keys in {path}: {unexpected[:5]}...")
    model.to(device)
    model.eval()
    return model


def _make_input(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, 3, image_size, image_size, device=device).contiguous()


def _measure(
    fn,
    x: torch.Tensor,
    n_warmup: int,
    n_iter: int,
    device: torch.device,
) -> tuple[float, float]:
    use_cuda = device.type == "cuda"
    for _ in range(n_warmup):
        fn(x)
    if use_cuda:
        torch.cuda.synchronize(device)
    timings = np.empty(n_iter)
    for i in range(n_iter):
        if use_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn(x)
        if use_cuda:
            torch.cuda.synchronize(device)
        timings[i] = time.perf_counter() - t0
    return float(timings.mean() * 1000), float(timings.std() * 1000)


# ── Method runners ─────────────────────────────────────────────────────


def benchmark_baseline(
    model: MultiLabelBoxClassifier,
    x: torch.Tensor,
    n_warmup: int,
    n_iter: int,
    device: torch.device,
) -> tuple[float, float]:
    @torch.no_grad()
    def fn(t):
        return model(t)

    return _measure(lambda t: fn(t), x, n_warmup, n_iter, device)


def benchmark_compile(
    model: MultiLabelBoxClassifier,
    x: torch.Tensor,
    n_warmup: int,
    n_iter: int,
    device: torch.device,
) -> tuple[float, float]:
    compiled = torch.compile(model)

    @torch.no_grad()
    def fn(t):
        return compiled(t)

    # Warmup inside no_grad to match timing context (avoids recompile)
    with torch.no_grad():
        compiled(x)
    return _measure(fn, x, n_warmup, n_iter, device)


def benchmark_tensorrt(
    model: MultiLabelBoxClassifier,
    x: torch.Tensor,
    n_warmup: int,
    n_iter: int,
    device: torch.device,
) -> tuple[float, float]:
    try:
        import torch_tensorrt  # registered as a torch.compile backend
    except ImportError:
        raise RuntimeError(
            "torch_tensorrt is required for TensorRT benchmarking.\n"
            "  pip install torch_tensorrt"
        )

    if device.type != "cuda":
        raise RuntimeError("TensorRT requires a CUDA device")

    compiled = torch.compile(model, backend="tensorrt")

    @torch.no_grad()
    def fn(t):
        return compiled(t)

    print(f"building engine...", end=" ", flush=True)
    build_start = time.perf_counter()
    with torch.no_grad():
        compiled(x)
    build_elapsed = time.perf_counter() - build_start
    print(f"({build_elapsed:.1f}s)", end=" ", flush=True)

    return _measure(fn, x, n_warmup, n_iter, device)


def benchmark_onnx(
    model: MultiLabelBoxClassifier,
    x: torch.Tensor,
    n_warmup: int,
    n_iter: int,
    device: torch.device,
    backbone: str,
    *,
    quantize: bool = False,
) -> tuple[float, float]:
    import onnxruntime

    ONNX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    suffix = "_int8.onnx" if quantize else ".onnx"
    onnx_path = ONNX_CACHE_DIR / f"{backbone}{suffix}"

    if not onnx_path.is_file():
        # Export to ONNX (always on CPU)
        model_cpu = model.cpu().eval()
        fp32_path = ONNX_CACHE_DIR / f"{backbone}.onnx"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                model_cpu,
                torch.randn(1, 3, image_size, image_size),
                fp32_path,
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes=None,
                opset_version=18,
                dynamo=False,
            )

        if quantize:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantize_dynamic(
                fp32_path.as_posix(),
                onnx_path.as_posix(),
                weight_type=QuantType.QInt8,
                op_types_to_quantize=["Conv"],
            )
    else:
        print(f"(cached)", end=" ", flush=True)

    # Build ORT session — prefer TensorRT EP on CUDA, fall back through CUDA → CPU
    if device.type == "cuda":
        providers = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]
    session = onnxruntime.InferenceSession(onnx_path.as_posix(), providers=providers)

    input_name = session.get_inputs()[0].name
    x_np = x.cpu().numpy()

    def fn(arr):
        session.run(None, {input_name: arr})

    # Warmup (ORT doesn't need synchronize)
    for _ in range(n_warmup):
        fn(x_np)

    timings = np.empty(n_iter)
    for i in range(n_iter):
        t0 = time.perf_counter()
        fn(x_np)
        timings[i] = time.perf_counter() - t0

    mean_ms = float(timings.mean() * 1000)
    std_ms = float(timings.std() * 1000)
    return mean_ms, std_ms


# ── Reporting ──────────────────────────────────────────────────────────


def _print_results(
    backbone: str,
    info: dict,
    device: torch.device,
    rows: list[tuple[str, float, float, float]],
) -> None:
    print()
    print(f"  Backbone: {backbone}  |  "
          f"Params: {info['total_params'] / 1e6:.1f}M  |  "
          f"Device: {device}")
    print()
    header = f"  {'Method':<20s} {'Mean (ms)':>10s} {'Std (ms)':>9s} {'Speedup':>8s}"
    print(header)
    print("  " + "-" * len(header))
    baseline_ms = next((r[1] for r in rows if r[0] == "baseline"), rows[0][1])
    for label, mean_ms, std_ms, _ in rows:
        speedup = baseline_ms / mean_ms if mean_ms > 0 else 0
        print(f"  {label:<20s} {mean_ms:>8.2f}    {std_ms:>6.2f}   {speedup:>5.2f}x")
    print()


# ── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark inference methods across backbones and devices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(METHODS_ALL),
        help=f"Comma-separated methods: {', '.join(METHODS_ALL)} (default: all)",
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA available")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",")]
    for m in methods:
        if m not in METHODS_ALL:
            raise ValueError(f"Unknown method {m!r}. Available: {METHODS_ALL}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Iterations: {args.iterations}  |  Warmup: {args.warmup}")

    entries = _detect_checkpoints(args.checkpoint, args.backbone)

    for ckpt_path, backbone in entries:
        print(f"\n{'=' * 60}")
        print(f"  Loading: {ckpt_path}")
        model = _load_model(ckpt_path, backbone, device)
        info = get_model_info(model)

        x = _make_input(1, device)
        rows: list[tuple[str, float, float, float]] = []

        if "baseline" in methods:
            print(f"  Running: baseline ...", end=" ", flush=True)
            mean_ms, std_ms = benchmark_baseline(
                model, x, args.warmup, args.iterations, device
            )
            print(f"{mean_ms:.1f}ms")
            rows.append(("baseline", mean_ms, std_ms, 0.0))

        if "compile" in methods:
            print(f"  Running: compile ...", end=" ", flush=True)
            mean_ms, std_ms = benchmark_compile(
                model, x, args.warmup, args.iterations, device
            )
            print(f"{mean_ms:.1f}ms")
            rows.append(("compile", mean_ms, std_ms, 0.0))

        if "tensorrt" in methods:
            if device.type != "cuda":
                print("  tensorrt: skipped (requires CUDA device)")
            else:
                try:
                    import torch_tensorrt  # noqa: F401 — registers the backend
                    print(f"  Running: tensorrt ...", end=" ", flush=True)
                    mean_ms, std_ms = benchmark_tensorrt(
                        model, x, args.warmup, args.iterations, device
                    )
                    print(f"{mean_ms:.1f}ms")
                    rows.append(("tensorrt", mean_ms, std_ms, 0.0))
                except ImportError:
                    print("  tensorrt: skipped (torch_tensorrt not installed)")
                except Exception as e:
                    print(f"  tensorrt: skipped ({e})")

        if "onnx" in methods:
            print(f"  Running: onnx FP32 ...", end=" ", flush=True)
            mean_ms, std_ms = benchmark_onnx(
                model, x, args.warmup, args.iterations, device, backbone, quantize=False
            )
            print(f"{mean_ms:.1f}ms")
            rows.append(("onnx FP32", mean_ms, std_ms, 0.0))

        if "onnx-int8" in methods:
            print(f"  Running: onnx int8 ...", end=" ", flush=True)
            mean_ms, std_ms = benchmark_onnx(
                model, x, args.warmup, args.iterations, device, backbone, quantize=True
            )
            print(f"{mean_ms:.1f}ms")
            rows.append(("onnx int8", mean_ms, std_ms, 0.0))

        _print_results(backbone, info, device, rows)


if __name__ == "__main__":
    main()
