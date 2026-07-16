from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from pxmodel.config import export_dir, image_size, output_dir
from pxmodel.export import export_tflite, load_model_from_checkpoint
from pxmodel.model import BACKBONE_REGISTRY

DEFAULT_QUANTIZED_DIR = Path(output_dir) / "quantized"


def _format_duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"


def _discover_training_checkpoints() -> list[Path]:
    paths: list[Path] = []
    for backbone in BACKBONE_REGISTRY:
        path = Path(output_dir) / f"best_{backbone}.pt"
        if path.is_file():
            paths.append(path)
    return paths


def _discover_quantized_checkpoints(quantized_dir: Path) -> list[Path]:
    if not quantized_dir.is_dir():
        return []
    return sorted(quantized_dir.glob("*.pt"))


def _infer_backbone_from_name(path: Path) -> str | None:
    for backbone in sorted(BACKBONE_REGISTRY, key=len, reverse=True):
        if backbone in path.stem:
            return backbone
    return None


def _write_summary(results: list[dict[str, Any]], destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "export_all_results.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        default=None,
        help="Explicit .pt checkpoints to export. Overrides discovery.",
    )
    parser.add_argument(
        "--include",
        choices=("trained", "quantized", "all"),
        default="all",
        help="Checkpoint sets to discover when --checkpoints is not provided",
    )
    parser.add_argument(
        "--quantized-dir",
        type=Path,
        default=DEFAULT_QUANTIZED_DIR,
        help=f"Directory containing quantized .pt files (default: {DEFAULT_QUANTIZED_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=export_dir,
        help=f"TFLite output directory (default: {export_dir})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip checkpoints whose target .tflite file already exists",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately instead of continuing after an export failure",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.checkpoints is not None:
        checkpoints = args.checkpoints
    else:
        checkpoints = []
        if args.include in {"trained", "all"}:
            checkpoints.extend(_discover_training_checkpoints())
        if args.include in {"quantized", "all"}:
            checkpoints.extend(_discover_quantized_checkpoints(args.quantized_dir))

    # Preserve discovery order while dropping duplicates.
    checkpoints = list(dict.fromkeys(path.resolve() for path in checkpoints))
    if not checkpoints:
        raise FileNotFoundError(
            "No .pt checkpoints found to export. Run training and/or quantization first, "
            "or pass explicit files with --checkpoints."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  EXPORT PYTORCH CHECKPOINTS TO TFLITE")
    print("=" * 78)
    print(f"  Checkpoints: {len(checkpoints)}")
    print(f"  Output:      {args.output_dir.resolve()}")
    print("=" * 78)

    results: list[dict[str, Any]] = []
    total_started = time.perf_counter()

    for index, checkpoint_path in enumerate(checkpoints, start=1):
        started = time.perf_counter()
        output_name = f"{checkpoint_path.stem}_multilabel"
        output_path = args.output_dir / f"{output_name}.tflite"
        backbone = _infer_backbone_from_name(checkpoint_path)
        record: dict[str, Any] = {
            "checkpoint": str(checkpoint_path),
            "backbone": backbone,
            "tflite": str(output_path),
            "status": "pending",
        }

        print(f"\n{'#' * 78}")
        print(f"  [{index}/{len(checkpoints)}] {checkpoint_path}")
        print(f"{'#' * 78}\n")

        try:
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            if args.resume and output_path.is_file():
                print(f"[RESUME] Existing export found: {output_path}")
                record["status"] = "skipped"
            else:
                model = load_model_from_checkpoint(
                    checkpoint_path,
                    torch.device("cpu"),
                    backbone,
                )
                tflite_path = export_tflite(
                    model,
                    args.output_dir,
                    image_size,
                    output_name=output_name,
                )
                del model
                record.update(
                    {
                        "status": "ok",
                        "tflite": str(tflite_path),
                        "tflite_mb": round(tflite_path.stat().st_size / 1024**2, 2),
                    }
                )
        except Exception as error:
            record.update(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"\n[FAILED] {checkpoint_path}: {record['error']}")
            if args.fail_fast:
                record["elapsed_s"] = round(time.perf_counter() - started, 1)
                results.append(record)
                summary_path = _write_summary(results, args.output_dir)
                print(f"Partial summary written to {summary_path}")
                raise

        record["elapsed_s"] = round(time.perf_counter() - started, 1)
        results.append(record)
        print(
            f"\n[{record['status'].upper()}] {checkpoint_path.name} "
            f"({_format_duration(record['elapsed_s'])})"
        )

    total_seconds = time.perf_counter() - total_started
    summary_path = _write_summary(results, args.output_dir)

    print("\n" + "=" * 78)
    print("  FINAL SUMMARY")
    print("=" * 78)
    for record in results:
        print(
            f"  {Path(record['checkpoint']).name:<42} {record['status']:<8} "
            f"{_format_duration(record['elapsed_s']):>12}"
        )
    print("-" * 78)
    print(f"  Total time: {_format_duration(total_seconds)}")
    print(f"  Summary: {summary_path}")
    print("=" * 78)

    failed = [record for record in results if record["status"] == "failed"]
    if failed:
        raise RuntimeError(
            "One or more exports failed: "
            + ", ".join(Path(record["checkpoint"]).name for record in failed)
        )


if __name__ == "__main__":
    main()
