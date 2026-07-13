"""Quantize every compatible checkpoint produced by train_all_backbones."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from pxmodel.config import output_dir
from pxmodel.labels import require_current_label_count
from pxmodel.model import BACKBONE_REGISTRY
from pxmodel.validate_data import validate_dataset

DEFAULT_QUANTIZED_DIR = Path(output_dir) / "quantized"


def _format_duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"


def _discover_checkpoints() -> dict[str, Path]:
    """Find five-class ``best_<backbone>.pt`` training checkpoints."""
    discovered: dict[str, Path] = {}
    checkpoint_dir = Path(output_dir)
    for backbone in BACKBONE_REGISTRY:
        path = checkpoint_dir / f"best_{backbone}.pt"
        if path.is_file():
            discovered[backbone] = path
    return discovered


def _validate_checkpoint(path: Path, backbone: str) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    require_current_label_count(checkpoint["num_labels"], f"Checkpoint {path}")
    saved_backbone = checkpoint.get("backbone")
    if saved_backbone != backbone:
        raise ValueError(
            f"Checkpoint {path} says backbone={saved_backbone!r}; "
            f"expected {backbone!r}"
        )
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint has no model_state_dict: {path}")


def _expected_outputs(
    backbone: str,
    destination: Path,
    include_qat: bool,
) -> list[Path]:
    outputs = [
        destination / f"{backbone}_int8_wo.pt",
        destination / f"{backbone}_int8_dynamic.pt",
    ]
    if include_qat:
        outputs.append(destination / f"{backbone}_qat_int4.pt")
    return outputs


def _write_summary(results: list[dict[str, Any]], destination: Path) -> Path:
    summary_path = destination / "quantize_all_results.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=sorted(BACKBONE_REGISTRY),
        default=None,
        help=(
            "Backbones to quantize (default: all compatible "
            "best_<backbone>.pt checkpoints found in checkpoints/)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_QUANTIZED_DIR,
        help=f"Quantized artifact directory (default: {DEFAULT_QUANTIZED_DIR})",
    )
    parser.add_argument(
        "--qat",
        action="store_true",
        help="Also run the substantially slower int4 QAT pipeline",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a backbone when all requested quantized outputs already exist",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately instead of continuing after a backbone fails",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    discovered = _discover_checkpoints()

    if args.backbones is None:
        selected = list(discovered)
    else:
        selected = args.backbones

    if not selected:
        raise FileNotFoundError(
            "No train-all checkpoints were found. Run ./train_all_backbones.sh first, "
            "or provide --backbones after placing best_<backbone>.pt files in "
            f"{Path(output_dir)}."
        )

    sample_count, _ = validate_dataset()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  QUANTIZE ALL TRAINED BACKBONES")
    print("=" * 78)
    print(f"  Device:      {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Samples:     {sample_count}")
    print(f"  Backbones:   {', '.join(selected)}")
    print(f"  Include QAT: {args.qat}")
    print(f"  Output:      {args.output_dir.resolve()}")
    print("=" * 78)

    results: list[dict[str, Any]] = []
    total_started = time.perf_counter()

    for index, backbone in enumerate(selected, start=1):
        checkpoint_path = Path(output_dir) / f"best_{backbone}.pt"
        outputs = _expected_outputs(backbone, args.output_dir, args.qat)
        started = time.perf_counter()
        record: dict[str, Any] = {
            "backbone": backbone,
            "checkpoint": str(checkpoint_path),
            "outputs": [str(path) for path in outputs],
            "status": "pending",
        }

        print(f"\n{'#' * 78}")
        print(f"  [{index}/{len(selected)}] {backbone}")
        print(f"{'#' * 78}\n")

        try:
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Training checkpoint is missing: {checkpoint_path}"
                )
            _validate_checkpoint(checkpoint_path, backbone)

            if args.resume and all(path.is_file() for path in outputs):
                print("[RESUME] All requested outputs already exist; skipping.")
                record["status"] = "skipped"
            else:
                command = [
                    sys.executable,
                    "-m",
                    "pxmodel.quantize",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--backbone",
                    backbone,
                    "--save-dir",
                    str(args.output_dir),
                ]
                if args.qat:
                    command.append("--qat")

                subprocess.run(command, check=True)

                missing_outputs = [path for path in outputs if not path.is_file()]
                if missing_outputs:
                    raise FileNotFoundError(
                        "Quantization completed without creating: "
                        + ", ".join(str(path) for path in missing_outputs)
                    )
                record["status"] = "ok"

            record["artifacts"] = [
                {
                    "path": str(path),
                    "size_mb": round(path.stat().st_size / 1024**2, 2),
                }
                for path in outputs
                if path.is_file()
            ]
        except (Exception, subprocess.CalledProcessError) as error:
            record.update(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"\n[FAILED] {backbone}: {record['error']}")
            if args.fail_fast:
                record["elapsed_s"] = round(time.perf_counter() - started, 1)
                results.append(record)
                summary_path = _write_summary(results, args.output_dir)
                print(f"Partial summary written to {summary_path}")
                raise

        record["elapsed_s"] = round(time.perf_counter() - started, 1)
        results.append(record)
        print(
            f"\n[{record['status'].upper()}] {backbone} "
            f"({_format_duration(record['elapsed_s'])})"
        )

    total_seconds = time.perf_counter() - total_started
    summary_path = _write_summary(results, args.output_dir)

    print("\n" + "=" * 78)
    print("  FINAL SUMMARY")
    print("=" * 78)
    for record in results:
        print(
            f"  {record['backbone']:<24} {record['status']:<8} "
            f"{_format_duration(record['elapsed_s']):>12}"
        )
    print("-" * 78)
    print(f"  Total time: {_format_duration(total_seconds)}")
    print(f"  Summary: {summary_path}")
    print("=" * 78)

    failed = [record for record in results if record["status"] == "failed"]
    if failed:
        names = ", ".join(record["backbone"] for record in failed)
        raise RuntimeError(f"One or more backbones failed: {names}")


if __name__ == "__main__":
    main()
