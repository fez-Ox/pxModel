"""Train multiple backbones on CUDA and save PyTorch checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

import pxmodel.train as train_module
from pxmodel.config import batch_size as configured_batch_size
from pxmodel.config import output_dir
from pxmodel.labels import NUM_LABELS, require_current_label_count
from pxmodel.model import BACKBONE_REGISTRY
from pxmodel.validate_data import validate_dataset


def _format_duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or (
        isinstance(error, RuntimeError)
        and "out of memory" in str(error).lower()
        and "cuda" in str(error).lower()
    )


def _checkpoint_is_reusable(checkpoint_path: Path, backbone: str) -> bool:
    if not checkpoint_path.is_file():
        return False
    try:
        checkpoint_data = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        require_current_label_count(
            checkpoint_data["num_labels"], f"Checkpoint {checkpoint_path}"
        )
        return (
            checkpoint_data.get("backbone") == backbone
            and "model_state_dict" in checkpoint_data
        )
    except (KeyError, TypeError, ValueError, RuntimeError):
        return False


def _train_with_oom_retry(
    backbone: str,
    checkpoint_name: str,
    initial_batch_size: int,
) -> tuple[dict[str, Any], int]:
    batch_size = initial_batch_size
    while True:
        train_module.batch_size = batch_size
        try:
            result = train_module.train(
                backbone_name=backbone,
                ckpt_name=checkpoint_name,
            )
            return result, batch_size
        except BaseException as error:
            if not _is_cuda_oom(error) or batch_size == 1:
                raise
            next_batch_size = max(1, batch_size // 2)
            print(
                f"\n[CUDA OOM] {backbone} could not train with batch size "
                f"{batch_size}. Clearing CUDA memory and restarting with "
                f"batch size {next_batch_size}.\n"
            )
            _release_memory()
            batch_size = next_batch_size


def _write_summary(results: list[dict[str, Any]], total_seconds: float) -> Path:
    output_path = Path(output_dir) / "train_all_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "num_labels": NUM_LABELS,
        "total_time_s": round(total_seconds, 1),
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=sorted(BACKBONE_REGISTRY),
        default=list(BACKBONE_REGISTRY),
        help="Backbones to train (default: every registered backbone)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=configured_batch_size,
        help=(
            f"Initial training batch size (default: {configured_batch_size}); "
            "automatically halved and retried after a CUDA OOM"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip compatible five-class checkpoints that already exist",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately instead of continuing after a backbone fails",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU execution (CUDA is required by default)",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def main() -> None:
    args = _parse_args()

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is not available. Verify the NVIDIA driver and PyTorch CUDA "
            "installation, or explicitly pass --allow-cpu."
        )

    sample_count, positives = validate_dataset()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device_name = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if torch.cuda.is_available()
        else "CPU"
    )
    print("=" * 78)
    print("  TRAIN ALL BACKBONES")
    print("=" * 78)
    print(f"  Device:        {device_name}")
    print(f"  Samples:       {sample_count}")
    print(f"  Label counts:  {positives}")
    print(f"  Backbones:     {', '.join(args.backbones)}")
    print(f"  Initial batch: {args.batch_size}")
    print(f"  Checkpoints:   {Path(output_dir).resolve()}")
    print("=" * 78)

    started = time.perf_counter()
    results: list[dict[str, Any]] = []

    for index, backbone in enumerate(args.backbones, start=1):
        checkpoint_name = f"best_{backbone}.pt"
        checkpoint_path = Path(output_dir) / checkpoint_name
        backbone_started = time.perf_counter()
        record: dict[str, Any] = {
            "backbone": backbone,
            "checkpoint": str(checkpoint_path),
            "status": "pending",
        }

        print(f"\n{'#' * 78}")
        print(f"  [{index}/{len(args.backbones)}] {backbone}")
        print(f"{'#' * 78}\n")

        try:
            _seed_everything(args.seed)
            reusable = args.resume and _checkpoint_is_reusable(
                checkpoint_path, backbone
            )

            if reusable:
                print(f"[RESUME] Reusing checkpoint: {checkpoint_path}")
                record.update(
                    {
                        "status": "skipped",
                        "reused_checkpoint": True,
                        "checkpoint_mb": round(
                            checkpoint_path.stat().st_size / 1024**2, 2
                        ),
                        "elapsed_s": round(time.perf_counter() - backbone_started, 1),
                    }
                )
            else:
                training_result, used_batch_size = _train_with_oom_retry(
                    backbone,
                    checkpoint_name,
                    args.batch_size,
                )
                record.update(training_result)
                record["batch_size"] = used_batch_size
                record["reused_checkpoint"] = False

            if record["status"] != "skipped":
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(
                        f"Training did not create the expected checkpoint: "
                        f"{checkpoint_path}"
                    )

                record.update(
                    {
                        "status": "ok",
                        "checkpoint_mb": round(
                            checkpoint_path.stat().st_size / 1024**2, 2
                        ),
                        "elapsed_s": round(time.perf_counter() - backbone_started, 1),
                    }
                )
            print(
                f"\n[{record['status'].upper()}] {backbone}: "
                f"checkpoint={checkpoint_path}, "
                f"time={_format_duration(record['elapsed_s'])}"
            )
        except BaseException as error:
            record.update(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "elapsed_s": round(time.perf_counter() - backbone_started, 1),
                }
            )
            print(f"\n[FAILED] {backbone}: {record['error']}")
            if args.fail_fast or isinstance(error, (KeyboardInterrupt, SystemExit)):
                results.append(record)
                summary_path = _write_summary(
                    results, time.perf_counter() - started
                )
                print(f"Partial summary written to {summary_path}")
                raise
        finally:
            if record not in results:
                results.append(record)
            _release_memory()

    total_seconds = time.perf_counter() - started
    summary_path = _write_summary(results, total_seconds)

    print("\n" + "=" * 78)
    print("  FINAL SUMMARY")
    print("=" * 78)
    for record in results:
        print(
            f"  {record['backbone']:<24} {record['status']:<8} "
            f"{_format_duration(record['elapsed_s']):>12}  "
            f"{record.get('checkpoint', '-')}"
        )
    print("-" * 78)
    print(f"  Total time: {_format_duration(total_seconds)}")
    print(f"  Machine-readable summary: {summary_path}")
    print("=" * 78)

    failed = [record for record in results if record["status"] == "failed"]
    if failed:
        names = ", ".join(record["backbone"] for record in failed)
        raise RuntimeError(f"One or more backbones failed: {names}")


if __name__ == "__main__":
    main()
