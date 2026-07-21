from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from ultralytics import YOLO

from pxmodel.augmentation import get_val_transform
from pxmodel.config import checkpoint as classifier_checkpoint
from pxmodel.config import image_size, images_dir, test_csv, threshold as classifier_threshold
from pxmodel.dataset_multilabel import MultiLabelBoxDataset
from pxmodel.labels import LABEL_NAMES
from pxmodel.predict import format_prediction, load_checkpoint, load_image, predict_single

DEFAULT_YOLO_MODEL = "yolo26n-seg.pt"
FALLBACK_YOLO_MODEL = "yolo11n-seg.pt"
DEFAULT_DATASET = "package-seg.yaml"
DEFAULT_GATE_CONF = 0.25
PACKAGE_LABELS = [name for name in LABEL_NAMES if name != "non_package"]
NON_PACKAGE_INDEX = LABEL_NAMES.index("non_package")


@dataclass
class GateResult:
    has_package: bool
    confidence: float
    inference_time_ms: float


def load_yolo(model_path: str | Path, *, fallback: bool = True) -> YOLO:
    """Load a YOLO segmentation model, falling back only for local package support gaps."""
    try:
        return YOLO(str(model_path))
    except Exception:
        if fallback and str(model_path) == DEFAULT_YOLO_MODEL:
            print(
                f"Warning: could not load {DEFAULT_YOLO_MODEL}; falling back to "
                f"{FALLBACK_YOLO_MODEL}. Upgrade ultralytics when YOLO26 weights are available."
            )
            return YOLO(FALLBACK_YOLO_MODEL)
        raise


def gate_predict(model: YOLO, image: str | Path | np.ndarray, conf: float = DEFAULT_GATE_CONF, imgsz: int = 640, device: str | None = None) -> GateResult:
    start = time.perf_counter()
    results = model.predict(image, imgsz=imgsz, conf=conf, device=device, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    boxes = results[0].boxes
    if boxes is None or boxes.conf is None or len(boxes.conf) == 0:
        return GateResult(False, 0.0, elapsed_ms)
    max_conf = float(boxes.conf.detach().cpu().max().item())
    return GateResult(max_conf >= conf, max_conf, elapsed_ms)


def package_presence_targets(dataset: MultiLabelBoxDataset) -> np.ndarray:
    package_indices = [LABEL_NAMES.index(name) for name in PACKAGE_LABELS]
    targets = dataset.labels
    return (targets[:, package_indices].sum(axis=1) > 0).astype(np.int32)


def iter_dataset_image_paths(dataset: MultiLabelBoxDataset) -> Iterable[Path]:
    for name in dataset.filenames:
        yield dataset.images_dir / name


def train_yolo(args: argparse.Namespace) -> None:
    model = load_yolo(args.model, fallback=args.allow_fallback)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        pretrained=True,
        patience=args.patience,
        workers=args.workers,
    )
    print(results)


def prepare_data(args: argparse.Namespace) -> None:
    # Ultralytics resolves and downloads package-seg.yaml automatically on train/val.
    # A one-epoch dry validation is the lightest reliable preparation/check path.
    model = load_yolo(args.model, fallback=args.allow_fallback)
    results = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=args.device, split="val")
    print(results)


def evaluate_gate(args: argparse.Namespace) -> None:
    model = load_yolo(args.model, fallback=False)
    dataset = MultiLabelBoxDataset(images_dir=images_dir, labels_csv=test_csv, transform=None, split=args.split)
    y_true = package_presence_targets(dataset)
    rows: list[dict] = []

    # Predict once with a very low confidence floor, then sweep thresholds over
    # the captured max confidence. This avoids rerunning YOLO for each threshold.
    predict_conf = min(args.conf, args.min_sweep_conf)
    for path, target in zip(iter_dataset_image_paths(dataset), y_true):
        result = gate_predict(model, path, conf=predict_conf, imgsz=args.imgsz, device=args.device)
        rows.append({"file": str(path), "target": int(target), **asdict(result)})

    confidences = np.array([r["confidence"] for r in rows], dtype=np.float32)
    thresholds = np.array(args.sweep or [args.conf], dtype=np.float32)
    sweep_rows = []
    for t in thresholds:
        y_pred_t = (confidences >= t).astype(np.int32)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_t, average="binary", zero_division=0)
        sweep_rows.append({"conf": float(t), "precision": float(precision), "recall": float(recall), "f1": float(f1)})

    selected = max(
        sweep_rows,
        key=lambda r: (r["recall"] >= args.target_recall, r["recall"], r["precision"], r["f1"]),
    )
    summary = {
        "model": str(args.model),
        "selected_conf": selected["conf"],
        "target_recall": args.target_recall,
        "samples": len(rows),
        "package_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "precision": selected["precision"],
        "recall": selected["recall"],
        "f1": selected["f1"],
        "mean_gate_ms": float(np.mean([r["inference_time_ms"] for r in rows])) if rows else 0.0,
        "sweep": sweep_rows,
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))


def export_yolo(args: argparse.Namespace) -> None:
    model = load_yolo(args.model, fallback=False)
    exported = model.export(format=args.format, imgsz=args.imgsz, int8=args.int8, half=args.half, device=args.device)
    exported_path = Path(exported)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / (args.output_name or exported_path.name)
    if exported_path.resolve() != target.resolve():
        target.write_bytes(exported_path.read_bytes())
    print(f"Exported YOLO gate: {target}")


def predict_pipeline(args: argparse.Namespace) -> None:
    gate = load_yolo(args.gate_model, fallback=False)
    gate_result = gate_predict(gate, args.image, conf=args.gate_conf, imgsz=args.gate_imgsz, device=args.device)
    print(f"Gate: has_package={gate_result.has_package} conf={gate_result.confidence:.4f} time={gate_result.inference_time_ms:.3f} ms")
    if not gate_result.has_package:
        print("No package detected; downstream classifier skipped.")
        return

    device = torch.device(args.classifier_device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_checkpoint(args.classifier_checkpoint, device, args.backbone)
    image = load_image(args.image)
    transform = get_val_transform(image_size=image_size)
    probs = predict_single(model, image, transform, device)
    print(format_prediction(Path(args.image).name, probs, [args.classifier_threshold] * len(LABEL_NAMES), LABEL_NAMES))


def benchmark_pipeline(args: argparse.Namespace) -> None:
    gate = load_yolo(args.gate_model, fallback=False)
    device = torch.device(args.classifier_device or ("cuda" if torch.cuda.is_available() else "cpu"))
    classifier = load_checkpoint(args.classifier_checkpoint, device, args.backbone)
    transform = get_val_transform(image_size=image_size)
    dataset = MultiLabelBoxDataset(images_dir=images_dir, labels_csv=test_csv, transform=None, split=args.split)

    def measure_classifier(path: Path) -> float:
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        tensor = transform(image=image)["image"].unsqueeze(0).to(device).contiguous()
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            classifier(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000.0

    rows = []
    for idx, path in enumerate(iter_dataset_image_paths(dataset)):
        if args.limit and idx >= args.limit:
            break
        baseline_cls_ms = measure_classifier(path)
        gate_result = gate_predict(gate, path, conf=args.gate_conf, imgsz=args.gate_imgsz, device=args.device)
        gated_cls_ms = baseline_cls_ms if gate_result.has_package else 0.0
        rows.append({
            "file": str(path),
            "classifier_only_ms": baseline_cls_ms,
            "gate_ms": gate_result.inference_time_ms,
            "classifier_ms": gated_cls_ms,
            "total_ms": gate_result.inference_time_ms + gated_cls_ms,
            "ran_classifier": gate_result.has_package,
            "gate_conf": gate_result.confidence,
        })

    summary = {
        "samples": len(rows),
        "gate_conf": args.gate_conf,
        "skip_rate": float(1.0 - np.mean([r["ran_classifier"] for r in rows])) if rows else 0.0,
        "mean_classifier_only_ms": float(np.mean([r["classifier_only_ms"] for r in rows])) if rows else 0.0,
        "mean_gate_ms": float(np.mean([r["gate_ms"] for r in rows])) if rows else 0.0,
        "mean_classifier_ms_when_run": float(np.mean([r["classifier_ms"] for r in rows if r["ran_classifier"]])) if any(r["ran_classifier"] for r in rows) else 0.0,
        "mean_gated_total_ms": float(np.mean([r["total_ms"] for r in rows])) if rows else 0.0,
        "mean_latency_delta_ms": float(np.mean([r["total_ms"] - r["classifier_only_ms"] for r in rows])) if rows else 0.0,
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO package gate utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    common_yolo = argparse.ArgumentParser(add_help=False)
    common_yolo.add_argument("--model", type=str, default=DEFAULT_YOLO_MODEL)
    common_yolo.add_argument("--data", type=str, default=DEFAULT_DATASET)
    common_yolo.add_argument("--imgsz", type=int, default=640)
    common_yolo.add_argument("--batch", type=int, default=8)
    common_yolo.add_argument("--device", type=str, default=None)
    common_yolo.add_argument("--allow-fallback", action="store_true")

    p = sub.add_parser("prepare-data", parents=[common_yolo])
    p.set_defaults(func=prepare_data)

    p = sub.add_parser("train", parents=[common_yolo])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--project", type=Path, default=Path("checkpoints/yolo_package"))
    p.add_argument("--name", type=str, default="yolo26n_package_seg")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--workers", type=int, default=2)
    p.set_defaults(func=train_yolo)

    p = sub.add_parser("evaluate-gate")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--conf", type=float, default=DEFAULT_GATE_CONF)
    p.add_argument("--sweep", type=float, nargs="*", default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
    p.add_argument("--min-sweep-conf", type=float, default=0.01)
    p.add_argument("--target-recall", type=float, default=0.98)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--output", type=Path, default=Path("checkpoints/yolo_package/gate_eval.json"))
    p.set_defaults(func=evaluate_gate)

    p = sub.add_parser("export")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--format", choices=["tflite", "onnx"], default="tflite")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--int8", action="store_true")
    p.add_argument("--half", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("exported_models"))
    p.add_argument("--output-name", type=str, default="yolo_package_gate.tflite")
    p.set_defaults(func=export_yolo)

    p = sub.add_parser("predict")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--gate-model", type=Path, required=True)
    p.add_argument("--gate-conf", type=float, default=DEFAULT_GATE_CONF)
    p.add_argument("--gate-imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default=None, help="YOLO device")
    p.add_argument("--classifier-checkpoint", type=Path, default=classifier_checkpoint)
    p.add_argument("--classifier-device", type=str, default=None)
    p.add_argument("--classifier-threshold", type=float, default=classifier_threshold)
    p.add_argument("--backbone", type=str, default=None)
    p.set_defaults(func=predict_pipeline)

    p = sub.add_parser("benchmark")
    p.add_argument("--gate-model", type=Path, required=True)
    p.add_argument("--gate-conf", type=float, default=DEFAULT_GATE_CONF)
    p.add_argument("--gate-imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default=None, help="YOLO device")
    p.add_argument("--classifier-checkpoint", type=Path, default=classifier_checkpoint)
    p.add_argument("--classifier-device", type=str, default=None)
    p.add_argument("--backbone", type=str, default=None)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("checkpoints/yolo_package/pipeline_benchmark.json"))
    p.set_defaults(func=benchmark_pipeline)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
