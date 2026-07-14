# pxModel — Agent Guide

## Project structure

```
pxmodel/               Python package (all source code)
├── __init__.py        Re-exports key symbols
├── model.py           MultiLabelBoxClassifier (EfficientNet-B0)
├── config.py          Shared configuration (edit hyper-parameters)
├── labels.py          Canonical five-class label schema
├── dataset_multilabel.py  CSV-backed PyTorch Dataset (OpenCV + Albumentations)
├── validate_data.py   Clone/dataset integrity validation
├── augmentation.py    Albumentations pipelines (train/val/TTA transforms)
├── train.py           Two-phase multi-label training (frozen → fine-tune)
├── train_all_backbones.py  CUDA multi-backbone training (saves .pt only)
├── quantize.py        Quantization pipeline (torchao — int8 wo, int8 dyn, QAT int4)
├── quantize_all_backbones.py  Batch quantization orchestration
├── export.py          Single-checkpoint TFLite export (training or quantized .pt → .tflite)
├── export_all_backbones.py  Batch TFLite export (discovers all .pt, exports each)
├── predict.py         Single/batch inference with optional TTA (supports --compile)
├── predict_onnx.py    ONNX Runtime inference with optional int8 quantization
├── evaluate.py        Standalone evaluation with threshold sweep
├── benchmark.py       Inference benchmarking
└── compare_backbones.py  Backbone comparison benchmark
data/                  Versioned annotation CSV; images are local/gitignored
checkpoints/           Saved model weights (gitignored)
  quantized/           Quantized .pt files (gitignored)
exported_models/ DCP remains available for OpenCode plugin users, but new features are landing in Sleev first. If you are starting fresh, we recommend trying Sleev:

      TFLite exports (gitignored)
android/               Android app (LiteRT / ONNX Runtime Mobile)
annotator/             Local web app to view/filter/edit annotations (untracked)
scraped/               Local scraper + curation web app for new images (untracked)
AGENTS.md              This file
```

## Annotation web app (`annotator/`, untracked)

Lets you browse, filter, and edit `data/annotations.csv` labels, and serves images
from `data/combined_dataset/`. Self-contained — deps are provided ephemerally via
`uv run --with`, so `pyproject.toml`/`uv.lock` are never touched.

```sh
./annotator/run.sh                      # http://127.0.0.1:8000
```

- `GET /api/labels` — label schema (from `pxmodel.labels`)
- `GET /api/annotations?page=&page_size=&<label>=0|1` — filtered, paginated rows
- `POST /api/annotations/{filename}` — update one label (writes CSV, auto `.bak`)
- Edits write directly to `data/annotations.csv`; first write creates `data/annotations.csv.bak`.

## Pipeline architecture

The project uses three independent pipelines. Each reads `.pt` artifacts and can be run separately.

```
Training                  Quantization                 Export
─────────                 ────────────                 ──────
train.py                  quantize.py                  export.py
train_all_backbones.py    quantize_all_backbones.py    export_all_backbones.py
        │                         │                           │
        ▼                         ▼                           ▼
checkpoints/best_*.pt     checkpoints/quantized/*.pt   exported_models/*.tflite
```

## Dependencies

- Python 3.12+; use **uv exclusively** for Python environments, dependencies, and command execution.
- Use `uv sync`, `uv add`, and `uv run`; do not use `pip` directly.
- Declare dependencies in `pyproject.toml` and lock them in `uv.lock`.
- Do not create or maintain pip-related dependency files such as `requirements.txt`, `requirements-dev.txt`, or constraints files.
- Configure any GPU-specific PyTorch source and dependency settings through uv/`pyproject.toml`.

## Five labels

`["damaged", "plastic_wrap", "sealed", "open", "non_package"]`

## Training

Training only produces `.pt` checkpoints — it does **not** export to TFLite.

```sh
uv run python -m pxmodel.validate_data
uv run python -m pxmodel.train
uv run python -m pxmodel.train --backbone <name>
```

## Train all backbones

```sh
./train_all_backbones.sh
```

Requires CUDA by default. Trains every entry in `BACKBONE_REGISTRY`, saves `checkpoints/best_<backbone>.pt`, retries CUDA OOMs at smaller batch sizes, and writes `checkpoints/train_all_results.json`. Use `--resume` to skip backbones whose checkpoints already exist, or `--backbones <name> ...` for a subset.

## Quantization

Quantization reads `.pt` training checkpoints and outputs quantized `.pt` files. It does **not** export to TFLite.

### Single backbone

```sh
uv run python -m pxmodel.quantize --checkpoint checkpoints/best_<backbone>.pt --backbone <name>
uv run python -m pxmodel.quantize --checkpoint checkpoints/best_<backbone>.pt --backbone <name> --save-dir checkpoints/quantized
uv run python -m pxmodel.quantize --checkpoint checkpoints/best_<backbone>.pt --backbone <name> --save-dir checkpoints/quantized --qat
```

### Quantize all trained backbones

```sh
./quantize_all_backbones.sh
./quantize_all_backbones.sh --qat  # also run int4 QAT
```

Discovers compatible `checkpoints/best_<backbone>.pt` files, runs the quantization pipeline in an isolated process per backbone, writes quantized `.pt` artifacts under `checkpoints/quantized/`, supports `--resume`, `--backbones`, and `--fail-fast`, and writes `quantize_all_results.json`.

Uses `torchao` (no `torch.ao` / `torch.ao.quantization.quantize_fx`).

**Important:** torchao's `quantize_()` with `Int8WeightOnlyConfig` / `Int8DynamicActivationInt8WeightConfig`
only targets `nn.Linear` layers. EfficientNet-B0 has 81 `Conv2d` layers (92% of compute) and only 2 `Linear`
layers (classifier head). Weight quantization **reduces file/memory size** but does **not improve latency**
because the Conv2d backbone runs in FP32 regardless.

| Goal | Tool | How |
|---|---|---|
| Smaller model files | `pxmodel.quantize` | torchao weight-only quant |
| Faster CPU inference | `--compile` or `predict_onnx` | torch.compile or ONNX Runtime FP32 |
| Both | quantize weights + compile, or use ONNX Runtime FP32 |

## Export (TFLite)

Export reads any `.pt` checkpoint (training **or** quantized) and produces `.tflite` files. It requires the `tflite` extra.

### Single checkpoint

```sh
uv sync --locked --extra tflite

# Export a training checkpoint
uv run python -m pxmodel.export --checkpoint checkpoints/best_efficientnet_b0.pt --backbone efficientnet_b0

# Export a quantized checkpoint
uv run python -m pxmodel.export --checkpoint checkpoints/quantized/efficientnet_b0_int8_wo.pt --backbone efficientnet_b0

# Custom output name
uv run python -m pxmodel.export --checkpoint checkpoints/best_model.pt --output-name my_model
```

### Export all backbones

```sh
./export_all_backbones.sh                     # export all discovered .pt (trained + quantized)
./export_all_backbones.sh --include trained    # only training checkpoints
./export_all_backbones.sh --include quantized  # only quantized checkpoints
./export_all_backbones.sh --resume             # skip if .tflite already exists
```

Output: `exported_models/<stem>_multilabel.tflite`

## Inference

### PyTorch (baseline)

```sh
uv run python -m pxmodel.predict --image <path> --checkpoint checkpoints/best_model.pt
```

### PyTorch + torch.compile (~3.5x speedup on CPU)

```sh
uv run python -m pxmodel.predict --image <path> --checkpoint checkpoints/best_model.pt --compile
```

### ONNX Runtime (FP32, fastest CPU option)

```sh
# FP32 (recommended — ~7x faster than baseline, no accuracy loss)
uv run python -m pxmodel.predict_onnx --image <path> --checkpoint checkpoints/best_model.pt

# int8 dynamic quantization (may reduce accuracy)
uv run python -m pxmodel.predict_onnx --image <path> --checkpoint checkpoints/best_model.pt --quantize

# Reuse cached ONNX file (skips PyTorch → ONNX export)
uv run python -m pxmodel.predict_onnx --image <path> --onnx-cache exported_models/model.onnx
```

### Batch eval

```sh
uv run python -m pxmodel.evaluate
```

## Configuration

Edit `pxmodel/config.py` to tune hyper-parameters, paths, model architecture, etc. All paths are relative (clone-friendly).

## Data format

CSV at `data/annotations.csv` with columns: `filename,damaged,open,sealed,plastic_wrap,non_package`. Label values are 0 or 1. Images are supplied separately in the gitignored `data/combined_dataset/`. Run `uv run python -m pxmodel.validate_data` after importing them. 70/15/15 deterministic split.

## Known issues

- **No tests or CI** — validate with `uv run python -m pxmodel.validate_data` and scripts on real data.
- **Checkpoint backbone metadata** — `_save_checkpoint` used to save the config-default `backbone_name` instead of the actual backbone. Fixed in current code, but old checkpoints (b3, mobilenet_v3_large, convnext_base) all say `"efficientnet_b0"`. Use `--backbone` argument with `quantize.py` as a workaround.
- Existing four-output checkpoints and exports are incompatible with the five-class schema and must be retrained/re-exported.
