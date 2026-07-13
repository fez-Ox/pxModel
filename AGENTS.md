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
├── train_all_backbones.py  CUDA multi-backbone training + TFLite export
├── quantize_all_backbones.py  Batch quantization orchestration
├── predict.py         Single/batch inference with optional TTA (supports --compile)
├── predict_onnx.py    ONNX Runtime inference with optional int8 quantization
├── evaluate.py        Standalone evaluation with threshold sweep
├── export.py          ONNX / TFLite export
├── quantize.py        Quantization pipeline (torchao — int8 wo, int8 dyn, QAT int4)
└── compare_backbones.py  Backbone comparison benchmark
data/                  Versioned annotation CSV; images are local/gitignored
checkpoints/           Saved model weights (gitignored)
exported_models/       Exports (gitignored)
android/               Android app (ONNX Runtime Mobile)
AGENTS.md              This file
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

```sh
uv run python -m pxmodel.validate_data
uv run python -m pxmodel.train
uv run python -m pxmodel.train --backbone <name>
```

## Train and export all backbones

```sh
./train_all_backbones.sh
```

This installs the locked `tflite` extra, requires CUDA by default, trains every entry in `BACKBONE_REGISTRY`, saves `checkpoints/best_<backbone>.pt`, exports `exported_models/<backbone>_multilabel.tflite`, retries CUDA OOMs at smaller batch sizes, and writes `checkpoints/train_all_results.json`. Use `--resume` to reuse compatible checkpoints or `--backbones <name> ...` for a subset.

## Quantize all trained backbones

```sh
./quantize_all_backbones.sh
./quantize_all_backbones.sh --qat  # also run int4 QAT
```

The script discovers compatible `checkpoints/best_<backbone>.pt` files, runs the existing quantization pipeline in an isolated process per backbone, writes artifacts under `checkpoints/quantized/`, supports `--resume`, subsets, and `--fail-fast`, and writes `quantize_all_results.json`.

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

## Export

```sh
uv sync --locked --extra tflite
uv run python -m pxmodel.export
```

Output: `exported_models/efficientnet_b0_multilabel.tflite`

## Quantization

```sh
uv run python -m pxmodel.quantize --backbone <name>       # full comparison
uv run python -m pxmodel.quantize --backbone <name> --qat # QAT (int4 weight-only)
```

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

## Configuration

Edit `pxmodel/config.py` to tune hyper-parameters, paths, model architecture, etc. All paths are relative (clone-friendly).

## Data format

CSV at `data/annotations.csv` with columns: `filename,damaged,open,sealed,plastic_wrap,non_package`. Label values are 0 or 1. Images are supplied separately in the gitignored `data/combined_dataset/`. Run `uv run python -m pxmodel.validate_data` after importing them. 70/15/15 deterministic split.

## Known issues

- **No tests or CI** — validate with `uv run python -m pxmodel.validate_data` and scripts on real data.
- **Checkpoint backbone metadata** — `_save_checkpoint` used to save the config-default `backbone_name` instead of the actual backbone. Fixed in current code, but old checkpoints (b3, mobilenet_v3_large, convnext_base) all say `"efficientnet_b0"`. Use `--backbone` argument with `quantize.py` as a workaround.
- Existing four-output checkpoints and exports are incompatible with the five-class schema and must be retrained/re-exported.
