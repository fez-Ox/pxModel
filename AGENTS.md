# pxModel — Agent Guide

## Project structure

```
pxmodel/               Python package (all source code)
├── __init__.py        Re-exports key symbols
├── model.py           MultiLabelBoxClassifier (EfficientNet-B0)
├── config.py          Shared configuration (edit hyper-parameters)
├── dataset_multilabel.py  CSV-backed PyTorch Dataset (OpenCV + Albumentations)
├── augmentation.py    Albumentations pipelines (train/val/TTA transforms)
├── train.py           Two-phase multi-label training (frozen → fine-tune)
├── predict.py         Single/batch inference with optional TTA (supports --compile)
├── predict_onnx.py    ONNX Runtime inference with optional int8 quantization
├── evaluate.py        Standalone evaluation with threshold sweep
├── export.py          ONNX / TFLite export
├── quantize.py        Quantization pipeline (torchao — int8 wo, int8 dyn, QAT int4)
└── compare_backbones.py  Backbone comparison benchmark
data/                  Images (+ annotation CSV)
checkpoints/           Saved model weights (gitignored)
exported_models/       Exports (gitignored)
android/               Android app (ONNX Runtime Mobile)
AGENTS.md              This file
```

## Dependencies

- Python 3.12+, venv at `.venv/`
- `pip install -r requirements.txt`
- For GPU: install CUDA torch before requirements

## Four labels

`["damaged", "plastic_wrap", "sealed", "open"]`

## Training

```sh
python -m pxmodel.train
```

## Inference

### PyTorch (baseline)

```sh
python -m pxmodel.predict --image <path> --checkpoint checkpoints/best_efficientnet_b0.pt
```

### PyTorch + torch.compile (~3.5x speedup on CPU)

```sh
python -m pxmodel.predict --image <path> --checkpoint checkpoints/best_efficientnet_b0.pt --compile
```

### ONNX Runtime (FP32, fastest CPU option)

```sh
# FP32 (recommended — ~7x faster than baseline, no accuracy loss)
python -m pxmodel.predict_onnx --image <path> --checkpoint checkpoints/best_efficientnet_b0.pt

# int8 dynamic quantization (may reduce accuracy)
python -m pxmodel.predict_onnx --image <path> --checkpoint checkpoints/best_efficientnet_b0.pt --quantize

# Reuse cached ONNX file (skips PyTorch → ONNX export)
python -m pxmodel.predict_onnx --image <path> --onnx-cache exported_models/model.onnx
```

### Batch eval

```sh
python -m pxmodel.evaluate
```

## Export

```sh
python -m pxmodel.export
```

Output: `exported_models/efficientnet_b0_multilabel.tflite`

## Quantization

```sh
python -m pxmodel.quantize           # full comparison
python -m pxmodel.quantize --qat     # QAT (int4 weight-only)
python -m pxmodel.quantize --backbone <name>  # override checkpoint metadata
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

CSV at `data/annotations.csv` with columns: `filename,damaged,open,sealed,plastic_wrap`. Label values are 0 or 1. 70/15/15 deterministic split.

## Known issues

- **No tests or CI** — validate by running scripts on real data.
- **Checkpoint backbone metadata** — `_save_checkpoint` used to save the config-default `backbone_name` instead of the actual backbone. Fixed in current code, but old checkpoints (b3, mobilenet_v3_large, convnext_base) all say `"efficientnet_b0"`. Use `--backbone` argument with `quantize.py` as a workaround.
- **timm version drift** — Checkpoints may fail to load even with correct `--backbone` if they were trained with a different `timm` version (model builder registries change across releases). Re-train after updating deps, or pin `timm` to the original version.
