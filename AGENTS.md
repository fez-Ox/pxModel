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
├── predict.py         Single/batch inference with optional TTA
├── evaluate.py        Standalone evaluation with threshold sweep
├── export.py          ONNX / TFLite export
├── quantize.py        Quantization pipeline (dynamic/static/QAT)
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

```sh
python -m pxmodel.predict <image_path>
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
python -m pxmodel.quantize --static  # static only
python -m pxmodel.quantize --qat     # QAT
```

## Configuration

Edit `pxmodel/config.py` to tune hyper-parameters, paths, model architecture, quantization backend, etc. All paths are relative (clone-friendly).

## Data format

CSV at `data/annotations.csv` with columns: `filename,damaged,open,sealed,plastic_wrap`. Label values are 0 or 1. 70/15/15 deterministic split.

## Known issues

- **No tests or CI** — validate by running scripts on real data.
