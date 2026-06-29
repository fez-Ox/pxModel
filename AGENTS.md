# pxModel — Agent Guide

## Project structure

```
config.py              Shared configuration (edit hyper-parameters)
train.py               Two-phase multi-label training (frozen → fine-tune)
predict.py             Single/batch inference with optional TTA
evaluate.py            Standalone evaluation with threshold sweep
export.py              ONNX export (opset 17, dynamic batch dims)
model.py               MultiLabelBoxClassifier (EfficientNet-B0)
dataset_multilabel.py  CSV-backed PyTorch Dataset (OpenCV + Albumentations)
augmentation.py        Albumentations pipelines (train/val/TTA transforms)
AGENTS.md              This file
```

Flat Python scripts, no package config, no tests, no CI.

## Dependencies

- Python 3.14, venv at `.venv/`
- **torch 2.12.1 (CPU)**, torchvision 0.27.1 (CPU) — no CUDA
- `albumentations`, `opencv-python`, `pandas`, `scikit-learn`, `matplotlib`

Use `.venv/bin/python` (no activation script needed).

## Four labels

`["damaged", "plastic_wrap", "sealed", "open"]` — defined in both `augmentation.py` and `train.py`.

## Training

```sh
.venv/bin/python train.py
```

Takes ~50 minutes total on CPU.  Two-phase training:
- **Phase 1** (10 epochs) — freezes backbone (`model.features`), trains classifier head only.
- **Phase 2** (25 epochs) — fine-tunes all weights with separate LR groups (backbone `1e-5`, head `1e-4`). Cosine annealing LR schedule.

Checkpoints saved to `checkpoints/best_model.pt`.

### If training times out

Run it in the background with nohup:

```sh
nohup .venv/bin/python train.py > training.log 2>&1 &
tail -f training.log
```

Single `nohup` run will complete all 35 epochs without timeout.

## Inference

```sh
.venv/bin/python predict.py   # single/batch inference
.venv/bin/python evaluate.py  # threshold sweep on test split
```

## Export

```sh
.venv/bin/python export.py
```

ONNX opset 17, dynamic batch dims.

## Configuration

Edit `config.py` to tune hyper-parameters (learning rates, batch size, epochs, etc.).  All scripts import from `config.py` — no CLI arguments needed.

Current data paths (already set correctly):
- `images_dir = "data/combined_dataset"` — all images
- `train_csv = val_csv = test_csv = "data/annotations.csv"` — single annotation file

The dataset performs a deterministic 70/15/15 train/val/test split in-code (`split` parameter).

## Data format

CSV at `data/annotations.csv` with columns:
```
filename,damaged,open,sealed,plastic_wrap
```

All label values are 0 or 1 (integers).  Dataset silently skips rows whose image file is missing on disk.  Images loaded via OpenCV (BGR→RGB), transforms via Albumentations.  `pos_weight` auto-calculated from label distribution for `BCEWithLogitsLoss`.

**1465 annotated images** in CSV.  **2011 total images** on disk (546 are unlabeled — not used).

## Known issues

- **No requirements.txt / pyproject.toml** — install manually with `pip install torch torchvision albumentations opencv-python pandas scikit-learn matplotlib`.
- **No tests or CI** — any changes must be validated by running the scripts on real data.
- **346 unlabeled images** exist in `data/combined_dataset/` but are not annotated in the CSV.
