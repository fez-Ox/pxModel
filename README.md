# pxModel — Five-Class Package Image Classifier

Multi-label package-image classification with five outputs:

`damaged`, `plastic_wrap`, `sealed`, `open`, `non_package`

The annotation CSV is versioned, but dataset images are supplied separately and intentionally excluded from Git.

## Requirements

- Git
- [uv](https://docs.astral.sh/uv/)
- Python 3.12 (installed automatically by uv when needed)
- Enough disk space for the separately supplied dataset, CUDA-enabled PyTorch environment, pretrained weights, and checkpoints
- An NVIDIA GPU is recommended but not required; training automatically falls back to CPU

## Fresh-clone setup

```sh
git clone https://github.com/fez-Ox/pxModel.git
cd pxModel
uv sync --locked
```

Verify the environment and accelerator:

```sh
uv run python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Import the dataset images into `data/combined_dataset/`. Filenames must match the `filename` column in `data/annotations.csv`; do not commit the image directory.

Validate the imported images and annotations:

```sh
uv run python -m pxmodel.validate_data

# Optional: also decode every image (slower)
uv run python -m pxmodel.validate_data --decode-images
```

The expected summary starts with:

```text
Dataset ready: 1879 images, 5 labels
```

## Training

Train the default EfficientNet-B0 model:

```sh
uv run python -m pxmodel.train
```

Select another supported backbone:

```sh
uv run python -m pxmodel.train --backbone convnext_tiny
```

Run `uv run python -m pxmodel.train --help` for the complete backbone list. Training uses a deterministic 70/15/15 split and writes the best checkpoint to `checkpoints/best_model.pt`. The training command validates the dataset before starting.

Training configuration is in `pxmodel/config.py`, including batch size, image size, epochs, workers, and the default backbone. Reduce `batch_size` if GPU memory is insufficient.

### Train and export multiple backbones

On a CUDA system, one script installs the locked TFLite extra, trains every registered backbone, and exports each best checkpoint:

```sh
./train_all_backbones.sh
```

Artifacts are written as `checkpoints/best_<backbone>.pt` and `exported_models/<backbone>_multilabel.tflite`. CUDA out-of-memory failures automatically retry that backbone with a halved batch size. A machine-readable run summary is written to `checkpoints/train_all_results.json`.

Useful options:

```sh
# Train a selected subset
./train_all_backbones.sh --backbones efficientnet_b0 convnext_tiny mobilenet_v3_large

# Reuse compatible checkpoints and export them without retraining
./train_all_backbones.sh --resume

# Start with a smaller batch or stop on the first failure
./train_all_backbones.sh --batch-size 8 --fail-fast
```

Run `./train_all_backbones.sh --help` for all options. CUDA is required unless `--allow-cpu` is explicitly supplied.

## Dataset

- Images: `data/combined_dataset/` (local, ignored by Git)
- Labels: `data/annotations.csv` (versioned)
- Samples: 1,879
- CSV columns: `filename,damaged,open,sealed,plastic_wrap,non_package`

All labels are binary. `non_package` is exclusive of the four package-state labels.

## YOLO package-presence gate

The Android pipeline can run a small YOLO segmentation gate before the five-label classifier:

```text
image → YOLO package gate → if package found → multi-label classifier
```

The gate uses detection confidence only; segmentation masks are not parsed on-device. Tune the gate for high recall because false negatives skip the classifier.

Prepare/check Ultralytics Package-Seg (`package-seg.yaml`):

```sh
uv run python -m pxmodel.yolo_package prepare-data --model yolo26n-seg.pt --allow-fallback
```

Train the smallest newest Ultralytics segmentation model first:

```sh
uv run python -m pxmodel.yolo_package train \
  --model yolo26n-seg.pt \
  --data package-seg.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 8
```

Evaluate and sweep confidence thresholds for high recall:

```sh
uv run python -m pxmodel.yolo_package evaluate-gate \
  --model checkpoints/yolo_package/yolo26n_package_seg/weights/best.pt \
  --target-recall 0.98
```

Export the gate for Android:

```sh
uv run python -m pxmodel.yolo_package export \
  --model checkpoints/yolo_package/yolo26n_package_seg/weights/best.pt \
  --format tflite \
  --output-name yolo_package_gate.tflite
```

Copy the exported artifact to:

```text
android/app/src/main/assets/yolo_package_gate.tflite
```

If this asset is absent, the Android app still runs in classifier-only fallback mode.

Build the Android app from a fresh checkout with the repository-local Gradle bootstrap:

```sh
cd android
./gradlew :app:assembleDebug
```

Run the two-stage Python pipeline:

```sh
uv run python -m pxmodel.yolo_package predict \
  --image path/to/image.jpg \
  --gate-model checkpoints/yolo_package/yolo26n_package_seg/weights/best.pt \
  --classifier-checkpoint checkpoints/best_model.pt \
  --gate-conf 0.25
```

Benchmark current classifier impact via the gated path:

```sh
uv run python -m pxmodel.yolo_package benchmark \
  --gate-model checkpoints/yolo_package/yolo26n_package_seg/weights/best.pt \
  --classifier-checkpoint checkpoints/best_model.pt \
  --gate-conf 0.25 \
  --limit 200
```

Latency rule of thumb:

```text
average gated latency ≈ gate latency + package_rate × classifier latency
```

Use a lower gate confidence (often `0.10–0.30`) if recall is insufficient.

## Inference and evaluation

A newly trained five-output checkpoint is required. Legacy four-output checkpoints are intentionally rejected.

```sh
uv run python -m pxmodel.predict \
  --image path/to/image.jpg \
  --checkpoint checkpoints/best_model.pt

uv run python -m pxmodel.evaluate
```

`pxmodel.evaluate` uses the checkpoint path configured in `pxmodel/config.py`; update it to the checkpoint being evaluated.

## Export and quantization

Install the optional TFLite dependencies when export is needed:

```sh
uv sync --locked --extra tflite
uv run python -m pxmodel.export
```

Quantize every compatible `best_<backbone>.pt` produced by the multi-backbone trainer:

```sh
./quantize_all_backbones.sh

# Also run the slower int4 QAT pipeline
./quantize_all_backbones.sh --qat

# Quantize a subset or skip already completed artifacts
./quantize_all_backbones.sh --backbones efficientnet_b0 convnext_tiny --resume
```

The default outputs are `checkpoints/quantized/<backbone>_int8_wo.pt` and `<backbone>_int8_dynamic.pt`, with optional `<backbone>_qat_int4.pt`. Results are recorded in `checkpoints/quantized/quantize_all_results.json`.

Single-checkpoint quantization and ONNX inference:

```sh
uv run python -m pxmodel.quantize \
  --checkpoint checkpoints/best_model.pt \
  --backbone efficientnet_b0

uv run python -m pxmodel.predict_onnx \
  --image path/to/image.jpg \
  --checkpoint checkpoints/best_model.pt
```

Generated checkpoints, ONNX files, TFLite files, and Android model assets are intentionally ignored. Retrain and re-export them from the five-class checkpoint.

## Project structure

```text
pxmodel/
├── labels.py              Canonical five-class schema
├── config.py              Paths and hyperparameters
├── dataset_multilabel.py  CSV-backed PyTorch dataset
├── validate_data.py       Clone/dataset integrity check
├── augmentation.py        Train, validation, and TTA transforms
├── model.py               Multi-backbone classifier
├── train.py               Frozen-head then full fine-tuning
├── train_all_backbones.py Multi-backbone CUDA training and TFLite export
├── quantize_all_backbones.py  Batch quantization orchestration
├── predict.py             PyTorch inference
├── predict_onnx.py        ONNX Runtime inference
├── evaluate.py            Test evaluation and threshold sweep
├── export.py              TFLite export
├── yolo_package.py        YOLO package-gate train/eval/export/pipeline benchmark
└── quantize.py            torchao quantization
data/
├── annotations.csv        Versioned labels
└── combined_dataset/      Local images (gitignored)
android/                   Android TFLite application
checkpoints/               Generated training output
train_all_backbones.sh     One-command multi-backbone runner
quantize_all_backbones.sh  Quantize all train-all checkpoints
```
