# pxModel — Multi-Label Image Classifier

Multi-label classification for box images using EfficientNet-B0 (and other backbones). Four labels: `damaged`, `plastic_wrap`, `sealed`, `open`.

## Setup

```sh
git clone <repo>
cd pxModel
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, install the CUDA version of PyTorch **before** `requirements.txt`:

```sh
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Data format

Place your data in `data/annotations.csv` with columns:

| filename | damaged | open | sealed | plastic_wrap |
|----------|---------|------|--------|--------------|

All labels are 0/1 integers. Place images in `data/combined_dataset/`.

## Usage

| Command | Description |
|---------|-------------|
| `python -m pxmodel.train` | Train (phase 1 frozen → phase 2 fine-tune) |
| `python -m pxmodel.predict image.jpg` | Single-image inference |
| `python -m pxmodel.evaluate` | Threshold sweep + evaluation on test split |
| `python -m pxmodel.export` | Export to TFLite (LiteRT) |
| `python -m pxmodel.quantize` | Full quantization comparison (dynamic + static + QAT) |
| `python -m pxmodel.quantize --dynamic` | Dynamic quantization only |
| `python -m pxmodel.quantize --static` | Static quantization only |
| `python -m pxmodel.quantize --qat` | Quantization-aware training |

## Configuration

Edit `pxmodel/config.py` to tune hyper-parameters, paths, model architecture, quantization backend, etc.

## Quantization

The quantization pipeline compares three methods:

1. **Dynamic** — int8 weights (Linear layers only), no calibration needed
2. **Static** — int8 weights + activations, requires calibration data
3. **QAT** — quantization-aware training with 5 epochs of fine-tuning

Results are printed as a comparison table and the best quantized model is saved to `exported_models/quantized_model.pt`.

## Project structure

```
pxmodel/               Python package
├── __init__.py        Re-exports key symbols
├── model.py           MultiLabelBoxClassifier (EfficientNet-B0)
├── config.py          Shared configuration
├── dataset_multilabel.py  CSV-backed PyTorch Dataset
├── augmentation.py    Albumentations pipelines
├── train.py           Two-phase training
├── predict.py         Single/batch inference
├── evaluate.py        Evaluation with threshold sweep
├── export.py          TFLite export
├── quantize.py        Quantization pipeline (dynamic / static / QAT)
└── compare_backbones.py  Backbone comparison benchmark
data/                  Images + annotation CSV
checkpoints/           Saved model weights (gitignored)
exported_models/       Exported models (gitignored)
```

## Android

See `android/` for the ONNX Runtime Mobile app. Export the TFLite model first:

```sh
python -m pxmodel.export
cp exported_models/efficientnet_b0_multilabel.tflite android/app/src/main/assets/
```
