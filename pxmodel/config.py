from pathlib import Path

# ----- Paths (edit these for your setup) -----
images_dir = Path("data/combined_dataset")
train_csv = Path("data/annotations.csv")
val_csv = Path("data/annotations.csv")
test_csv = Path("data/annotations.csv")
checkpoint = Path("checkpoints/best_model.pt")
output_dir = Path("checkpoints")
export_dir = Path("exported_models")
input_path = Path("image.jpg")

# ----- Model -----
image_size = 224
dropout = 0.3
backbone_name = "efficientnet_b0"

# ----- Training -----
batch_size = 32
epochs_phase1 = 10
epochs_phase2 = 25
lr_head = 1e-3
lr_backbone = 1e-5
weight_decay = 1e-4
num_workers = 4
skip_phase1 = False

# ----- Inference / Eval -----
threshold = 0.5
use_tta = True
output_csv = False
find_best_thresholds = True

# ----- Export -----
do_onnx = False
