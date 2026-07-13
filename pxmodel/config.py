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
backbone_name = "efficientnet_b0"
image_size = 224
dropout = 0.3

# ----- Training -----
batch_size = 32
epochs_phase1 = 10
epochs_phase2 = 25
num_workers = 2

# ----- Inference / Eval -----
threshold = 0.6
use_tta = False
output_csv = False
find_best_thresholds = False

# ----- Export -----
do_onnx = False
