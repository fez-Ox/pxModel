"""Export convnext_base checkpoint → TFLite."""
import warnings; warnings.filterwarnings("ignore")

import torch
from pxmodel.model import MultiLabelBoxClassifier
from pxmodel.config import export_dir
from pxmodel.labels import require_current_label_count

import litert_torch

ckpt = torch.load("checkpoints/best_convnext_base.pt", map_location="cpu", weights_only=True)
require_current_label_count(ckpt["num_labels"], "ConvNeXt-Base checkpoint")

model = MultiLabelBoxClassifier(
    num_labels=ckpt["num_labels"],
    backbone_name="convnext_base",
    pretrained=False,
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

dummy = torch.randn(1, 3, 224, 224)
print("Converting to TFLite …")
edge = litert_torch.convert(model, (dummy,))
export_dir.mkdir(parents=True, exist_ok=True)
out = export_dir / "convnext_base_multilabel.tflite"
edge.export(out)
print(f"Done — {out.stat().st_size / 1024 / 1024:.1f} MB")
