from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from roboflow import Roboflow

IMAGES_DIR = Path("data/combined_dataset")
LABELS_CSV = Path("data/annotations.csv")
PROJECT_NAME = "pxmodel-multilabel"
PROJECT_TYPE = "multi-label-classification"
PROJECT_LICENSE = "MIT"
LABEL_COLS = ["damaged", "plastic_wrap", "sealed", "open"]
SPLIT_RATIOS = (0.7, 0.15, 0.15)
SLEEP_BETWEEN_UPLOADS = 0.05


def main() -> None:
    df = pd.read_csv(LABELS_CSV)
    print(f"Loaded {len(df)} rows from {LABELS_CSV}", flush=True)

    valid_mask: list[bool] = []
    for fname in df["filename"]:
        valid_mask.append((IMAGES_DIR / fname).is_file())
    df = df.loc[valid_mask].reset_index(drop=True)
    print(f"Images found on disk: {len(df)}", flush=True)

    n = len(df)
    indices = np.random.RandomState(42).permutation(n)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * (SPLIT_RATIOS[0] + SPLIT_RATIOS[1]))

    split_map: dict[str, np.ndarray] = {
        "train": indices[:n_train],
        "valid": indices[n_train:n_val],
        "test": indices[n_val:],
    }

    rf = Roboflow()
    ws = rf.workspace()

    existing = [p["name"] for p in ws.project_list]
    if PROJECT_NAME in existing:
        print(f"Project '{PROJECT_NAME}' already exists, using it.", flush=True)
        project = ws.project(PROJECT_NAME)
    else:
        print(f"Creating project '{PROJECT_NAME}'...", flush=True)
        project = ws.create_project(
            project_name=PROJECT_NAME,
            project_type=PROJECT_TYPE,
            project_license=PROJECT_LICENSE,
            annotation="box",
        )
        print(f"Created: {project.id}", flush=True)

    uploaded = 0
    annot_errors = 0
    upload_errors = 0
    total = sum(len(v) for v in split_map.values())

    for split_name, split_indices in split_map.items():
        print(f"\n--- Uploading {split_name} split ({len(split_indices)} images) ---", flush=True)
        for idx in split_indices:
            row = df.iloc[idx]
            img_path = IMAGES_DIR / row["filename"]

            if not img_path.is_file():
                print(f"  [SKIP] File not found: {img_path}", flush=True)
                continue

            active_classes = [col for col in LABEL_COLS if row[col] == 1]

            try:
                result = project.single_upload(
                    image_path=str(img_path),
                    split=split_name,
                    num_retry_uploads=3,
                )
                image_id = result["image"]["id"]
            except Exception as e:
                print(f"  [UPLOAD ERROR] {row['filename']}: {e}", flush=True)
                upload_errors += 1
                continue

            for cls in active_classes:
                try:
                    project.save_annotation(
                        annotation_path=cls,
                        image_id=image_id,
                        num_retry_uploads=3,
                    )
                except Exception as e:
                    print(f"  [ANNOT ERROR] {row['filename']} class={cls}: {e}", flush=True)
                    annot_errors += 1

            uploaded += 1
            if uploaded % 50 == 0 or uploaded == 1:
                print(f"  Progress: {uploaded}/{total} ({upload_errors} upload err, {annot_errors} annot err)...", flush=True)

            time.sleep(SLEEP_BETWEEN_UPLOADS)

    print(f"\n{'='*50}", flush=True)
    print(f"Upload complete: {uploaded} images, {upload_errors} upload err, {annot_errors} annot err", flush=True)
    print(f"{'='*50}", flush=True)


if __name__ == "__main__":
    main()
