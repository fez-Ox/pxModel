from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pxmodel.labels import LABEL_NAMES as CANONICAL_LABEL_NAMES


class MultiLabelBoxDataset(Dataset):
    LABEL_NAMES = CANONICAL_LABEL_NAMES

    def __init__(
        self,
        images_dir: str | Path,
        labels_csv: str | Path,
        transform: Optional[Any] = None,
        split: Optional[str] = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.transform = transform

        df = pd.read_csv(labels_csv)

        # ------------------------------------------------------------------
        # Filter out rows whose image file does not exist on disk.
        # ------------------------------------------------------------------
        valid_mask: List[bool] = []
        for fname in df["filename"]:
            img_path = self.images_dir / fname
            if img_path.is_file():
                valid_mask.append(True)
            else:
                print(f"[WARNING] Image not found, skipping: {img_path}")
                valid_mask.append(False)

        df = df.loc[valid_mask].reset_index(drop=True)

        # ------------------------------------------------------------------
        # Optional train / val / test split (deterministic, seed=42).
        # ------------------------------------------------------------------
        if split is not None:
            n = len(df)
            indices = np.random.RandomState(42).permutation(n)
            n_train = int(n * 0.7)
            n_val = int(n * 0.85)
            if split == "train":
                df = df.iloc[indices[:n_train]].reset_index(drop=True)
            elif split == "val":
                df = df.iloc[indices[n_train:n_val]].reset_index(drop=True)
            else:
                df = df.iloc[indices[n_val:]].reset_index(drop=True)

        self.filenames: List[str] = df["filename"].tolist()
        self.labels: np.ndarray = df[self.LABEL_NAMES].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        img_path = self.images_dir / self.filenames[idx]

        # Load with OpenCV (BGR) then convert to RGB for Albumentations.
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label_tensor

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def get_pos_weight(self) -> torch.Tensor:
        """Compute per-label ``pos_weight`` for ``BCEWithLogitsLoss``.

        For each label column the weight is calculated as::

            pos_weight = num_negatives / num_positives

        Returns
        -------
        torch.Tensor
            Float32 tensor of shape ``[num_labels]``.
        """
        positives = self.labels.sum(axis=0)
        negatives = len(self) - positives
        # Guard against division by zero (label never positive).
        pos_weight = np.where(positives > 0, negatives / positives, 0.0)
        return torch.tensor(pos_weight, dtype=torch.float32)

    def get_label_distribution(self) -> Dict[str, Dict[str, int]]:
        """Return per-label positive / negative counts.

        Returns
        -------
        dict
            ``{label_name: {"positive": int, "negative": int}}``
        """
        total = len(self)
        positives = self.labels.sum(axis=0).astype(int)

        distribution: Dict[str, Dict[str, int]] = {}
        for name, pos in zip(self.LABEL_NAMES, positives):
            distribution[name] = {
                "positive": int(pos),
                "negative": total - int(pos),
            }
        return distribution
