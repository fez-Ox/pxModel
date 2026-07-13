"""Canonical label schema shared by training, evaluation, and inference."""

from __future__ import annotations

LABEL_NAMES: list[str] = [
    "damaged",
    "plastic_wrap",
    "sealed",
    "open",
    "non_package",
]
NUM_LABELS = len(LABEL_NAMES)


def require_current_label_count(num_labels: int, source: str = "Model") -> None:
    """Reject artifacts trained with a label schema other than the current one."""
    if num_labels != NUM_LABELS:
        raise ValueError(
            f"{source} has {num_labels} outputs, but the current dataset requires "
            f"{NUM_LABELS} ({', '.join(LABEL_NAMES)}). Retrain and re-export the "
            "model with the fifth 'non_package' class."
        )
