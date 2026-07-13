"""Validate that the checked-out dataset is complete and trainable."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from pxmodel.config import images_dir, train_csv
from pxmodel.labels import LABEL_NAMES

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def validate_dataset(
    csv_path: Path = train_csv,
    image_dir: Path = images_dir,
    *,
    decode_images: bool = False,
) -> tuple[int, dict[str, int]]:
    """Validate the CSV schema, labels, filenames, and checked-out images."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Annotation file is missing: {csv_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory is missing: {image_dir}. Import the dataset images "
            "before validating or training."
        )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected_columns = {"filename", *LABEL_NAMES}
        actual_columns = set(reader.fieldnames or ())
        if actual_columns != expected_columns:
            raise ValueError(
                f"CSV columns are {sorted(actual_columns)}; expected "
                f"{sorted(expected_columns)}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"Annotation file contains no samples: {csv_path}")

    filenames: list[str] = []
    positives = {label: 0 for label in LABEL_NAMES}
    for line_number, row in enumerate(rows, start=2):
        filename = row["filename"]
        if not filename or Path(filename).name != filename:
            raise ValueError(f"Invalid filename on CSV line {line_number}: {filename!r}")
        filenames.append(filename)

        for label in LABEL_NAMES:
            value = row[label]
            if value not in {"0", "1"}:
                raise ValueError(
                    f"Invalid {label} value on CSV line {line_number}: {value!r}"
                )
            positives[label] += int(value)

        if row["non_package"] == "1" and any(
            row[label] == "1" for label in LABEL_NAMES if label != "non_package"
        ):
            raise ValueError(
                f"non_package must be exclusive on CSV line {line_number}"
            )

    duplicates = sorted(
        name for name, count in Counter(filenames).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate CSV filenames (first 5): {duplicates[:5]}")

    annotated = set(filenames)
    images = {
        path.name
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    missing = sorted(annotated - images)
    unannotated = sorted(images - annotated)
    if missing:
        raise FileNotFoundError(f"Annotated images missing (first 5): {missing[:5]}")
    if unannotated:
        raise ValueError(f"Images without annotations (first 5): {unannotated[:5]}")

    if decode_images:
        import cv2

        for filename in filenames:
            if cv2.imread(str(image_dir / filename)) is None:
                raise ValueError(f"Image cannot be decoded: {image_dir / filename}")

    empty_labels = [label for label, count in positives.items() if count == 0]
    if empty_labels:
        raise ValueError(f"Labels with no positive samples: {empty_labels}")

    return len(rows), positives


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decode-images",
        action="store_true",
        help="Decode every image in addition to checking filenames (slower)",
    )
    args = parser.parse_args()

    sample_count, positives = validate_dataset(decode_images=args.decode_images)
    print(f"Dataset ready: {sample_count} images, {len(LABEL_NAMES)} labels")
    for label in LABEL_NAMES:
        print(f"  {label:<15} positives={positives[label]}")


if __name__ == "__main__":
    main()
