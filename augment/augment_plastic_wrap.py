from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import albumentations as A
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "annotations.csv"
BACKUP_PATH = PROJECT_ROOT / "data" / "annotations.csv.bak"
IMAGES_DIR = PROJECT_ROOT / "data" / "combined_dataset"

AUG_PER_IMAGE = 5
LABEL_NAMES = ["damaged", "open", "sealed", "plastic_wrap", "non_package"]
OUT_PREFIX = "pw_aug_"
IMG_EXT = ".jpg"


def build_transform(image_size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(
                limit=12,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.6,
            ),
            A.ShiftScaleRotate(
                translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
                scale=(0.92, 1.06),
                rotate_limit=10,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.6,
            ),
            A.Perspective(scale=(0.02, 0.04), p=0.15),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.18, contrast_limit=0.18
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=8,
                        sat_shift_limit=20,
                        val_shift_limit=20,
                    ),
                    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
                    A.CLAHE(clip_limit=2.0),
                    A.RandomGamma(gamma_limit=(80, 120)),
                ],
                p=0.9,
            ),
            A.OneOf(
                [
                    A.GaussNoise(p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.MotionBlur(blur_limit=5),
                ],
                p=0.25,
            ),
        ]
    )


def read_sources() -> tuple[list[str], list[dict]]:
    with CSV_PATH.open(newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        rows = list(reader)
    sources = []
    for row in rows:
        if row.get("plastic_wrap") != "1":
            continue
        fname = row["filename"]
        if fname.startswith(OUT_PREFIX):
            continue
        if not (IMAGES_DIR / fname).exists():
            print(f"  skip (missing image): {fname}", file=sys.stderr)
            continue
        sources.append((fname, row))
    return header, sources


def write_csv_atomic(header: list[str], rows: list[dict]) -> None:
    if not BACKUP_PATH.exists():
        BACKUP_PATH.write_bytes(CSV_PATH.read_bytes())
    import tempfile

    fd, tmp = tempfile.mkstemp(dir=str(CSV_PATH.parent))
    try:
        with os.fdopen(fd, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, CSV_PATH)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline augment plastic_wrap images.")
    ap.add_argument("--aug-per-image", type=int, default=AUG_PER_IMAGE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="Regenerate (drops existing pw_aug_ rows/files).")
    args = ap.parse_args()

    header, sources = read_sources()
    if not header:
        print("ERROR: could not read annotations.csv header", file=sys.stderr)
        return

    if args.force and not args.dry_run:
        existing = []
        with CSV_PATH.open(newline="") as fh:
            reader = csv.DictReader(fh)
            h = reader.fieldnames or []
            for r in reader:
                if r["filename"].startswith(OUT_PREFIX):
                    p = IMAGES_DIR / r["filename"]
                    if p.exists():
                        p.unlink()
                else:
                    existing.append(r)
        write_csv_atomic(h, existing)
        print(f"--force: removed prior pw_aug_ outputs ({len(existing)} base rows remain)")

    existing_filenames: set[str] = set()
    with CSV_PATH.open(newline="") as fh:
        for r in csv.DictReader(fh):
            existing_filenames.add(r["filename"])

    transform = build_transform()
    total_written = 0
    total_rows = 0

    for src_idx, (src_name, src_row) in enumerate(sources):
        img_bgr = cv2.imread(str(IMAGES_DIR / src_name))
        if img_bgr is None:
            print(f"  skip (unreadable): {src_name}", file=sys.stderr)
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        seed_base = abs(hash(src_name)) % (2**32)
        random.seed(seed_base)

        for k in range(1, args.aug_per_image + 1):
            out_name = f"{OUT_PREFIX}{src_idx:03d}_{k:02d}{IMG_EXT}"
            out_path = IMAGES_DIR / out_name

            if out_path.exists() and not args.force:
                continue
            if out_name in existing_filenames and not args.force:
                continue

            random.seed((seed_base + k) % (2**32))
            aug_rgb = transform(image=img_rgb)["image"]
            aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)

            if args.dry_run:
                print(f"  [dry-run] would write {out_name} (from {src_name})")
                total_written += 1
                total_rows += 1
                continue

            cv2.imwrite(str(out_path), aug_bgr)
            new_row = {key: src_row.get(key, "0") for key in header}
            new_row["filename"] = out_name

            rows = []
            with CSV_PATH.open(newline="") as fh:
                reader = csv.DictReader(fh)
                h = reader.fieldnames or []
                rows = list(reader)
            rows.append(new_row)
            write_csv_atomic(h, rows)
            existing_filenames.add(out_name)
            total_written += 1
            total_rows += 1

    if args.dry_run:
        print(f"\n[dry-run] sources={len(sources)} -> would add {total_written} images and {total_rows} CSV rows")
    else:
        print(f"\nDone. sources={len(sources)} -> wrote {total_written} images, appended {total_rows} CSV rows")


if __name__ == "__main__":
    main()
