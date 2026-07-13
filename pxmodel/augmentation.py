from __future__ import annotations

from typing import List

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from pxmodel.labels import LABEL_NAMES

IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

DEFAULT_IMAGE_SIZE: int = 224


def _tail_transforms() -> List[A.BasicTransform]:
    return [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]


def get_train_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.7, 1.0),
            ),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)},
                scale=(0.85, 1.15),
                rotate=(-20, 20),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.7,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3, contrast_limit=0.3
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=25,
                        val_shift_limit=25,
                    ),
                    A.CLAHE(clip_limit=3.0),
                ],
                p=0.7,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=7),
                    A.GaussNoise(p=1.0),
                ],
                p=0.3,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(0.02, 0.15),
                hole_width_range=(0.02, 0.15),
                fill=0,
                p=0.3,
            ),
            *_tail_transforms(),
        ]
    )


def get_val_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            *_tail_transforms(),
        ]
    )


def get_tta_transforms(image_size: int) -> List[A.Compose]:
    tail = _tail_transforms

    original = A.Compose(
        [
            A.Resize(image_size, image_size),
            *tail(),
        ]
    )

    hflip = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=1.0),
            *tail(),
        ]
    )

    rot_pos = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Rotate(
                limit=(10, 10),
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0,
            ),
            *tail(),
        ]
    )

    rot_neg = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Rotate(
                limit=(-10, -10),
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0,
            ),
            *tail(),
        ]
    )

    bright = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.RandomBrightnessContrast(
                brightness_limit=(0.1, 0.1),
                contrast_limit=(0, 0),
                p=1.0,
            ),
            *tail(),
        ]
    )

    return [original, hflip, rot_pos, rot_neg, bright]
