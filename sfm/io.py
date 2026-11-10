"""Image loading."""

import re
from pathlib import Path

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _natural_key(path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def list_images(folder, step=1):
    """Image paths sorted by file name (IMG_2 before IMG_10), keeping one image every `step`.

    Sorting by name keeps the capture order of phone photos. The original scripts
    sorted by modification time, which changes when files are copied.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    paths = sorted(
        (p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS),
        key=_natural_key,
    )
    paths = paths[:: max(1, int(step))]
    if len(paths) < 2:
        raise RuntimeError(f"Need at least two images in {folder} (after step={step}).")
    return paths


def read_image(path, max_size=None, color=False):
    """Read an image in grayscale (or BGR), optionally downscaled so its long side <= max_size."""
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(str(path), flag)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    if max_size:
        h, w = img.shape[:2]
        s = max_size / float(max(h, w))
        if s < 1.0:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img
