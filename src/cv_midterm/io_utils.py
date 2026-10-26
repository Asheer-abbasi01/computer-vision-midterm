"""I/O and validation helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np



def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def load_image(path: Path, grayscale: bool = False) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    return image



def save_image(path: Path, image: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise ValueError(f"Failed to save image to: {path}")
