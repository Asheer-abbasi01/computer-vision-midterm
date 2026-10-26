"""Dataset download and synthetic data helpers."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np

from .io_utils import ensure_dir



def expected_layout(dataset_root: Path, scene: str = "tsukuba") -> dict[str, Path]:
    scene_dir = dataset_root / scene
    return {
        "left": scene_dir / "left.png",
        "right": scene_dir / "right.png",
        "gt": scene_dir / "disparity_gt.png",
    }



def download_stereo_files(target_paths: dict[str, Path], urls: dict[str, str]) -> None:
    missing = [k for k in ("left", "right") if k not in urls]
    if missing:
        raise ValueError(f"Missing URLs for required keys: {missing}")

    for key, path in target_paths.items():
        if key not in urls:
            continue
        ensure_dir(path.parent)
        try:
            urlretrieve(urls[key], str(path))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to download {key} from {urls[key]}: {exc}") from exc

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Downloaded file is not a readable image: {path}")



def generate_synthetic_stereo_scene(output_dir: Path, width: int = 160, height: int = 120, disparity: int = 6) -> dict[str, Path]:
    if disparity <= 0:
        raise ValueError("disparity must be > 0")
    ensure_dir(output_dir)

    left = np.zeros((height, width, 3), dtype=np.uint8)
    gradient = np.tile(np.linspace(30, 180, width, dtype=np.uint8), (height, 1))
    left[..., 1] = gradient
    left[..., 2] = 255 - gradient

    cv2.rectangle(left, (40, 30), (90, 85), (240, 240, 240), -1)
    cv2.circle(left, (120, 70), 18, (180, 80, 220), -1)

    right = np.zeros_like(left)
    right[:, :-disparity] = left[:, disparity:]

    gt = np.zeros((height, width), dtype=np.uint8)
    valid = np.any(right > 0, axis=2)
    gt[valid] = disparity

    left_path = output_dir / "left.png"
    right_path = output_dir / "right.png"
    gt_path = output_dir / "disparity_gt.png"

    cv2.imwrite(str(left_path), left)
    cv2.imwrite(str(right_path), right)
    cv2.imwrite(str(gt_path), gt)

    return {"left": left_path, "right": right_path, "gt": gt_path}
