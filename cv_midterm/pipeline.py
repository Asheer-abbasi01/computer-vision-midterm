from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class PipelineConfig:
    filter_type: str = "gaussian"
    edge_method: str = "canny"
    rotation_deg: float = 0.0
    translate_x: float = 0.0
    translate_y: float = 0.0
    num_disparities: int = 64
    block_size: int = 15


@dataclass(frozen=True)
class PipelineResult:
    images: Dict[str, np.ndarray]
    summary: str


def _load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file was not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image file could not be decoded: {path}")
    return image


def generate_synthetic_stereo_pair(width: int = 320, height: int = 240, shift: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)

    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[..., 1] = (120 + 100 * xv).astype(np.uint8)
    base[..., 2] = (60 + 120 * yv).astype(np.uint8)

    cv2.rectangle(base, (30, 40), (130, 170), (220, 220, 220), -1)
    cv2.circle(base, (220, 120), 45, (255, 180, 80), -1)
    cv2.line(base, (0, height - 20), (width, height - 20), (90, 90, 90), 5)

    right = np.roll(base, shift=shift, axis=1)
    right[:, :shift, :] = right[:, shift : shift + 1, :]
    return base, right


def _apply_filter(gray: np.ndarray, filter_type: str) -> np.ndarray:
    filter_type = filter_type.lower()
    if filter_type == "gaussian":
        return cv2.GaussianBlur(gray, (5, 5), 0)
    if filter_type == "median":
        return cv2.medianBlur(gray, 5)
    if filter_type == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(gray, -1, kernel)
    raise ValueError(f"Unsupported filter_type: {filter_type}")


def _apply_edge(gray: np.ndarray, edge_method: str) -> np.ndarray:
    edge_method = edge_method.lower()
    if edge_method == "canny":
        return cv2.Canny(gray, 80, 160)
    if edge_method == "sobel":
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(sx, sy)
        return cv2.convertScaleAbs(mag)
    if edge_method == "laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        return cv2.convertScaleAbs(lap)
    raise ValueError(f"Unsupported edge_method: {edge_method}")


def _transform_image(gray: np.ndarray, rotation_deg: float, translate_x: float, translate_y: float) -> np.ndarray:
    h, w = gray.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
    rot[:, 2] += [translate_x, translate_y]
    return cv2.warpAffine(gray, rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _normalize_disparity_params(num_disparities: int, block_size: int) -> Tuple[int, int]:
    num_disparities = max(16, int(num_disparities / 16) * 16)
    block_size = max(5, block_size | 1)
    return num_disparities, block_size


def _compute_disparity(left_gray: np.ndarray, right_gray: np.ndarray, num_disparities: int, block_size: int) -> np.ndarray:
    num_disparities, block_size = _normalize_disparity_params(num_disparities, block_size)
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    return disparity


def _disparity_to_depth(disparity: np.ndarray) -> np.ndarray:
    valid = disparity > 0
    depth = np.zeros_like(disparity, dtype=np.float32)
    depth[valid] = 1.0 / (disparity[valid] + 1e-6)
    return depth


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray


def run_pipeline(left_path: str | Path, right_path: str | Path, config: PipelineConfig | None = None) -> PipelineResult:
    config = config or PipelineConfig()

    left = _load_image(left_path)
    right = _load_image(right_path)

    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    filtered = _apply_filter(left_gray, config.filter_type)
    edges = _apply_edge(filtered, config.edge_method)

    transformed_right = _transform_image(right_gray, config.rotation_deg, config.translate_x, config.translate_y)
    disparity = _compute_disparity(left_gray, transformed_right, config.num_disparities, config.block_size)
    depth = _disparity_to_depth(disparity)

    valid_mask = disparity > 0
    disp_vis = np.zeros_like(left_gray)
    if np.any(valid_mask):
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)

    near_ratio = float(np.mean(disparity[valid_mask] > np.percentile(disparity[valid_mask], 75))) if np.any(valid_mask) else 0.0
    mean_disp = float(np.mean(disparity[valid_mask])) if np.any(valid_mask) else 0.0

    summary = (
        f"Valid disparity pixels: {int(valid_mask.sum())} | "
        f"Mean disparity: {mean_disp:.2f} | "
        f"Near-region ratio: {near_ratio:.2%}"
    )

    images = {
        "left_image": left,
        "right_image": right,
        "filtered_output": _to_bgr(filtered),
        "edge_output": _to_bgr(edges),
        "transformed_right": _to_bgr(transformed_right),
        "disparity_map": disparity_color,
        "depth_map": depth_color,
    }

    return PipelineResult(images=images, summary=summary)
