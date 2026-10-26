"""Stereo correspondence, disparity, and depth analysis."""

from __future__ import annotations

import cv2
import numpy as np



def preprocess_stereo_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if left.shape[:2] != right.shape[:2]:
        raise ValueError("Stereo images must have matching dimensions")
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
    left_f = cv2.equalizeHist(cv2.GaussianBlur(left_gray, (3, 3), 0))
    right_f = cv2.equalizeHist(cv2.GaussianBlur(right_gray, (3, 3), 0))
    return left_f, right_f



def ssd_disparity(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    disparity_min: int = 0,
    disparity_max: int = 32,
    window_size: int = 3,
) -> np.ndarray:
    if left_gray.shape != right_gray.shape:
        raise ValueError("left_gray and right_gray must have identical shapes")
    if left_gray.ndim != 2:
        raise ValueError("ssd_disparity expects grayscale images")
    if window_size % 2 == 0 or window_size < 3:
        raise ValueError("window_size must be odd and >= 3")
    if disparity_min < 0 or disparity_max <= disparity_min:
        raise ValueError("Invalid disparity range")

    left_f = left_gray.astype(np.float32)
    right_f = right_gray.astype(np.float32)
    h, w = left_f.shape
    half = window_size // 2
    disp = np.zeros((h, w), dtype=np.float32)

    for y in range(half, h - half):
        for x in range(half + disparity_max, w - half):
            left_patch = left_f[y - half : y + half + 1, x - half : x + half + 1]
            best_d = disparity_min
            best_score = np.inf
            for d in range(disparity_min, disparity_max + 1):
                xr = x - d
                if xr - half < 0:
                    break
                right_patch = right_f[y - half : y + half + 1, xr - half : xr + half + 1]
                score = float(np.sum((left_patch - right_patch) ** 2))
                if score < best_score:
                    best_score = score
                    best_d = d
            disp[y, x] = best_d

    return disp



def depth_from_disparity(disparity: np.ndarray, focal_length_px: float, baseline_m: float) -> np.ndarray:
    if focal_length_px <= 0 or baseline_m <= 0:
        raise ValueError("focal_length_px and baseline_m must be positive")
    disparity = disparity.astype(np.float32)
    depth = np.full_like(disparity, np.inf, dtype=np.float32)
    valid = disparity > 0
    depth[valid] = (focal_length_px * baseline_m) / disparity[valid]
    return depth



def colorize_depth(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth)
    if not np.any(finite):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    min_d = float(depth[finite].min())
    max_d = float(depth[finite].max())
    if np.isclose(min_d, max_d):
        scaled = np.zeros_like(depth, dtype=np.uint8)
    else:
        scaled = np.zeros_like(depth, dtype=np.float32)
        scaled[finite] = (depth[finite] - min_d) / (max_d - min_d)
        scaled = (255 * (1.0 - scaled)).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)



def near_far_analysis(depth: np.ndarray) -> dict[str, float | int]:
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return {"near_threshold": float("nan"), "far_threshold": float("nan"), "near_pixels": 0, "far_pixels": 0}
    near_t = float(np.percentile(finite, 25))
    far_t = float(np.percentile(finite, 75))
    near_pixels = int(np.sum((depth <= near_t) & np.isfinite(depth)))
    far_pixels = int(np.sum((depth >= far_t) & np.isfinite(depth)))
    return {
        "near_threshold": near_t,
        "far_threshold": far_t,
        "near_pixels": near_pixels,
        "far_pixels": far_pixels,
    }



def obstacle_region_mask(depth: np.ndarray, near_threshold: float) -> np.ndarray:
    mask = ((depth <= near_threshold) & np.isfinite(depth)).astype(np.uint8) * 255
    return mask



def compare_disparity_to_ground_truth(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    if pred.shape != gt.shape:
        raise ValueError("Predicted and ground-truth disparity must have matching shape")
    gt_f = gt.astype(np.float32)
    pred_f = pred.astype(np.float32)
    valid = gt_f > 0
    if not np.any(valid):
        return {"mae": float("nan"), "rmse": float("nan"), "valid_ratio": 0.0}
    diff = pred_f[valid] - gt_f[valid]
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    ratio = float(np.sum(valid) / valid.size)
    return {"mae": mae, "rmse": rmse, "valid_ratio": ratio}
