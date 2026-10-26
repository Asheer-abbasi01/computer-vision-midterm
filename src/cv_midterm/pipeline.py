"""Pipeline runner for end-to-end and stage-based processing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import CVConfig
from .edges import (
    canny,
    gradient_magnitude_orientation,
    laplacian,
    log_edges,
    non_maximum_suppression,
    prewitt,
    sobel_xy,
    threshold_magnitude,
)
from .filtering import (
    bilateral_filter,
    custom_convolution,
    gaussian_blur,
    high_pass_filter,
    low_pass_filter,
    median_filter,
    sharpen,
)
from .fundamentals import adjust_brightness_contrast, binary_threshold, color_channels, to_grayscale
from .io_utils import ensure_dir, load_image, save_image
from .stereo import (
    colorize_depth,
    compare_disparity_to_ground_truth,
    depth_from_disparity,
    near_far_analysis,
    obstacle_region_mask,
    preprocess_stereo_pair,
    ssd_disparity,
)
from .transforms import affine_transform, crop, flip, perspective_transform, resize, rotate, scale, shear, translate



def _norm_to_u8(arr: np.ndarray) -> np.ndarray:
    return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)



def run_fundamentals(left: np.ndarray, output_dir: Path) -> None:
    gray = to_grayscale(left)
    b, g, r = color_channels(left)
    bright = adjust_brightness_contrast(left, alpha=1.2, beta=20)
    thresh = binary_threshold(gray, 127)
    save_image(output_dir / "01_gray.png", gray)
    save_image(output_dir / "01_channel_b.png", b)
    save_image(output_dir / "01_channel_g.png", g)
    save_image(output_dir / "01_channel_r.png", r)
    save_image(output_dir / "01_brightness_contrast.png", bright)
    save_image(output_dir / "01_binary_threshold.png", thresh)



def run_filtering(left: np.ndarray, output_dir: Path) -> None:
    gauss = gaussian_blur(left, 5)
    median = median_filter(left, 5)
    low = low_pass_filter(left, 5)
    high = high_pass_filter(left)
    bilateral = bilateral_filter(left)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    custom = custom_convolution(left, kernel)
    sharp = sharpen(left)
    save_image(output_dir / "02_gaussian.png", gauss)
    save_image(output_dir / "02_median.png", median)
    save_image(output_dir / "02_low_pass.png", low)
    save_image(output_dir / "02_high_pass.png", high)
    save_image(output_dir / "02_bilateral.png", bilateral)
    save_image(output_dir / "02_custom_convolution.png", custom)
    save_image(output_dir / "02_sharpen.png", sharp)



def run_edges(left: np.ndarray, output_dir: Path) -> None:
    gray = to_grayscale(left)
    sx, sy = sobel_xy(gray)
    mag, orient = gradient_magnitude_orientation(sx, sy)
    nms = non_maximum_suppression(mag, orient)
    mag_t = threshold_magnitude(mag, threshold=float(np.percentile(mag, 75)))
    px, py, pmag = prewitt(gray)
    lap = laplacian(gray)
    log_im = log_edges(gray)
    can = canny(gray)
    save_image(output_dir / "03_sobel_x.png", _norm_to_u8(sx))
    save_image(output_dir / "03_sobel_y.png", _norm_to_u8(sy))
    save_image(output_dir / "03_gradient_magnitude.png", _norm_to_u8(mag))
    save_image(output_dir / "03_gradient_orientation.png", _norm_to_u8(orient))
    save_image(output_dir / "03_gradient_thresholded.png", mag_t)
    save_image(output_dir / "03_nms.png", _norm_to_u8(nms))
    save_image(output_dir / "03_prewitt_x.png", _norm_to_u8(px))
    save_image(output_dir / "03_prewitt_y.png", _norm_to_u8(py))
    save_image(output_dir / "03_prewitt_magnitude.png", _norm_to_u8(pmag))
    save_image(output_dir / "03_laplacian.png", _norm_to_u8(lap))
    save_image(output_dir / "03_log.png", _norm_to_u8(log_im))
    save_image(output_dir / "03_canny.png", can)



def run_transforms(left: np.ndarray, output_dir: Path) -> None:
    h, w = left.shape[:2]
    res = resize(left, max(1, w // 2), max(1, h // 2))
    cr = crop(left, w // 4, h // 4, max(1, w // 2), max(1, h // 2))
    rot = rotate(left, 15)
    flp = flip(left, horizontal=True)
    tr = translate(left, tx=10, ty=5)
    scl = scale(left, 0.9, 0.9)
    shr = shear(left, shx=0.1, shy=0.0)

    src_aff = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    dst_aff = np.float32([[10, 5], [w - 20, 15], [20, h - 10]])
    aff = affine_transform(left, src_aff, dst_aff)

    src_p = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst_p = np.float32([[20, 10], [w - 20, 5], [w - 5, h - 10], [15, h - 5]])
    per = perspective_transform(left, src_p, dst_p)

    save_image(output_dir / "04_resize.png", res)
    save_image(output_dir / "04_crop.png", cr)
    save_image(output_dir / "04_rotate.png", rot)
    save_image(output_dir / "04_flip.png", flp)
    save_image(output_dir / "04_translate.png", tr)
    save_image(output_dir / "04_scale.png", scl)
    save_image(output_dir / "04_shear.png", shr)
    save_image(output_dir / "04_affine.png", aff)
    save_image(output_dir / "04_perspective.png", per)



def run_stereo(config: CVConfig, left: np.ndarray, right: np.ndarray, output_dir: Path) -> dict[str, float | int]:
    left_pre, right_pre = preprocess_stereo_pair(left, right)
    disp_small = ssd_disparity(
        left_pre,
        right_pre,
        disparity_min=config.disparity_min,
        disparity_max=config.disparity_max,
        window_size=config.window_size,
    )
    disp_large = ssd_disparity(
        left_pre,
        right_pre,
        disparity_min=config.disparity_min,
        disparity_max=config.disparity_max,
        window_size=config.large_window_size,
    )

    depth = depth_from_disparity(disp_small, config.focal_length_px, config.baseline_m)
    depth_color = colorize_depth(depth)
    analysis = near_far_analysis(depth)
    obstacle_mask = obstacle_region_mask(depth, float(analysis["near_threshold"]))

    save_image(output_dir / "05_left_preprocessed.png", left_pre)
    save_image(output_dir / "05_right_preprocessed.png", right_pre)
    save_image(output_dir / "05_disparity_w3.png", _norm_to_u8(disp_small))
    save_image(output_dir / "05_disparity_large_window.png", _norm_to_u8(disp_large))
    save_image(output_dir / "05_depth_colormap.png", depth_color)
    save_image(output_dir / "05_obstacle_mask.png", obstacle_mask)

    if config.ground_truth_disparity and Path(config.ground_truth_disparity).exists():
        gt = load_image(Path(config.ground_truth_disparity), grayscale=True)
        if gt.shape == disp_small.shape:
            metrics = compare_disparity_to_ground_truth(disp_small, gt)
            with open(output_dir / "05_ground_truth_comparison.txt", "w", encoding="utf-8") as f:
                for k, v in metrics.items():
                    f.write(f"{k}: {v}\n")

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        for k, v in analysis.items():
            f.write(f"{k}: {v}\n")

    return analysis



def run_full_pipeline(config: CVConfig) -> dict[str, float | int]:
    config.validate()
    ensure_dir(config.output_dir)
    left = load_image(config.left_image)
    right = load_image(config.right_image)

    run_fundamentals(left, config.output_dir)
    run_filtering(left, config.output_dir)
    run_edges(left, config.output_dir)
    run_transforms(left, config.output_dir)
    return run_stereo(config, left, right, config.output_dir)



def run_stage(config: CVConfig, stage: str) -> dict[str, float | int]:
    valid = {"full", "fundamentals", "filtering", "edges", "transforms", "stereo"}
    if stage not in valid:
        raise ValueError(f"Unsupported stage '{stage}'. Choose from {sorted(valid)}")

    config.validate()
    ensure_dir(config.output_dir)
    left = load_image(config.left_image)
    right = load_image(config.right_image)

    if stage == "full":
        return run_full_pipeline(config)
    if stage == "fundamentals":
        run_fundamentals(left, config.output_dir)
        return {}
    if stage == "filtering":
        run_filtering(left, config.output_dir)
        return {}
    if stage == "edges":
        run_edges(left, config.output_dir)
        return {}
    if stage == "transforms":
        run_transforms(left, config.output_dir)
        return {}

    return run_stereo(config, left, right, config.output_dir)
