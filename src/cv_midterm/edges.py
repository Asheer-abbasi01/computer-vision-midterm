"""Edge detection and gradient analysis methods."""

from __future__ import annotations

import cv2
import numpy as np



def _as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



def sobel_xy(image: np.ndarray, ksize: int = 3) -> tuple[np.ndarray, np.ndarray]:
    gray = _as_gray(image)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    return sobel_x, sobel_y



def gradient_magnitude_orientation(grad_x: np.ndarray, grad_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    magnitude = np.hypot(grad_x, grad_y)
    orientation = np.arctan2(grad_y, grad_x)
    return magnitude, orientation



def prewitt(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = _as_gray(image).astype(np.float32)
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    px = cv2.filter2D(gray, -1, kernel_x)
    py = cv2.filter2D(gray, -1, kernel_y)
    mag = np.hypot(px, py)
    return px, py, mag



def laplacian(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    gray = _as_gray(image)
    return cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)



def log_edges(image: np.ndarray, blur_ksize: int = 5) -> np.ndarray:
    gray = _as_gray(image)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return cv2.Laplacian(blurred, cv2.CV_64F)



def canny(image: np.ndarray, threshold1: int = 50, threshold2: int = 150) -> np.ndarray:
    gray = _as_gray(image)
    return cv2.Canny(gray, threshold1, threshold2)



def threshold_magnitude(magnitude: np.ndarray, threshold: float) -> np.ndarray:
    return (magnitude >= threshold).astype(np.uint8) * 255



def non_maximum_suppression(magnitude: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    if magnitude.shape != orientation.shape:
        raise ValueError("magnitude and orientation must have the same shape")
    h, w = magnitude.shape
    out = np.zeros_like(magnitude, dtype=np.float32)
    angle = (np.rad2deg(orientation) + 180) % 180

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            q = 0.0
            r = 0.0
            a = angle[y, x]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q = magnitude[y, x + 1]
                r = magnitude[y, x - 1]
            elif 22.5 <= a < 67.5:
                q = magnitude[y + 1, x - 1]
                r = magnitude[y - 1, x + 1]
            elif 67.5 <= a < 112.5:
                q = magnitude[y + 1, x]
                r = magnitude[y - 1, x]
            else:
                q = magnitude[y - 1, x - 1]
                r = magnitude[y + 1, x + 1]

            if magnitude[y, x] >= q and magnitude[y, x] >= r:
                out[y, x] = magnitude[y, x]

    return out
