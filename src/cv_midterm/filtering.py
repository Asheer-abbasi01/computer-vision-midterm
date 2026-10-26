"""Filtering, convolution, and sharpening utilities."""

from __future__ import annotations

import cv2
import numpy as np



def gaussian_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    if ksize % 2 == 0 or ksize < 3:
        raise ValueError("ksize must be odd and >= 3")
    return cv2.GaussianBlur(image, (ksize, ksize), 0)



def median_filter(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    if ksize % 2 == 0 or ksize < 3:
        raise ValueError("ksize must be odd and >= 3")
    return cv2.medianBlur(image, ksize)



def low_pass_filter(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    kernel = np.ones((ksize, ksize), dtype=np.float32) / float(ksize * ksize)
    return cv2.filter2D(image, -1, kernel)



def high_pass_filter(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)



def bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: int = 75, sigma_space: int = 75) -> np.ndarray:
    if d <= 0:
        raise ValueError("d must be > 0")
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)



def custom_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if kernel.ndim != 2 or kernel.size == 0:
        raise ValueError("kernel must be a non-empty 2D array.")
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("kernel dimensions must be odd.")
    return cv2.filter2D(image, -1, kernel.astype(np.float32))



def sharpen(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return custom_convolution(image, kernel)
