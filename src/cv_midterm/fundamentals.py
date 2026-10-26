"""Image fundamentals and basic preprocessing."""

from __future__ import annotations

import cv2
import numpy as np



def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.copy()
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a grayscale or BGR image with 3 channels.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



def color_channels(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR image with 3 channels.")
    b, g, r = cv2.split(image)
    return b, g, r



def adjust_brightness_contrast(image: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    if alpha <= 0:
        raise ValueError("alpha (contrast) must be > 0")
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted



def binary_threshold(gray: np.ndarray, threshold: int = 127) -> np.ndarray:
    if gray.ndim != 2:
        raise ValueError("binary_threshold expects a grayscale image.")
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be in [0, 255].")
    _, out = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return out
