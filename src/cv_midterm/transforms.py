"""Geometric image transformation utilities."""

from __future__ import annotations

import cv2
import numpy as np



def resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)



def crop(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    if min(x, y, width, height) < 0 or width == 0 or height == 0:
        raise ValueError("Invalid crop parameters")
    x2, y2 = x + width, y + height
    if x2 > image.shape[1] or y2 > image.shape[0]:
        raise ValueError("Crop region exceeds image bounds")
    return image[y:y2, x:x2]



def rotate(image: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(image, mat, (w, h))



def flip(image: np.ndarray, horizontal: bool = True) -> np.ndarray:
    return cv2.flip(image, 1 if horizontal else 0)



def translate(image: np.ndarray, tx: float, ty: float) -> np.ndarray:
    h, w = image.shape[:2]
    mat = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    return cv2.warpAffine(image, mat, (w, h))



def scale(image: np.ndarray, sx: float, sy: float) -> np.ndarray:
    if sx <= 0 or sy <= 0:
        raise ValueError("Scale factors must be positive")
    h, w = image.shape[:2]
    mat = np.array([[sx, 0, 0], [0, sy, 0]], dtype=np.float32)
    return cv2.warpAffine(image, mat, (w, h))



def shear(image: np.ndarray, shx: float = 0.0, shy: float = 0.0) -> np.ndarray:
    h, w = image.shape[:2]
    mat = np.array([[1, shx, 0], [shy, 1, 0]], dtype=np.float32)
    return cv2.warpAffine(image, mat, (w, h))



def affine_transform(image: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    if src_pts.shape != (3, 2) or dst_pts.shape != (3, 2):
        raise ValueError("src_pts and dst_pts must have shape (3, 2)")
    h, w = image.shape[:2]
    mat = cv2.getAffineTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))
    return cv2.warpAffine(image, mat, (w, h))



def perspective_transform(image: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    if src_pts.shape != (4, 2) or dst_pts.shape != (4, 2):
        raise ValueError("src_pts and dst_pts must have shape (4, 2)")
    h, w = image.shape[:2]
    mat = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))
    return cv2.warpPerspective(image, mat, (w, h))
