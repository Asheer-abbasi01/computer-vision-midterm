import numpy as np

from cv_midterm.edges import gradient_magnitude_orientation, non_maximum_suppression, sobel_xy



def test_non_maximum_suppression_keeps_peak() -> None:
    mag = np.array([[0, 0, 0], [0, 10, 0], [0, 0, 0]], dtype=float)
    ori = np.zeros((3, 3), dtype=float)
    out = non_maximum_suppression(mag, ori)
    assert out[1, 1] == 10



def test_sobel_and_gradient_shapes() -> None:
    img = np.zeros((16, 16), dtype=np.uint8)
    img[:, 8:] = 255
    sx, sy = sobel_xy(img)
    mag, ori = gradient_magnitude_orientation(sx, sy)
    assert sx.shape == sy.shape == mag.shape == ori.shape == img.shape
