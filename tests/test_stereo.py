import numpy as np

from cv_midterm.stereo import depth_from_disparity, ssd_disparity



def test_ssd_disparity_recovers_constant_shift() -> None:
    h, w = 30, 40
    d = 4
    left = np.tile(np.arange(w, dtype=np.uint8), (h, 1))
    right = np.zeros_like(left)
    right[:, :-d] = left[:, d:]

    disp = ssd_disparity(left, right, disparity_min=0, disparity_max=8, window_size=3)
    core = disp[5:-5, 12:-5]
    assert np.median(core) == d



def test_depth_handles_zero_disparity_as_inf() -> None:
    disparity = np.array([[0.0, 2.0]], dtype=np.float32)
    depth = depth_from_disparity(disparity, focal_length_px=100.0, baseline_m=0.2)
    assert np.isinf(depth[0, 0])
    assert depth[0, 1] == 10.0
