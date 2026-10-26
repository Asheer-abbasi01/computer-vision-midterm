import numpy as np
import pytest

from cv_midterm.transforms import crop, translate



def test_translate_moves_pixel() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    image[4, 4] = 255
    moved = translate(image, tx=2, ty=0)
    assert moved[4, 6] == 255



def test_crop_bounds_validation() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        crop(image, 8, 8, 5, 5)
