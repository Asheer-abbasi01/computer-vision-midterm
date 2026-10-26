import numpy as np

from cv_midterm.filtering import custom_convolution, gaussian_blur



def test_custom_convolution_identity_kernel_keeps_image() -> None:
    image = np.arange(25, dtype=np.uint8).reshape(5, 5)
    kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    out = custom_convolution(image, kernel)
    assert np.array_equal(out, image)



def test_gaussian_blur_preserves_shape() -> None:
    image = np.random.randint(0, 255, (20, 20), dtype=np.uint8)
    out = gaussian_blur(image, 5)
    assert out.shape == image.shape
