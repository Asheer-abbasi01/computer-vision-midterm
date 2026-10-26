from pathlib import Path

import pytest

from cv_midterm.config import CVConfig



def test_config_rejects_even_window() -> None:
    cfg = CVConfig(
        left_image=Path("left.png"),
        right_image=Path("right.png"),
        ground_truth_disparity=None,
        output_dir=Path("out"),
        window_size=4,
    )
    with pytest.raises(ValueError):
        cfg.validate()
