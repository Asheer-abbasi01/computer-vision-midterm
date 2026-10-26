"""Configuration models for the CV pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CVConfig:
    """Runtime configuration for the project pipeline."""

    left_image: Path
    right_image: Path
    ground_truth_disparity: Path | None
    output_dir: Path
    focal_length_px: float = 700.0
    baseline_m: float = 0.1
    disparity_min: int = 0
    disparity_max: int = 32
    window_size: int = 3
    large_window_size: int = 21

    def validate(self) -> None:
        if self.window_size % 2 == 0 or self.window_size < 3:
            raise ValueError("window_size must be an odd integer >= 3")
        if self.large_window_size % 2 == 0 or self.large_window_size < 3:
            raise ValueError("large_window_size must be an odd integer >= 3")
        if self.disparity_min < 0 or self.disparity_max <= self.disparity_min:
            raise ValueError("Invalid disparity range. Expected 0 <= min < max.")
        if self.focal_length_px <= 0 or self.baseline_m <= 0:
            raise ValueError("focal_length_px and baseline_m must be positive.")
