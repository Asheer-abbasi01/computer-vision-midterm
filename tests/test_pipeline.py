from pathlib import Path

import cv2

from cv_midterm.cli import main as cli_main
from cv_midterm.pipeline import PipelineConfig, generate_synthetic_stereo_pair, run_pipeline


def test_pipeline_runs_with_synthetic_data(tmp_path: Path) -> None:
    left, right = generate_synthetic_stereo_pair(width=220, height=180, shift=10)
    left_path = tmp_path / "left.png"
    right_path = tmp_path / "right.png"
    cv2.imwrite(str(left_path), left)
    cv2.imwrite(str(right_path), right)

    result = run_pipeline(
        left_path,
        right_path,
        PipelineConfig(filter_type="gaussian", edge_method="canny", num_disparities=64, block_size=15),
    )

    assert "Mean disparity" in result.summary
    assert result.images["disparity_map"].shape == left.shape
    assert result.images["depth_map"].shape == left.shape


def test_cli_synthetic_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    exit_code = cli_main(["--synthetic", "--no-plot", "--output-dir", str(output_dir)])
    assert exit_code == 0
    assert (output_dir / "analysis_summary.txt").exists()
    assert (output_dir / "disparity_map.png").exists()
