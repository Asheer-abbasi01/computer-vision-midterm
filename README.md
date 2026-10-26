# computer-vision-midterm

Computer Vision lab midterm project implementing a complete classical stereo-image pipeline with both CLI and GUI interfaces.

## Features

- Image input from left/right stereo image files (no camera required)
- Preprocessing and filtering (Gaussian, Median, Sharpen)
- Edge detection (Canny, Sobel, Laplacian)
- Geometric transformations for stereo alignment (rotation + translation)
- Stereo matching with OpenCV StereoBM (configurable disparities and block size)
- Disparity map and depth-map visualization
- Analysis summary for valid disparity coverage and near-region ratio
- Lightweight professional GUI (light neutral palette, sectioned controls, comparison panels)

## Dataset

Use any simple stereo image pair (recommended: Middlebury examples such as Tsukuba or Cones), or run synthetic data from CLI for quick smoke testing.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run CLI

```bash
python -m cv_midterm.cli --left /absolute/path/to/left.png --right /absolute/path/to/right.png --output-dir /absolute/path/to/outputs
```

Quick synthetic smoke run:

```bash
python -m cv_midterm.cli --synthetic --no-plot --output-dir /absolute/path/to/outputs
```

## Run GUI

```bash
python -m cv_midterm.gui
```

Headless GUI smoke test command:

```bash
python -m cv_midterm.gui --smoke-test
```

## Visual Design Choices

The GUI intentionally uses a clean light palette for readability and lab demonstration:

- Off-white main background
- Light gray control panels
- Soft neutral borders
- Restrained accent color for action emphasis
- High-contrast dark text for accessibility

Controls are grouped into sections: image input, preprocessing/filtering, edge detection, transformations, stereo matching, depth analysis, and output handling. The result area is arranged for side-by-side comparison of key outputs (left/right, filtered, edges, transformed, disparity, depth, and summary).

## Tests

Run all tests:

```bash
pytest
```

Included smoke tests cover:

- End-to-end synthetic pipeline execution
- CLI synthetic run and output generation
- Headless GUI smoke mode (`--smoke-test`) without opening an interactive window
