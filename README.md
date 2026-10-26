# Computer Vision Midterm — Classical Stereo Image Pipeline

A complete, dataset-based Python project that implements core Computer Vision lab topics in one coherent pipeline using **OpenCV**, **NumPy**, and **Matplotlib**.

## Why this project

This project demonstrates a realistic safety-oriented stereo vision workflow: preprocess stereo images, extract edges/features, apply geometric transformations, estimate disparity/depth, and analyze near/far obstacle regions.

## Lab-topic mapping

- **Lab 1 (Fundamentals):** read/save images, grayscale, channels, brightness/contrast, thresholding
- **Lab 2 (Filtering):** Gaussian, median, low-pass, high-pass, bilateral, custom convolution, sharpening
- **Lab 3 (Edges):** Sobel X/Y, gradient magnitude/orientation, thresholded gradients, non-maximum suppression, Prewitt, Laplacian, LoG, Canny
- **Lab 4 (Transforms):** resize, crop, rotate, flip, translation, scaling, shearing, affine, perspective
- **Lab 5 (Stereo/Depth):** SSD correspondence (configurable windows, including 3x3 and larger), disparity maps, depth maps, obstacle-region analysis, optional ground-truth comparison

## Prerequisites

- Python 3.10+
- pip

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project structure

```text
src/cv_midterm/
  cli.py
  config.py
  dataset.py
  fundamentals.py
  filtering.py
  edges.py
  transforms.py
  stereo.py
  pipeline.py
tests/
scripts/download_middlebury_tsukuba.py
```

## Dataset usage (no camera required)

### Preferred real dataset (Middlebury Tsukuba-style layout)

Expected layout:

```text
data/
  tsukuba/
    left.png
    right.png
    disparity_gt.png   # optional but recommended
```

You can either:
1. Download from Middlebury manually and place files in the layout above, or
2. Use helper script with explicit URLs:

```bash
PYTHONPATH=src python scripts/download_middlebury_tsukuba.py \
  --left-url "<left-image-url>" \
  --right-url "<right-image-url>" \
  --gt-url "<gt-disparity-url>"
```

### Built-in synthetic dataset (for smoke tests)

```bash
PYTHONPATH=src python -m cv_midterm prepare-dataset --mode synthetic --dataset-root data --scene synthetic
```

## Run commands

### End-to-end pipeline

```bash
PYTHONPATH=src python -m cv_midterm run \
  --stage full \
  --left data/synthetic/left.png \
  --right data/synthetic/right.png \
  --gt data/synthetic/disparity_gt.png \
  --output outputs/smoke
```

### Run a specific stage

```bash
PYTHONPATH=src python -m cv_midterm run --stage stereo --left data/synthetic/left.png --right data/synthetic/right.png --output outputs/stereo_only
```

## Output explanation

Outputs are saved in the selected output directory, grouped by stage prefix:

- `01_*` fundamentals
- `02_*` filtering/convolution
- `03_*` edge detection
- `04_*` transformations
- `05_*` stereo/depth
- `summary.txt` near/far analysis
- `05_ground_truth_comparison.txt` (if GT provided and size matches)

## Tests

```bash
PYTHONPATH=src pytest -q
```

Tests use synthetic arrays/images only and do not require dataset download.

## Limitations

- SSD stereo matching is educational and not optimized for speed.
- Exact metric depth depends on correct focal length/baseline calibration.
- This implementation is designed for lab demonstration and clarity.

## Troubleshooting

- **Image not found / decode failed:** check dataset paths and file names.
- **Mismatched stereo dimensions:** ensure left/right images are same size.
- **Poor disparity quality:** try larger matching window and cleaner/rectified inputs.
- **No GT report:** provide `--gt` image with matching shape.
