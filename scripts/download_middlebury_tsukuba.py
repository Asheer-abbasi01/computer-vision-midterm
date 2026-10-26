#!/usr/bin/env python3
"""Download helper for a Tsukuba-style stereo layout.

Usage example:
python scripts/download_middlebury_tsukuba.py \
  --left-url "<left image url>" \
  --right-url "<right image url>" \
  --gt-url "<ground-truth disparity url>"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cv_midterm.dataset import download_stereo_files, expected_layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data"))
    parser.add_argument("--scene", default="tsukuba")
    parser.add_argument("--left-url", required=True)
    parser.add_argument("--right-url", required=True)
    parser.add_argument("--gt-url", default=None)
    args = parser.parse_args()

    layout = expected_layout(args.dataset_root, args.scene)
    urls = {"left": args.left_url, "right": args.right_url}
    if args.gt_url:
        urls["gt"] = args.gt_url
    download_stereo_files(layout, urls)
    print(f"Downloaded files into: {layout['left'].parent}")


if __name__ == "__main__":
    main()
