"""Command line interface for the CV midterm project."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import CVConfig
from .dataset import download_stereo_files, expected_layout, generate_synthetic_stereo_scene
from .pipeline import run_stage



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Computer Vision Midterm Pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run full pipeline or a selected stage")
    run.add_argument("--stage", default="full", choices=["full", "fundamentals", "filtering", "edges", "transforms", "stereo"])
    run.add_argument("--left", type=Path, required=True)
    run.add_argument("--right", type=Path, required=True)
    run.add_argument("--gt", type=Path, default=None)
    run.add_argument("--output", type=Path, default=Path("outputs"))
    run.add_argument("--focal-length", type=float, default=700.0)
    run.add_argument("--baseline", type=float, default=0.1)
    run.add_argument("--disp-min", type=int, default=0)
    run.add_argument("--disp-max", type=int, default=32)
    run.add_argument("--window", type=int, default=3)
    run.add_argument("--large-window", type=int, default=21)

    prep = sub.add_parser("prepare-dataset", help="Generate synthetic data or download files")
    prep.add_argument("--mode", choices=["synthetic", "download"], default="synthetic")
    prep.add_argument("--dataset-root", type=Path, default=Path("data"))
    prep.add_argument("--scene", default="tsukuba")
    prep.add_argument("--left-url", default=None)
    prep.add_argument("--right-url", default=None)
    prep.add_argument("--gt-url", default=None)

    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare-dataset":
        layout = expected_layout(args.dataset_root, args.scene)
        if args.mode == "synthetic":
            out = generate_synthetic_stereo_scene(layout["left"].parent)
            print(f"Synthetic stereo dataset generated: {out}")
            return

        urls = {}
        if args.left_url:
            urls["left"] = args.left_url
        if args.right_url:
            urls["right"] = args.right_url
        if args.gt_url:
            urls["gt"] = args.gt_url
        download_stereo_files(layout, urls)
        print(f"Dataset files downloaded to {layout['left'].parent}")
        return

    config = CVConfig(
        left_image=args.left,
        right_image=args.right,
        ground_truth_disparity=args.gt,
        output_dir=args.output,
        focal_length_px=args.focal_length,
        baseline_m=args.baseline,
        disparity_min=args.disp_min,
        disparity_max=args.disp_max,
        window_size=args.window,
        large_window_size=args.large_window,
    )
    result = run_stage(config, args.stage)
    print("Pipeline completed.")
    print(result)


if __name__ == "__main__":
    main()
