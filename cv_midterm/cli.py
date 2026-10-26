from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from .pipeline import PipelineConfig, generate_synthetic_stereo_pair, run_pipeline


def _save_images(images: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        cv2.imwrite(str(output_dir / f"{name}.png"), image)


def _plot_results(images: dict[str, object], summary: str) -> None:
    ordered = [
        ("Left", "left_image"),
        ("Right", "right_image"),
        ("Filtered", "filtered_output"),
        ("Edges", "edge_output"),
        ("Transformed", "transformed_right"),
        ("Disparity", "disparity_map"),
        ("Depth", "depth_map"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for idx, (label, key) in enumerate(ordered):
        ax = axes.flat[idx]
        rgb = cv2.cvtColor(images[key], cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(label)
        ax.axis("off")
    axes.flat[-1].text(0.02, 0.6, summary, fontsize=10)
    axes.flat[-1].axis("off")
    fig.suptitle("Computer Vision Midterm Pipeline", fontsize=14)
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the computer vision midterm pipeline")
    parser.add_argument("--left", type=str, help="Path to left stereo image")
    parser.add_argument("--right", type=str, help="Path to right stereo image")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for output images")
    parser.add_argument("--filter", dest="filter_type", choices=["gaussian", "median", "sharpen"], default="gaussian")
    parser.add_argument("--edge", dest="edge_method", choices=["canny", "sobel", "laplacian"], default="canny")
    parser.add_argument("--rotation", type=float, default=0.0)
    parser.add_argument("--tx", type=float, default=0.0)
    parser.add_argument("--ty", type=float, default=0.0)
    parser.add_argument("--num-disparities", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=15)
    parser.add_argument("--synthetic", action="store_true", help="Use generated synthetic stereo images")
    parser.add_argument("--no-plot", action="store_true", help="Skip interactive plot display")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        left, right = generate_synthetic_stereo_pair()
        left_path = output_dir / "synthetic_left.png"
        right_path = output_dir / "synthetic_right.png"
        cv2.imwrite(str(left_path), left)
        cv2.imwrite(str(right_path), right)
    else:
        if not args.left or not args.right:
            parser.error("--left and --right are required unless --synthetic is used")
        left_path = Path(args.left)
        right_path = Path(args.right)

    config = PipelineConfig(
        filter_type=args.filter_type,
        edge_method=args.edge_method,
        rotation_deg=args.rotation,
        translate_x=args.tx,
        translate_y=args.ty,
        num_disparities=args.num_disparities,
        block_size=args.block_size,
    )

    result = run_pipeline(left_path, right_path, config)
    _save_images(result.images, output_dir)

    summary_path = output_dir / "analysis_summary.txt"
    summary_path.write_text(result.summary + "\n", encoding="utf-8")

    if not args.no_plot:
        _plot_results(result.images, result.summary)

    print(f"Saved outputs to: {output_dir}")
    print(result.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
