from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .pipeline import PipelineConfig, run_pipeline

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ModuleNotFoundError:  # pragma: no cover
    tk = None
    filedialog = None
    messagebox = None
    ttk = None

try:
    from PIL import Image, ImageTk
except ModuleNotFoundError:  # pragma: no cover
    Image = None
    ImageTk = None


def get_palette() -> dict[str, str]:
    return {
        "bg": "#fbfbf8",
        "panel": "#f1f2f1",
        "border": "#d5d8d6",
        "text": "#222222",
        "accent": "#2f8f83",
        "accent_soft": "#e8f4f2",
        "warning": "#d97941",
    }


def _validate_gui_runtime() -> None:
    if tk is None or ttk is None or Image is None or ImageTk is None:
        raise RuntimeError(
            "GUI dependencies are not available in this environment. "
            "Install Tkinter and Pillow to launch the interactive GUI."
        )


def launch_gui() -> int:
    _validate_gui_runtime()

    class ToolTip:
        def __init__(self, widget: tk.Widget, text: str) -> None:
            self.widget = widget
            self.text = text
            self.tip: tk.Toplevel | None = None
            widget.bind("<Enter>", self.show)
            widget.bind("<Leave>", self.hide)

        def show(self, _event: tk.Event) -> None:
            if self.tip:
                return
            x, y, _, _ = self.widget.bbox("insert") or (0, 0, 0, 0)
            x += self.widget.winfo_rootx() + 20
            y += self.widget.winfo_rooty() + 20
            self.tip = tk.Toplevel(self.widget)
            self.tip.wm_overrideredirect(True)
            self.tip.wm_geometry(f"+{x}+{y}")
            label = tk.Label(self.tip, text=self.text, bg="#fffaf0", relief="solid", borderwidth=1, padx=6, pady=3)
            label.pack()

        def hide(self, _event: tk.Event) -> None:
            if self.tip:
                self.tip.destroy()
                self.tip = None

    class CVMidtermGUI(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.title("Computer Vision Midterm Pipeline")
            self.minsize(1200, 760)
            self.palette = get_palette()
            self.configure(bg=self.palette["bg"])

            self.left_path = tk.StringVar()
            self.right_path = tk.StringVar()
            self.filter_type = tk.StringVar(value="gaussian")
            self.edge_method = tk.StringVar(value="canny")
            self.rotation = tk.DoubleVar(value=0.0)
            self.tx = tk.DoubleVar(value=0.0)
            self.ty = tk.DoubleVar(value=0.0)
            self.num_disparities = tk.IntVar(value=64)
            self.block_size = tk.IntVar(value=15)

            self._image_refs: dict[str, ImageTk.PhotoImage] = {}
            self._current_result = None

            self._configure_style()
            self._build_layout()

        def _configure_style(self) -> None:
            style = ttk.Style(self)
            style.configure("Main.TFrame", background=self.palette["bg"])
            style.configure("Panel.TLabelframe", background=self.palette["panel"], bordercolor=self.palette["border"])
            style.configure(
                "Panel.TLabelframe.Label",
                background=self.palette["panel"],
                foreground=self.palette["text"],
                font=("Segoe UI", 10, "bold"),
            )
            style.configure("Panel.TLabel", background=self.palette["panel"], foreground=self.palette["text"])
            style.configure("Main.TLabel", background=self.palette["bg"], foreground=self.palette["text"])

        def _build_layout(self) -> None:
            main = ttk.Frame(self, style="Main.TFrame", padding=10)
            main.grid(row=0, column=0, sticky="nsew")
            self.grid_rowconfigure(0, weight=1)
            self.grid_columnconfigure(0, weight=1)
            main.grid_columnconfigure(0, weight=1)
            main.grid_columnconfigure(1, weight=2)
            main.grid_rowconfigure(1, weight=1)

            controls = ttk.LabelFrame(main, text="Pipeline Controls", style="Panel.TLabelframe", padding=10)
            controls.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
            controls.grid_columnconfigure(1, weight=1)

            row = 0
            row = self._section_input(controls, row)
            row = self._section_preprocessing(controls, row)
            row = self._section_edges(controls, row)
            row = self._section_transform(controls, row)
            row = self._section_stereo(controls, row)
            row = self._section_output(controls, row)

            ttk.Button(controls, text="Run Pipeline", command=self.run_pipeline_action).grid(row=row, column=0, columnspan=2, sticky="ew", pady=(12, 6))
            ttk.Button(controls, text="Save Results...", command=self.save_results_action).grid(row=row + 1, column=0, columnspan=2, sticky="ew")

            result_panel = ttk.LabelFrame(main, text="Results Comparison", style="Panel.TLabelframe", padding=10)
            result_panel.grid(row=0, column=1, rowspan=2, sticky="nsew")
            for i in range(4):
                result_panel.grid_columnconfigure(i, weight=1)
            for i in range(2):
                result_panel.grid_rowconfigure(i, weight=1)

            self.result_labels: dict[str, ttk.Label] = {}
            slots = [
                ("Left Image", "left_image"),
                ("Right Image", "right_image"),
                ("Filtered Output", "filtered_output"),
                ("Edge Output", "edge_output"),
                ("Transformed/Aligned", "transformed_right"),
                ("Disparity Map", "disparity_map"),
                ("Depth Map", "depth_map"),
                ("Analysis Summary", "summary"),
            ]
            for idx, (title, key) in enumerate(slots):
                r, c = divmod(idx, 4)
                box = ttk.Frame(result_panel)
                box.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)
                ttk.Label(box, text=title, style="Main.TLabel").pack(anchor="w")
                label = ttk.Label(box, text="No data", style="Main.TLabel", anchor="center")
                label.pack(fill="both", expand=True)
                self.result_labels[key] = label

            self.status = tk.StringVar(value="Select left and right images, then run the pipeline.")
            ttk.Label(main, textvariable=self.status, style="Main.TLabel", wraplength=850).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        def _section_input(self, parent: ttk.LabelFrame, row: int) -> int:
            ttk.Label(parent, text="Image Input", style="Panel.TLabel").grid(row=row, column=0, columnspan=2, sticky="w", pady=(2, 4))
            row += 1
            ttk.Entry(parent, textvariable=self.left_path).grid(row=row, column=0, sticky="ew", padx=(0, 6))
            b_left = ttk.Button(parent, text="Left...", command=self.pick_left)
            b_left.grid(row=row, column=1, sticky="ew")
            ToolTip(b_left, "Choose the left stereo image file")
            row += 1
            ttk.Entry(parent, textvariable=self.right_path).grid(row=row, column=0, sticky="ew", padx=(0, 6), pady=(4, 8))
            b_right = ttk.Button(parent, text="Right...", command=self.pick_right)
            b_right.grid(row=row, column=1, sticky="ew", pady=(4, 8))
            ToolTip(b_right, "Choose the right stereo image file")
            return row + 1

        def _section_preprocessing(self, parent: ttk.LabelFrame, row: int) -> int:
            ttk.Label(parent, text="Preprocessing/Filtering", style="Panel.TLabel").grid(row=row, column=0, sticky="w", pady=(4, 2))
            row += 1
            combo = ttk.Combobox(parent, textvariable=self.filter_type, values=["gaussian", "median", "sharpen"], state="readonly")
            combo.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 8))
            ToolTip(combo, "Select a light preprocessing filter")
            return row + 1

        def _section_edges(self, parent: ttk.LabelFrame, row: int) -> int:
            ttk.Label(parent, text="Edge Detection", style="Panel.TLabel").grid(row=row, column=0, sticky="w", pady=(2, 2))
            row += 1
            combo = ttk.Combobox(parent, textvariable=self.edge_method, values=["canny", "sobel", "laplacian"], state="readonly")
            combo.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 8))
            ToolTip(combo, "Choose edge extraction method")
            return row + 1

        def _section_transform(self, parent: ttk.LabelFrame, row: int) -> int:
            ttk.Label(parent, text="Transformations", style="Panel.TLabel").grid(row=row, column=0, sticky="w", pady=(2, 2))
            row += 1
            ttk.Label(parent, text="Rotate (deg)", style="Panel.TLabel").grid(row=row, column=0, sticky="w")
            ttk.Entry(parent, textvariable=self.rotation, width=8).grid(row=row, column=1, sticky="ew")
            row += 1
            ttk.Label(parent, text="Translate X", style="Panel.TLabel").grid(row=row, column=0, sticky="w")
            ttk.Entry(parent, textvariable=self.tx, width=8).grid(row=row, column=1, sticky="ew")
            row += 1
            ttk.Label(parent, text="Translate Y", style="Panel.TLabel").grid(row=row, column=0, sticky="w")
            ttk.Entry(parent, textvariable=self.ty, width=8).grid(row=row, column=1, sticky="ew", pady=(0, 8))
            return row + 1

        def _section_stereo(self, parent: ttk.LabelFrame, row: int) -> int:
            ttk.Label(parent, text="Stereo Matching", style="Panel.TLabel").grid(row=row, column=0, sticky="w", pady=(2, 2))
            row += 1
            ttk.Label(parent, text="Num disparities", style="Panel.TLabel").grid(row=row, column=0, sticky="w")
            ttk.Entry(parent, textvariable=self.num_disparities).grid(row=row, column=1, sticky="ew")
            row += 1
            ttk.Label(parent, text="Block size", style="Panel.TLabel").grid(row=row, column=0, sticky="w")
            ttk.Entry(parent, textvariable=self.block_size).grid(row=row, column=1, sticky="ew", pady=(0, 8))
            return row + 1

        def _section_output(self, parent: ttk.LabelFrame, row: int) -> int:
            ttk.Label(parent, text="Depth Analysis", style="Panel.TLabel").grid(row=row, column=0, sticky="w", pady=(2, 2))
            row += 1
            ttk.Label(parent, text="Output includes disparity, depth map, and analysis summary.", style="Panel.TLabel", wraplength=250).grid(
                row=row, column=0, columnspan=2, sticky="w", pady=(0, 8)
            )
            return row + 1

        def pick_left(self) -> None:
            path = filedialog.askopenfilename(title="Choose left image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
            if path:
                self.left_path.set(path)

        def pick_right(self) -> None:
            path = filedialog.askopenfilename(title="Choose right image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
            if path:
                self.right_path.set(path)

        def _image_for_display(self, img_bgr: cv2.typing.MatLike, size: tuple[int, int] = (260, 150)) -> ImageTk.PhotoImage:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            pil.thumbnail(size)
            return ImageTk.PhotoImage(pil)

        def run_pipeline_action(self) -> None:
            left = self.left_path.get().strip()
            right = self.right_path.get().strip()
            if not left or not right:
                messagebox.showwarning("Missing input", "Please select both left and right image files.")
                return

            config = PipelineConfig(
                filter_type=self.filter_type.get(),
                edge_method=self.edge_method.get(),
                rotation_deg=self.rotation.get(),
                translate_x=self.tx.get(),
                translate_y=self.ty.get(),
                num_disparities=self.num_disparities.get(),
                block_size=self.block_size.get(),
            )

            try:
                result = run_pipeline(left, right, config)
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Processing error", f"Could not process images:\n{exc}")
                self.status.set("Processing failed. Please verify file paths and parameter choices.")
                return

            self._current_result = result
            for key in ["left_image", "right_image", "filtered_output", "edge_output", "transformed_right", "disparity_map", "depth_map"]:
                img = self._image_for_display(result.images[key])
                self._image_refs[key] = img
                self.result_labels[key].configure(image=img, text="")

            self.result_labels["summary"].configure(text=result.summary, image="", anchor="w", justify="left")
            self.status.set("Pipeline completed. Review the comparison panels and save results if needed.")

        def save_results_action(self) -> None:
            if not self._current_result:
                messagebox.showinfo("No results", "Run the pipeline first to generate results.")
                return

            out_dir = filedialog.askdirectory(title="Save output images")
            if not out_dir:
                return

            path = Path(out_dir)
            path.mkdir(parents=True, exist_ok=True)
            for name, image in self._current_result.images.items():
                cv2.imwrite(str(path / f"{name}.png"), image)
            (path / "analysis_summary.txt").write_text(self._current_result.summary + "\n", encoding="utf-8")
            self.status.set(f"Saved outputs to {path}")

    app = CVMidtermGUI()
    app.mainloop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the computer vision GUI")
    parser.add_argument("--smoke-test", action="store_true", help="Run non-interactive GUI smoke test")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke_test:
        palette = get_palette()
        required = {"bg", "panel", "border", "text", "accent", "warning"}
        missing = required - set(palette)
        if missing:
            raise RuntimeError(f"Palette missing keys: {missing}")
        print("GUI smoke test passed")
        return 0

    return launch_gui()


if __name__ == "__main__":
    raise SystemExit(main())
