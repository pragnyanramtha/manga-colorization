#!/usr/bin/env python3
"""Manga Colorizer — single entry point with environment validation.

Usage:
    ./colorize image.png                     # colorize one panel
    ./colorize -p ./chapter/                 # batch colorize a directory
    ./colorize -t ./colored_images/          # generate training data
"""

import sys
import os
import shutil
from pathlib import Path

# ── Minimum Python version ──────────────────────────────────────────────────

REQUIRED_PYTHON = (3, 11)

if sys.version_info < REQUIRED_PYTHON:
    sys.exit(
        f"Error: Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+ required, "
        f"got {sys.version_info.major}.{sys.version_info.minor}"
    )

# ── Dependency checks ───────────────────────────────────────────────────────

REQUIRED_PACKAGES = {
    "torch": "torch",
    "diffusers": "diffusers",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "cv2": "opencv-python",
    "skimage": "scikit-image",
    "PIL": "Pillow",
    "numpy": "numpy",
    "safetensors": "safetensors",
}


def check_dependencies():
    """Check all required packages are installed. Exit with clear error if not."""
    missing = []
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        sys.exit(
            f"Error: Missing required packages:\n"
            f"  {', '.join(missing)}\n\n"
            f"Install with:\n"
            f"  pip install {' '.join(missing)}\n\n"
            f"Or install everything:\n"
            f"  pip install -e '.[dev]'"
        )


def check_gpu():
    """Check NVIDIA GPU is available. Warn (don't exit) if not."""
    import torch
    if not torch.cuda.is_available():
        print(
            "Warning: No NVIDIA GPU detected. Inference will be extremely slow on CPU.\n"
            "  Recommended: NVIDIA GPU with 8GB+ VRAM",
            file=sys.stderr,
        )
        return False

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name} ({vram_gb:.1f} GB VRAM)")

    if vram_gb < 7.5:
        print(
            f"Warning: {vram_gb:.1f} GB VRAM detected. 8GB+ recommended. "
            f"May run out of memory.",
            file=sys.stderr,
        )
    return True


def check_image_file(path_str):
    """Validate an image file exists and is a supported format."""
    path = Path(path_str)
    if not path.exists():
        sys.exit(f"Error: File not found: {path}")
    if not path.is_file():
        sys.exit(f"Error: Not a file: {path}")

    valid_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    if path.suffix.lower() not in valid_ext:
        sys.exit(
            f"Error: Unsupported image format '{path.suffix}'\n"
            f"  Supported: {', '.join(sorted(valid_ext))}"
        )
    return path


def check_directory(path_str):
    """Validate a directory exists and contains images."""
    path = Path(path_str)
    if not path.exists():
        sys.exit(f"Error: Directory not found: {path}")
    if not path.is_dir():
        sys.exit(f"Error: Not a directory: {path}")

    extensions = {".png", ".jpg", ".jpeg", ".webp"}
    images = [f for f in path.iterdir() if f.suffix.lower() in extensions]
    if not images:
        sys.exit(f"Error: No images found in {path}")

    return path, images


def check_lora_file(path_str):
    """Validate a LoRA file exists."""
    path = Path(path_str)
    if not path.exists():
        sys.exit(f"Error: LoRA file not found: {path}")
    if path.suffix.lower() != ".safetensors":
        sys.exit(f"Error: LoRA file must be .safetensors, got '{path.suffix}'")
    return path


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="colorize",
        description="Manga Colorizer V1 — AI-powered manga panel colorization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  ./colorize page.png -r ref.png              # colorize one panel\n"
            "  ./colorize -p ./chapter/ -r ref.png          # batch a directory\n"
            "  ./colorize -p ./ch/ -r ref.png --harmonize   # batch + color consistency\n"
            "  ./colorize -t ./colored_images/              # generate training data\n"
        ),
    )

    # Mode selection (mutually exclusive)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "image", nargs="?", default=None,
        help="Path to a B&W manga panel to colorize",
    )
    mode.add_argument(
        "-p", "--batch", metavar="DIR",
        help="Batch colorize all images in a directory",
    )
    mode.add_argument(
        "-t", "--train", metavar="DIR",
        help="Generate synthetic training data from colored images",
    )

    # Reference image (required for colorize/batch, not for train)
    parser.add_argument(
        "-r", "--reference",
        help="Path to colored reference image (character sheet, colored page)",
    )

    # Output
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output path (file for single, directory for batch/train)",
    )

    # Model options
    parser.add_argument("--lora", help="Path to LoRA .safetensors weights")
    parser.add_argument(
        "--steps", type=int, default=20,
        help="Inference steps (default: 20)",
    )
    parser.add_argument("--seed", type=int, help="Random seed")

    # Processing options
    parser.add_argument(
        "--harmonize", action="store_true",
        help="Color harmonization (batch: between panels, single: provide --harmonize-ref)",
    )
    parser.add_argument(
        "--harmonize-ref",
        help="Reference image for single-panel color harmonization",
    )
    parser.add_argument(
        "--denoise", action="store_true",
        help="Denoise input before colorization (for scanned manga)",
    )
    parser.add_argument(
        "--denoise-strength", type=int, default=25,
        help="Denoising strength (default: 25)",
    )

    # Utility
    parser.add_argument(
        "--check", action="store_true",
        help="Only check dependencies and GPU, don't run anything",
    )

    return parser


# ── Commands ─────────────────────────────────────────────────────────────────

def cmd_colorize_single(args):
    """Colorize a single B&W manga panel."""
    import cv2
    import numpy as np
    from src.pipeline import MangaColorizerPipeline
    from src.postprocess import harmonize_colors, denoise_image

    # Validate inputs
    if not args.reference:
        sys.exit("Error: --reference (-r) is required for colorization")

    input_path = check_image_file(args.image)
    ref_path = check_image_file(args.reference)
    output_path = args.output or input_path.stem + "_colored.png"

    if args.lora:
        check_lora_file(args.lora)

    check_gpu()

    print("Loading models...")
    pipeline = MangaColorizerPipeline()

    actual_input = str(input_path)
    if args.denoise:
        print(f"Denoising (strength={args.denoise_strength})...")
        import tempfile
        img = cv2.imread(actual_input, cv2.IMREAD_GRAYSCALE)
        denoised = denoise_image(img, strength=args.denoise_strength)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, denoised)
        actual_input = tmp.name

    print(f"Colorizing {input_path.name}...")
    result = pipeline.colorize_panel(
        actual_input,
        str(ref_path),
        character_lora=args.lora,
        num_inference_steps=args.steps,
        seed=args.seed,
    )

    result_cv = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    if args.harmonize_ref:
        ref_img = cv2.imread(args.harmonize_ref)
        if ref_img is None:
            sys.exit(f"Error: Cannot read harmonize reference: {args.harmonize_ref}")
        result_cv = harmonize_colors(result_cv, ref_img)

    cv2.imwrite(str(output_path), result_cv)
    print(f"Saved: {output_path}")


def cmd_batch(args):
    """Batch colorize a directory of B&W manga panels."""
    import cv2
    import numpy as np
    from src.pipeline import MangaColorizerPipeline
    from src.postprocess import harmonize_colors, denoise_image

    if not args.reference:
        sys.exit("Error: --reference (-r) is required for colorization")

    input_dir, panels = check_directory(args.batch)
    ref_path = check_image_file(args.reference)
    output_dir = Path(args.output or "output")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.lora:
        check_lora_file(args.lora)

    panels = sorted(panels)
    print(f"Found {len(panels)} panels in {input_dir}")

    check_gpu()

    print("Loading models...")
    pipeline = MangaColorizerPipeline()

    if args.denoise:
        print(f"  Denoising enabled (strength={args.denoise_strength})")
    if args.harmonize:
        print("  Color harmonization enabled (consecutive panels)")

    prev_result = None
    for i, panel_path in enumerate(panels, 1):
        out_path = output_dir / panel_path.name

        actual_input = str(panel_path)
        if args.denoise:
            import tempfile
            img = cv2.imread(actual_input, cv2.IMREAD_GRAYSCALE)
            denoised = denoise_image(img, strength=args.denoise_strength)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            cv2.imwrite(tmp.name, denoised)
            actual_input = tmp.name

        print(f"  [{i}/{len(panels)}] {panel_path.name}...")
        result = pipeline.colorize_panel(
            actual_input,
            str(ref_path),
            character_lora=args.lora,
            num_inference_steps=args.steps,
            seed=args.seed,
        )

        result_cv = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

        if args.harmonize and prev_result is not None:
            result_cv = harmonize_colors(result_cv, prev_result)

        cv2.imwrite(str(out_path), result_cv)

        if args.harmonize:
            prev_result = result_cv

    print(f"Done. {len(panels)} panels saved to {output_dir}/")


def cmd_train(args):
    """Generate synthetic training data from colored images."""
    from src.preprocess import process_directory

    input_dir, images = check_directory(args.train)
    output_dir = Path(args.output or "dataset")

    bw_dir = str(output_dir / "train_bw")
    color_dir = str(output_dir / "train_color")

    print(f"Processing {len(images)} images from {input_dir}...")
    count = process_directory(str(input_dir), bw_dir, color_dir)
    print(f"Done. Processed {count} images.")
    print(f"  B&W:   {bw_dir}")
    print(f"  Color: {color_dir}")
    print()
    print("Next steps:")
    print("  1. Upload the dataset/ folder to Kaggle")
    print("  2. Open notebooks/kaggle_train.ipynb and run all cells")
    print("  3. Download the .safetensors LoRA file")
    print("  4. Use it: ./colorize page.png -r ref.png --lora your_style.safetensors")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Dep check always runs
    check_dependencies()

    if args.check:
        check_gpu()
        print("All dependencies OK.")
        return

    # Dispatch
    if args.train:
        cmd_train(args)
    elif args.batch:
        cmd_batch(args)
    elif args.image:
        cmd_colorize_single(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
