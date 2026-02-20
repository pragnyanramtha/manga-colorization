"""CLI entry point for Manga Colorizer V1."""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manga Colorizer V1 — AI-powered manga panel colorization"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- colorize command ---
    color_parser = subparsers.add_parser(
        "colorize", help="Colorize a B&W manga panel"
    )
    color_parser.add_argument("input", help="Path to B&W manga panel")
    color_parser.add_argument("reference", help="Path to colored reference image")
    color_parser.add_argument(
        "-o", "--output", default="output.png", help="Output path (default: output.png)"
    )
    color_parser.add_argument("--lora", help="Path to LoRA .safetensors weights")
    color_parser.add_argument(
        "--steps", type=int, default=20, help="Inference steps (default: 20)"
    )
    color_parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    color_parser.add_argument(
        "--harmonize",
        help="Path to reference image for color harmonization (e.g., previous panel)",
    )
    color_parser.add_argument(
        "--denoise", action="store_true",
        help="Denoise input before colorization (for scanned manga)",
    )
    color_parser.add_argument(
        "--denoise-strength", type=int, default=25,
        help="Denoising filter strength (default: 25, higher = more smoothing)",
    )

    # --- batch command ---
    batch_parser = subparsers.add_parser(
        "batch", help="Colorize a directory of B&W manga panels"
    )
    batch_parser.add_argument("input_dir", help="Directory of B&W manga panels")
    batch_parser.add_argument("reference", help="Path to colored reference image")
    batch_parser.add_argument(
        "-o", "--output-dir", default="output", help="Output directory (default: output)"
    )
    batch_parser.add_argument("--lora", help="Path to LoRA .safetensors weights")
    batch_parser.add_argument(
        "--steps", type=int, default=20, help="Inference steps (default: 20)"
    )
    batch_parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    batch_parser.add_argument(
        "--harmonize", action="store_true",
        help="Apply color harmonization between consecutive panels",
    )
    batch_parser.add_argument(
        "--denoise", action="store_true",
        help="Denoise inputs before colorization (for scanned manga)",
    )
    batch_parser.add_argument(
        "--denoise-strength", type=int, default=25,
        help="Denoising filter strength (default: 25)",
    )

    # --- preprocess command ---
    prep_parser = subparsers.add_parser(
        "preprocess", help="Generate synthetic manga training data"
    )
    prep_parser.add_argument("input_dir", help="Directory of colored images")
    prep_parser.add_argument(
        "-o", "--output-dir", default="dataset", help="Output directory (default: dataset)"
    )

    return parser


def _colorize_single(pipeline, input_path, reference_path, output_path, args,
                     harmonize_ref=None):
    """Colorize a single panel with optional denoise and harmonization."""
    import cv2
    import numpy as np
    from src.postprocess import harmonize_colors, denoise_image

    actual_input = str(input_path)

    if args.denoise:
        import tempfile
        img = cv2.imread(actual_input, cv2.IMREAD_GRAYSCALE)
        denoised = denoise_image(img, strength=args.denoise_strength)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, denoised)
        actual_input = tmp.name

    result = pipeline.colorize_panel(
        actual_input,
        str(reference_path),
        character_lora=args.lora,
        num_inference_steps=args.steps,
        seed=args.seed,
    )

    result_cv = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    if harmonize_ref is not None:
        result_cv = harmonize_colors(result_cv, harmonize_ref)

    cv2.imwrite(str(output_path), result_cv)
    return result_cv


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "colorize":
        from src.pipeline import MangaColorizerPipeline

        print("Loading models...")
        pipeline = MangaColorizerPipeline()

        harmonize_ref = None
        if args.harmonize:
            import cv2
            harmonize_ref = cv2.imread(args.harmonize)

        print(f"Colorizing {args.input} with reference {args.reference}...")
        if args.denoise:
            print(f"  Denoising enabled (strength={args.denoise_strength})")

        _colorize_single(
            pipeline, args.input, args.reference, args.output, args,
            harmonize_ref=harmonize_ref,
        )
        print(f"Saved: {args.output}")

    elif args.command == "batch":
        from src.pipeline import MangaColorizerPipeline

        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        extensions = {".png", ".jpg", ".jpeg", ".webp"}
        panels = sorted(
            f for f in input_dir.iterdir() if f.suffix.lower() in extensions
        )

        if not panels:
            print(f"No images found in {input_dir}")
            return

        print(f"Found {len(panels)} panels in {input_dir}")
        print("Loading models...")
        pipeline = MangaColorizerPipeline()

        if args.denoise:
            print(f"  Denoising enabled (strength={args.denoise_strength})")
        if args.harmonize:
            print("  Color harmonization enabled (consecutive panels)")

        prev_result = None
        for i, panel_path in enumerate(panels, 1):
            out_path = output_dir / panel_path.name

            harmonize_ref = prev_result if (args.harmonize and prev_result is not None) else None

            print(f"  [{i}/{len(panels)}] {panel_path.name}...")
            result_cv = _colorize_single(
                pipeline, panel_path, args.reference, out_path, args,
                harmonize_ref=harmonize_ref,
            )

            if args.harmonize:
                prev_result = result_cv

        print(f"Done. {len(panels)} panels saved to {output_dir}/")

    elif args.command == "preprocess":
        from src.preprocess import process_directory

        bw_dir = str(Path(args.output_dir) / "train_bw")
        color_dir = str(Path(args.output_dir) / "train_color")

        print(f"Processing images from {args.input_dir}...")
        count = process_directory(args.input_dir, bw_dir, color_dir)
        print(f"Done. Processed {count} images.")
        print(f"  B&W:   {bw_dir}")
        print(f"  Color: {color_dir}")


if __name__ == "__main__":
    main()
