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

    # --- preprocess command ---
    prep_parser = subparsers.add_parser(
        "preprocess", help="Generate synthetic manga training data"
    )
    prep_parser.add_argument("input_dir", help="Directory of colored images")
    prep_parser.add_argument(
        "-o", "--output-dir", default="dataset", help="Output directory (default: dataset)"
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "colorize":
        import cv2
        import numpy as np
        from src.pipeline import MangaColorizerPipeline
        from src.postprocess import harmonize_colors

        print("Loading models...")
        pipeline = MangaColorizerPipeline()

        print(f"Colorizing {args.input} with reference {args.reference}...")
        result = pipeline.colorize_panel(
            args.input,
            args.reference,
            character_lora=args.lora,
            num_inference_steps=args.steps,
            seed=args.seed,
        )

        result_cv = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

        if args.harmonize:
            print(f"Harmonizing colors with {args.harmonize}...")
            ref = cv2.imread(args.harmonize)
            result_cv = harmonize_colors(result_cv, ref)

        cv2.imwrite(args.output, result_cv)
        print(f"Saved: {args.output}")

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
