# tests/test_main.py
from main import build_parser


def test_colorize_command_parses():
    parser = build_parser()
    args = parser.parse_args(["colorize", "panel.png", "ref.png"])
    assert args.command == "colorize"
    assert args.input == "panel.png"
    assert args.reference == "ref.png"
    assert args.output == "output.png"  # default
    assert args.denoise is False
    assert args.denoise_strength == 25


def test_colorize_with_options():
    parser = build_parser()
    args = parser.parse_args([
        "colorize", "panel.png", "ref.png",
        "-o", "result.png",
        "--lora", "style.safetensors",
        "--steps", "30",
        "--seed", "42",
        "--harmonize", "prev_panel.png",
        "--denoise",
        "--denoise-strength", "40",
    ])
    assert args.output == "result.png"
    assert args.lora == "style.safetensors"
    assert args.steps == 30
    assert args.seed == 42
    assert args.harmonize == "prev_panel.png"
    assert args.denoise is True
    assert args.denoise_strength == 40


def test_preprocess_command_parses():
    parser = build_parser()
    args = parser.parse_args(["preprocess", "/path/to/images"])
    assert args.command == "preprocess"
    assert args.input_dir == "/path/to/images"
    assert args.output_dir == "dataset"  # default


def test_preprocess_with_output():
    parser = build_parser()
    args = parser.parse_args(["preprocess", "/images", "-o", "/custom/output"])
    assert args.output_dir == "/custom/output"


def test_batch_command_parses():
    parser = build_parser()
    args = parser.parse_args(["batch", "/panels", "ref.png"])
    assert args.command == "batch"
    assert args.input_dir == "/panels"
    assert args.reference == "ref.png"
    assert args.output_dir == "output"  # default
    assert args.harmonize is False
    assert args.denoise is False


def test_batch_with_all_options():
    parser = build_parser()
    args = parser.parse_args([
        "batch", "/panels", "ref.png",
        "-o", "/colored",
        "--lora", "style.safetensors",
        "--steps", "30",
        "--seed", "42",
        "--harmonize",
        "--denoise",
        "--denoise-strength", "35",
    ])
    assert args.output_dir == "/colored"
    assert args.lora == "style.safetensors"
    assert args.steps == 30
    assert args.seed == 42
    assert args.harmonize is True
    assert args.denoise is True
    assert args.denoise_strength == 35
