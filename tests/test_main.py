# tests/test_main.py
from main import build_parser


def test_colorize_command_parses():
    parser = build_parser()
    args = parser.parse_args(["colorize", "panel.png", "ref.png"])
    assert args.command == "colorize"
    assert args.input == "panel.png"
    assert args.reference == "ref.png"
    assert args.output == "output.png"  # default


def test_colorize_with_options():
    parser = build_parser()
    args = parser.parse_args([
        "colorize", "panel.png", "ref.png",
        "-o", "result.png",
        "--lora", "style.safetensors",
        "--steps", "30",
        "--seed", "42",
        "--harmonize", "prev_panel.png",
    ])
    assert args.output == "result.png"
    assert args.lora == "style.safetensors"
    assert args.steps == 30
    assert args.seed == 42
    assert args.harmonize == "prev_panel.png"


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
