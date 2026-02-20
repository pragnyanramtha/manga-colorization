# tests/test_main.py
from colorize import build_parser, check_dependencies


def test_single_image_parses():
    parser = build_parser()
    args = parser.parse_args(["page.png", "-r", "ref.png"])
    assert args.image == "page.png"
    assert args.reference == "ref.png"
    assert args.output is None
    assert args.denoise is False
    assert args.denoise_strength == 25


def test_single_image_with_options():
    parser = build_parser()
    args = parser.parse_args([
        "page.png", "-r", "ref.png",
        "-o", "result.png",
        "--lora", "style.safetensors",
        "--steps", "30",
        "--seed", "42",
        "--harmonize-ref", "prev.png",
        "--denoise",
        "--denoise-strength", "40",
    ])
    assert args.image == "page.png"
    assert args.output == "result.png"
    assert args.lora == "style.safetensors"
    assert args.steps == 30
    assert args.seed == 42
    assert args.harmonize_ref == "prev.png"
    assert args.denoise is True
    assert args.denoise_strength == 40


def test_batch_parses():
    parser = build_parser()
    args = parser.parse_args(["-p", "/panels", "-r", "ref.png"])
    assert args.batch == "/panels"
    assert args.image is None
    assert args.reference == "ref.png"


def test_batch_with_options():
    parser = build_parser()
    args = parser.parse_args([
        "-p", "/panels", "-r", "ref.png",
        "-o", "/colored",
        "--harmonize",
        "--denoise",
        "--denoise-strength", "35",
    ])
    assert args.batch == "/panels"
    assert args.output == "/colored"
    assert args.harmonize is True
    assert args.denoise is True
    assert args.denoise_strength == 35


def test_train_parses():
    parser = build_parser()
    args = parser.parse_args(["-t", "/colored_images"])
    assert args.train == "/colored_images"
    assert args.image is None
    assert args.batch is None


def test_train_with_output():
    parser = build_parser()
    args = parser.parse_args(["-t", "/images", "-o", "/custom_dataset"])
    assert args.train == "/images"
    assert args.output == "/custom_dataset"


def test_check_flag():
    parser = build_parser()
    args = parser.parse_args(["--check"])
    assert args.check is True


def test_defaults():
    parser = build_parser()
    args = parser.parse_args(["img.png", "-r", "ref.png"])
    assert args.steps == 20
    assert args.seed is None
    assert args.lora is None
    assert args.harmonize is False
    assert args.denoise is False
    assert args.denoise_strength == 25
    assert args.check is False


def test_check_dependencies_passes():
    """Dep check should pass in our test environment."""
    check_dependencies()  # should not raise
