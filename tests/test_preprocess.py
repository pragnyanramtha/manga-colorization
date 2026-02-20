# tests/test_preprocess.py
import numpy as np
import cv2
from src.preprocess import extract_lines


def _make_test_image_with_edges(size=200):
    """Create a BGR image with a clear black rectangle on white background."""
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), 3)
    return img


def test_extract_lines_output_is_grayscale():
    img = _make_test_image_with_edges()
    lines = extract_lines(img)
    assert lines.ndim == 2  # single channel


def test_extract_lines_output_shape_matches_input():
    img = _make_test_image_with_edges()
    lines = extract_lines(img)
    assert lines.shape == img.shape[:2]


def test_extract_lines_manga_convention():
    """Output should follow manga convention: dark lines on white background."""
    img = _make_test_image_with_edges()
    lines = extract_lines(img)

    # The majority of the image should be white (background)
    white_ratio = np.sum(lines > 200) / lines.size
    assert white_ratio > 0.5, f"Expected mostly white background, got {white_ratio:.2f}"


def test_extract_lines_detects_edges():
    """Lines should appear where the rectangle edges are."""
    img = _make_test_image_with_edges()
    lines = extract_lines(img)

    # The rectangle border region should have dark pixels
    border_region = lines[48:53, 50:150]  # top edge of rectangle
    has_dark_pixels = np.any(border_region < 100)
    assert has_dark_pixels, "Expected dark pixels at edge locations"


def test_extract_lines_invalid_method_raises():
    img = _make_test_image_with_edges()
    try:
        extract_lines(img, method="invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


from src.preprocess import simulate_screentones


def _make_gradient_image(size=200):
    """Create a BGR image with a horizontal grayscale gradient."""
    gray = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_screentones_output_is_grayscale():
    img = _make_gradient_image()
    tones = simulate_screentones(img)
    assert tones.ndim == 2


def test_screentones_output_shape_matches():
    img = _make_gradient_image(150)
    tones = simulate_screentones(img)
    assert tones.shape == img.shape[:2]


def test_screentones_not_simple_desaturation():
    """Screentones should produce a pattern, not a flat grayscale conversion."""
    img = _make_gradient_image()
    tones = simulate_screentones(img)
    simple_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Screentones should differ significantly from simple grayscale
    diff = np.abs(tones.astype(float) - simple_gray.astype(float))
    assert np.mean(diff) > 10, "Screentones should differ from simple desaturation"


def test_screentones_has_dot_pattern():
    """Dark areas should produce visible dots, light areas should be mostly white."""
    # Dark image
    dark = np.zeros((100, 100, 3), dtype=np.uint8) + 30
    tones_dark = simulate_screentones(dark)

    # Light image
    light = np.zeros((100, 100, 3), dtype=np.uint8) + 230
    tones_light = simulate_screentones(light)

    # Dark areas should have more dark pixels (more/larger dots)
    dark_pixel_ratio_dark = np.sum(tones_dark < 128) / tones_dark.size
    dark_pixel_ratio_light = np.sum(tones_light < 128) / tones_light.size

    assert dark_pixel_ratio_dark > dark_pixel_ratio_light
