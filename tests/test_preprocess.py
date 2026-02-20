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
