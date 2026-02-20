# tests/test_postprocess.py
import numpy as np
import cv2
from src.postprocess import harmonize_colors


def _make_solid_image(bgr_color, shape=(100, 100)):
    """Create a solid-color BGR image."""
    img = np.zeros((*shape, 3), dtype=np.uint8)
    img[:] = bgr_color
    return img


def test_harmonize_preserves_lightness():
    """The L channel of the source should be preserved."""
    source = _make_solid_image((50, 100, 200))  # some BGR color
    reference = _make_solid_image((200, 50, 100))  # different color

    result = harmonize_colors(source, reference)

    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    result_lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)

    # L channel should be identical
    np.testing.assert_array_equal(result_lab[:, :, 0], source_lab[:, :, 0])


def test_harmonize_shifts_color_channels():
    """The a,b channels should shift toward the reference distribution."""
    # Blue-ish source
    source = _make_solid_image((200, 50, 50))
    # Red-ish reference
    reference = _make_solid_image((50, 50, 200))

    result = harmonize_colors(source, reference)

    # Result should differ from source (colors shifted)
    assert not np.array_equal(result, source)


def test_harmonize_same_image_is_stable():
    """Harmonizing an image with itself should return roughly the same image."""
    img = _make_solid_image((100, 150, 200))
    result = harmonize_colors(img, img)

    # Should be very close (allowing for float rounding)
    np.testing.assert_allclose(result.astype(float), img.astype(float), atol=2)


def test_harmonize_output_shape_matches_source():
    """Output shape should match source shape."""
    source = _make_solid_image((100, 100, 100), shape=(200, 300))
    reference = _make_solid_image((50, 50, 50), shape=(150, 250))

    result = harmonize_colors(source, reference)

    assert result.shape == source.shape


def test_harmonize_output_dtype_is_uint8():
    """Output should be uint8."""
    source = _make_solid_image((100, 100, 100))
    reference = _make_solid_image((50, 50, 50))

    result = harmonize_colors(source, reference)

    assert result.dtype == np.uint8
