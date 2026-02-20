"""Post-processing utilities: denoising and color consistency."""

import cv2
import numpy as np
from skimage.exposure import match_histograms


def denoise_image(image: np.ndarray, strength: int = 25) -> np.ndarray:
    """Denoise a manga image using non-local means filtering.

    Useful for scanned manga with compression artifacts or paper grain.
    Not recommended for clean digital sources — will soften fine lines.

    Args:
        image: Grayscale or BGR uint8 image
        strength: Filter strength (higher = more denoising, default 25)

    Returns:
        Denoised uint8 image (same shape/channels as input)
    """
    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(image, None, h=strength)
    return cv2.fastNlMeansDenoisingColored(image, None, h=strength, hColor=strength)


def harmonize_colors(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    """Match color distribution of source to reference, preserving source lightness.

    Works in LAB color space:
    - L (lightness) channel is kept from the source image
    - a, b (color) channels are histogram-matched to the reference

    Args:
        source_img: BGR uint8 image (the generated/colorized panel)
        reference_img: BGR uint8 image (the reference or previous panel)

    Returns:
        BGR uint8 image with matched color distribution
    """
    source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB)
    reference_lab = cv2.cvtColor(reference_img, cv2.COLOR_BGR2LAB)

    # Histogram-match all channels, then restore source lightness
    matched_lab = match_histograms(
        source_lab, reference_lab, channel_axis=2
    ).astype(np.uint8)

    # Preserve the source's lightness
    matched_lab[:, :, 0] = source_lab[:, :, 0]

    result = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)
    return result
