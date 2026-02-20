"""Color consistency post-processing via LAB histogram matching."""

import cv2
import numpy as np
from skimage.exposure import match_histograms


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
