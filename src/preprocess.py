"""Synthetic manga data generation from colored anime images."""

import cv2
import numpy as np
from pathlib import Path


def extract_lines(image: np.ndarray, method: str = "canny") -> np.ndarray:
    """Extract line art from a colored image.

    Args:
        image: BGR uint8 image
        method: Edge detection method ("canny")

    Returns:
        Grayscale image with dark lines on white background (manga convention)
    """
    if method != "canny":
        raise ValueError(f"Unknown method: {method}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Invert: Canny gives white-on-black, manga needs dark-on-white
    return 255 - edges
