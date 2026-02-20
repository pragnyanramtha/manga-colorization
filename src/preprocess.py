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


def simulate_screentones(image: np.ndarray, dot_size: int = 4) -> np.ndarray:
    """Convert an image into halftone-like screentone patterns.

    Simulates the dot-pattern printing technique used in manga.
    Darker areas get larger/denser dots; lighter areas get smaller/sparser dots.

    Args:
        image: BGR uint8 image
        dot_size: Base radius for halftone dots (pixels)

    Returns:
        Grayscale image with screentone pattern (dark dots on white)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # White canvas
    halftone = np.ones_like(gray) * 255
    step = dot_size * 2

    for y in range(0, h, step):
        for x in range(0, w, step):
            # Sample the average intensity of this cell
            cell = gray[y : y + step, x : x + step]
            if cell.size == 0:
                continue

            avg_intensity = np.mean(cell)

            # Darker pixel -> larger dot (invert: 0=black -> big dot)
            radius = int((1.0 - avg_intensity / 255.0) * dot_size)
            if radius > 0:
                cy = min(y + dot_size, h - 1)
                cx = min(x + dot_size, w - 1)
                cv2.circle(halftone, (cx, cy), radius, 0, -1)

    return halftone
