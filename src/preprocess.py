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


def add_paper_texture(image: np.ndarray, intensity: float = 0.03) -> np.ndarray:
    """Add subtle paper-like texture noise to simulate scanned manga.

    Args:
        image: Grayscale uint8 image
        intensity: Noise strength (0.0 to 1.0, default 0.03)

    Returns:
        Grayscale uint8 image with paper texture
    """
    noise = np.random.normal(0, intensity * 255, image.shape).astype(np.float32)
    result = np.clip(image.astype(np.float32) + noise, 0, 255)
    return result.astype(np.uint8)


def create_synthetic_manga(image: np.ndarray) -> np.ndarray:
    """Convert a colored image into a synthetic manga page.

    Combines line extraction, screentone simulation, and paper texture
    to produce a realistic B&W manga appearance.

    Args:
        image: BGR uint8 colored image

    Returns:
        Grayscale uint8 synthetic manga image
    """
    lines = extract_lines(image)
    tones = simulate_screentones(image)

    # Combine: take the darker value at each pixel (lines dominate, tones fill)
    combined = np.minimum(lines, tones)
    combined = add_paper_texture(combined)

    return combined


def process_directory(
    input_dir: str, output_bw_dir: str, output_color_dir: str
) -> int:
    """Process all images in a directory to create paired training data.

    Args:
        input_dir: Directory containing colored anime/manga images
        output_bw_dir: Output directory for synthetic B&W manga images
        output_color_dir: Output directory for original colored images

    Returns:
        Number of images processed
    """
    input_path = Path(input_dir)
    bw_path = Path(output_bw_dir)
    color_path = Path(output_color_dir)

    bw_path.mkdir(parents=True, exist_ok=True)
    color_path.mkdir(parents=True, exist_ok=True)

    extensions = {".png", ".jpg", ".jpeg", ".webp"}
    count = 0

    for img_file in sorted(input_path.iterdir()):
        if img_file.suffix.lower() not in extensions:
            continue

        image = cv2.imread(str(img_file))
        if image is None:
            continue

        bw = create_synthetic_manga(image)

        cv2.imwrite(str(bw_path / img_file.name), bw)
        cv2.imwrite(str(color_path / img_file.name), image)
        count += 1

    return count
