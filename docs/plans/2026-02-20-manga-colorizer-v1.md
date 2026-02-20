# Manga Colorizer V1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a local manga colorization pipeline that converts B&W manga panels into colored images using SD1.5 + ControlNet + IP-Adapter, runnable on 8GB VRAM.

**Architecture:** Reference-based style transfer with no text prompts. A B&W manga panel is structurally guided by ControlNet (lineart), while colors come from an IP-Adapter reference image. Post-processing ensures color consistency between panels via LAB histogram matching.

**Tech Stack:** Python 3.11, HuggingFace Diffusers, PyTorch, OpenCV, scikit-image, Pillow

---

## Critical API Notes (From Research)

These override/clarify the original PRD:

1. **Counterfeit-V3.0 is a single-file checkpoint** — must use `from_single_file()`, not `from_pretrained()`
2. **IP-Adapter Plus requires explicit ViT-H image encoder** — must load `CLIPVisionModelWithProjection` from `h94/IP-Adapter` subfolder `models/image_encoder` and inject it into the pipeline at construction time
3. **The s2 ControlNet variant recommends `num_hidden_layers=11`** on the CLIP text encoder (skip last layer)
4. **`enable_model_cpu_offload()` MUST be called AFTER `load_ip_adapter()`** — reversed order causes errors
5. **Weight name:** `ip-adapter-plus_sd15.safetensors` (not `.bin`)

---

## Task 1: Project Setup

**Files:**
- Modify: `pyproject.toml`
- Create: `requirements.txt`
- Modify: `.gitignore`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Update pyproject.toml with dependencies**

```toml
[project]
name = "manga-colorization"
version = "0.1.0"
description = "AI pipeline for colorizing B&W manga panels using reference-based style transfer"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.1.0",
    "diffusers>=0.28.0",
    "transformers>=4.38.0",
    "accelerate>=0.27.0",
    "safetensors>=0.4.0",
    "opencv-python>=4.9.0",
    "scikit-image>=0.22.0",
    "Pillow>=10.0.0",
    "numpy>=1.26.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
]

[project.scripts]
manga-colorize = "main:main"
```

**Step 2: Create requirements.txt (for Kaggle/pip compatibility)**

```
torch>=2.1.0
diffusers>=0.28.0
transformers>=4.38.0
accelerate>=0.27.0
safetensors>=0.4.0
opencv-python>=4.9.0
scikit-image>=0.22.0
Pillow>=10.0.0
numpy>=1.26.0
```

**Step 3: Update .gitignore**

Append to existing `.gitignore`:

```
# Models (large files)
models/

# Dataset outputs
dataset/

# Generated outputs
output/
*.png
*.jpg
*.jpeg

# Jupyter
.ipynb_checkpoints/
```

**Step 4: Create directory structure**

```bash
mkdir -p src tests notebooks models
touch src/__init__.py tests/__init__.py
```

**Step 5: Commit**

```bash
git add pyproject.toml requirements.txt .gitignore src/__init__.py tests/__init__.py
git commit -m "feat: project setup with dependencies and directory structure"
```

---

## Task 2: Post-Processing Module (TDD)

**Files:**
- Create: `tests/test_postprocess.py`
- Create: `src/postprocess.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_postprocess.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.postprocess'`

**Step 3: Write minimal implementation**

```python
# src/postprocess.py
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_postprocess.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add tests/test_postprocess.py src/postprocess.py
git commit -m "feat: add LAB histogram matching post-processing"
```

---

## Task 3: Preprocessing Module — Line Extraction (TDD)

**Files:**
- Create: `tests/test_preprocess.py`
- Create: `src/preprocess.py`

**Step 1: Write the failing test for line extraction**

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_preprocess.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation for line extraction**

```python
# src/preprocess.py
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_preprocess.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add tests/test_preprocess.py src/preprocess.py
git commit -m "feat: add line extraction for synthetic manga generation"
```

---

## Task 4: Preprocessing Module — Screentone Simulation (TDD)

**Files:**
- Modify: `tests/test_preprocess.py`
- Modify: `src/preprocess.py`

**Step 1: Write failing tests for screentone simulation**

Append to `tests/test_preprocess.py`:

```python
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
```

**Step 2: Run tests to verify new tests fail**

Run: `pytest tests/test_preprocess.py -v -k screentone`
Expected: FAIL with `ImportError`

**Step 3: Add screentone implementation**

Add to `src/preprocess.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_preprocess.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add tests/test_preprocess.py src/preprocess.py
git commit -m "feat: add screentone simulation for synthetic manga"
```

---

## Task 5: Preprocessing Module — Full Pipeline & Directory Processing (TDD)

**Files:**
- Modify: `tests/test_preprocess.py`
- Modify: `src/preprocess.py`

**Step 1: Write failing tests for paper texture and full pipeline**

Append to `tests/test_preprocess.py`:

```python
from src.preprocess import add_paper_texture, create_synthetic_manga, process_directory
import tempfile
import os


def test_paper_texture_adds_noise():
    img = np.ones((100, 100), dtype=np.uint8) * 128
    textured = add_paper_texture(img, intensity=0.1)

    # Should not be identical (noise added)
    assert not np.array_equal(img, textured)

    # But should be close (subtle noise)
    np.testing.assert_allclose(
        textured.astype(float), img.astype(float), atol=50
    )


def test_paper_texture_output_is_uint8():
    img = np.ones((100, 100), dtype=np.uint8) * 128
    textured = add_paper_texture(img)
    assert textured.dtype == np.uint8


def test_create_synthetic_manga_full_pipeline():
    img = _make_test_image_with_edges()
    result = create_synthetic_manga(img)

    assert result.ndim == 2
    assert result.shape == img.shape[:2]
    assert result.dtype == np.uint8


def test_process_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        bw_dir = os.path.join(tmpdir, "bw")
        color_dir = os.path.join(tmpdir, "color")
        os.makedirs(input_dir)

        # Create two test images
        for name in ["test1.png", "test2.png"]:
            img = _make_test_image_with_edges()
            cv2.imwrite(os.path.join(input_dir, name), img)

        # Also create a non-image file that should be skipped
        with open(os.path.join(input_dir, "readme.txt"), "w") as f:
            f.write("not an image")

        count = process_directory(input_dir, bw_dir, color_dir)

        assert count == 2
        assert os.path.exists(os.path.join(bw_dir, "test1.png"))
        assert os.path.exists(os.path.join(bw_dir, "test2.png"))
        assert os.path.exists(os.path.join(color_dir, "test1.png"))
        assert os.path.exists(os.path.join(color_dir, "test2.png"))
        assert not os.path.exists(os.path.join(bw_dir, "readme.txt"))


def test_process_directory_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        os.makedirs(input_dir)
        count = process_directory(
            input_dir,
            os.path.join(tmpdir, "bw"),
            os.path.join(tmpdir, "color"),
        )
        assert count == 0
```

**Step 2: Run tests to verify new tests fail**

Run: `pytest tests/test_preprocess.py -v -k "paper or synthetic or directory"`
Expected: FAIL with `ImportError`

**Step 3: Add remaining preprocessing functions**

Add to `src/preprocess.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_preprocess.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add tests/test_preprocess.py src/preprocess.py
git commit -m "feat: add full preprocessing pipeline with paper texture and directory processing"
```

---

## Task 6: Inference Pipeline — Core Class

**Files:**
- Create: `src/pipeline.py`
- Create: `tests/test_pipeline.py`

This module depends on large model downloads for full integration testing. We test what we can (config, resizing, argument validation) and provide a manual smoke test for the full pipeline.

**Step 1: Write the testable parts**

```python
# tests/test_pipeline.py
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np

from src.pipeline import MangaColorizerPipeline


def test_resize_for_sd_portrait():
    """Portrait images should resize to 512x768."""
    pipe = MangaColorizerPipeline.__new__(MangaColorizerPipeline)
    img = Image.new("RGB", (400, 600))  # portrait
    resized = pipe._resize_for_sd(img)
    assert resized.size == (512, 768)


def test_resize_for_sd_landscape():
    """Landscape images should resize to 768x512."""
    pipe = MangaColorizerPipeline.__new__(MangaColorizerPipeline)
    img = Image.new("RGB", (600, 400))  # landscape
    resized = pipe._resize_for_sd(img)
    assert resized.size == (768, 512)


def test_resize_for_sd_square():
    """Square images should resize to 512x768 (portrait default)."""
    pipe = MangaColorizerPipeline.__new__(MangaColorizerPipeline)
    img = Image.new("RGB", (500, 500))  # square
    resized = pipe._resize_for_sd(img)
    assert resized.size == (512, 768)


def test_negative_prompt_constant():
    """Negative prompt should be hardcoded."""
    assert "monochrome" in MangaColorizerPipeline.NEGATIVE_PROMPT
    assert "greyscale" in MangaColorizerPipeline.NEGATIVE_PROMPT


def test_default_model_ids():
    """Default model IDs should be set."""
    assert MangaColorizerPipeline.DEFAULT_CONTROLNET_ID is not None
    assert MangaColorizerPipeline.DEFAULT_IP_ADAPTER_REPO is not None
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the pipeline implementation**

```python
# src/pipeline.py
"""Core manga colorization pipeline using SD1.5 + ControlNet + IP-Adapter."""

import torch
from pathlib import Path
from PIL import Image

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from transformers import CLIPTextModel, CLIPVisionModelWithProjection


class MangaColorizerPipeline:
    """Reference-based manga colorization using Stable Diffusion 1.5.

    Uses ControlNet for structural guidance (lineart) and IP-Adapter
    for color/style transfer from a reference image.
    """

    NEGATIVE_PROMPT = "monochrome, greyscale, lowres, bad anatomy"

    # Default model identifiers
    DEFAULT_CHECKPOINT_URL = "https://huggingface.co/gsdf/Counterfeit-V3.0/resolve/main/Counterfeit-V3.0_fix_fp16.safetensors"
    DEFAULT_CONTROLNET_ID = "lllyasviel/control_v11p_sd15s2_lineart_anime"
    DEFAULT_IP_ADAPTER_REPO = "h94/IP-Adapter"
    DEFAULT_IP_ADAPTER_WEIGHT = "ip-adapter-plus_sd15.safetensors"
    DEFAULT_SD15_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    def __init__(
        self,
        checkpoint_url: str | None = None,
        controlnet_id: str | None = None,
        ip_adapter_repo: str | None = None,
    ):
        checkpoint_url = checkpoint_url or self.DEFAULT_CHECKPOINT_URL
        controlnet_id = controlnet_id or self.DEFAULT_CONTROLNET_ID
        ip_adapter_repo = ip_adapter_repo or self.DEFAULT_IP_ADAPTER_REPO

        # Load ViT-H image encoder (required for IP-Adapter Plus)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            ip_adapter_repo,
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
        )

        # Load ControlNet
        controlnet = ControlNetModel.from_pretrained(
            controlnet_id,
            torch_dtype=torch.float16,
        )

        # Load text encoder with skip-last-layer (recommended for s2 ControlNet)
        text_encoder = CLIPTextModel.from_pretrained(
            self.DEFAULT_SD15_REPO,
            subfolder="text_encoder",
            num_hidden_layers=11,
            torch_dtype=torch.float16,
        )

        # Build pipeline from single-file checkpoint
        self.pipe = StableDiffusionControlNetPipeline.from_single_file(
            checkpoint_url,
            controlnet=controlnet,
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            torch_dtype=torch.float16,
            safety_checker=None,
        )

        # Load IP-Adapter (MUST happen before enable_model_cpu_offload)
        self.pipe.load_ip_adapter(
            ip_adapter_repo,
            subfolder="models",
            weight_name=self.DEFAULT_IP_ADAPTER_WEIGHT,
        )
        self.pipe.set_ip_adapter_scale(0.6)

        # Scheduler
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        # Memory optimization (MUST be after load_ip_adapter)
        self.pipe.enable_model_cpu_offload()

    def _resize_for_sd(self, image: Image.Image) -> Image.Image:
        """Resize image to SD1.5 native resolution.

        Portrait/square -> 512x768, landscape -> 768x512.
        """
        w, h = image.size
        if w > h:
            target = (768, 512)
        else:
            target = (512, 768)
        return image.resize(target, Image.LANCZOS)

    def colorize_panel(
        self,
        bw_image_path: str,
        reference_image_path: str,
        character_lora: str | None = None,
        num_inference_steps: int = 20,
        controlnet_scale: float = 1.1,
        ip_adapter_scale: float = 0.6,
        seed: int | None = None,
    ) -> Image.Image:
        """Colorize a B&W manga panel using a colored reference image.

        Args:
            bw_image_path: Path to the B&W manga panel
            reference_image_path: Path to the colored reference/character sheet
            character_lora: Optional path to a LoRA .safetensors file
            num_inference_steps: Number of diffusion steps (default 20)
            controlnet_scale: ControlNet conditioning strength (default 1.1)
            ip_adapter_scale: IP-Adapter influence strength (default 0.6)
            seed: Random seed for reproducibility

        Returns:
            PIL Image of the colorized panel
        """
        bw_image = Image.open(bw_image_path).convert("RGB")
        reference_image = Image.open(reference_image_path).convert("RGB")

        bw_image = self._resize_for_sd(bw_image)
        reference_image = self._resize_for_sd(reference_image)

        # Update IP-Adapter scale if changed
        self.pipe.set_ip_adapter_scale(ip_adapter_scale)

        # Load optional LoRA
        if character_lora:
            self.pipe.load_lora_weights(character_lora)

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        result = self.pipe(
            prompt="high quality, anime, detailed, vibrant colors",
            negative_prompt=self.NEGATIVE_PROMPT,
            image=bw_image,
            ip_adapter_image=reference_image,
            controlnet_conditioning_scale=controlnet_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        # Unload LoRA if loaded
        if character_lora:
            self.pipe.unload_lora_weights()

        return result.images[0]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat: add core inference pipeline with SD1.5 + ControlNet + IP-Adapter"
```

---

## Task 7: CLI Entry Point

**Files:**
- Modify: `main.py`
- Create: `tests/test_main.py`

**Step 1: Write failing tests for CLI argument parsing**

```python
# tests/test_main.py
import sys
from unittest.mock import patch, MagicMock
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_main.py -v`
Expected: FAIL with `ImportError`

**Step 3: Implement main.py**

```python
# main.py
"""CLI entry point for Manga Colorizer V1."""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manga Colorizer V1 — AI-powered manga panel colorization"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- colorize command ---
    color_parser = subparsers.add_parser(
        "colorize", help="Colorize a B&W manga panel"
    )
    color_parser.add_argument("input", help="Path to B&W manga panel")
    color_parser.add_argument("reference", help="Path to colored reference image")
    color_parser.add_argument(
        "-o", "--output", default="output.png", help="Output path (default: output.png)"
    )
    color_parser.add_argument("--lora", help="Path to LoRA .safetensors weights")
    color_parser.add_argument(
        "--steps", type=int, default=20, help="Inference steps (default: 20)"
    )
    color_parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    color_parser.add_argument(
        "--harmonize",
        help="Path to reference image for color harmonization (e.g., previous panel)",
    )

    # --- preprocess command ---
    prep_parser = subparsers.add_parser(
        "preprocess", help="Generate synthetic manga training data"
    )
    prep_parser.add_argument("input_dir", help="Directory of colored images")
    prep_parser.add_argument(
        "-o", "--output-dir", default="dataset", help="Output directory (default: dataset)"
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "colorize":
        import cv2
        import numpy as np
        from src.pipeline import MangaColorizerPipeline
        from src.postprocess import harmonize_colors

        print("Loading models...")
        pipeline = MangaColorizerPipeline()

        print(f"Colorizing {args.input} with reference {args.reference}...")
        result = pipeline.colorize_panel(
            args.input,
            args.reference,
            character_lora=args.lora,
            num_inference_steps=args.steps,
            seed=args.seed,
        )

        result_cv = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

        if args.harmonize:
            print(f"Harmonizing colors with {args.harmonize}...")
            ref = cv2.imread(args.harmonize)
            result_cv = harmonize_colors(result_cv, ref)

        cv2.imwrite(args.output, result_cv)
        print(f"Saved: {args.output}")

    elif args.command == "preprocess":
        from src.preprocess import process_directory

        bw_dir = str(Path(args.output_dir) / "train_bw")
        color_dir = str(Path(args.output_dir) / "train_color")

        print(f"Processing images from {args.input_dir}...")
        count = process_directory(args.input_dir, bw_dir, color_dir)
        print(f"Done. Processed {count} images.")
        print(f"  B&W:   {bw_dir}")
        print(f"  Color: {color_dir}")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_main.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: add CLI entry point with colorize and preprocess commands"
```

---

## Task 8: Kaggle Training Notebook

**Files:**
- Create: `notebooks/kaggle_train.ipynb`

**Step 1: Create the Kaggle LoRA training notebook**

Create a Jupyter notebook with the following cells:

**Cell 1 (Markdown):**
```markdown
# Manga Colorizer — LoRA Training on Kaggle (2x T4)

This notebook trains a style-specific LoRA on Kaggle using `kohya-ss/sd-scripts`.
Upload your preprocessed dataset (from `python main.py preprocess`) before running.
```

**Cell 2 (Code) — Install dependencies:**
```python
# Install kohya_ss training scripts and dependencies
!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
!pip install -q xformers
!git clone https://github.com/kohya-ss/sd-scripts.git /kaggle/working/sd-scripts
!pip install -q -r /kaggle/working/sd-scripts/requirements.txt
!pip install -q accelerate safetensors
```

**Cell 3 (Code) — Configuration:**
```python
# === CONFIGURATION ===
# Modify these paths for your Kaggle dataset

BASE_MODEL = "gsdf/Counterfeit-V3.0"
DATASET_DIR = "/kaggle/input/your-dataset"  # Upload your dataset here
OUTPUT_DIR = "/kaggle/working/lora_output"
LORA_NAME = "manga_style_lora"

# Training hyperparameters (tuned for 2x T4, 200 images)
RESOLUTION = 512
BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 1e-4
NETWORK_DIM = 32  # LoRA rank
NETWORK_ALPHA = 16
```

**Cell 4 (Code) — Prepare dataset metadata:**
```python
import json
import os
from pathlib import Path

# Create dataset metadata for kohya training
dataset_config = {
    "general": {
        "resolution": RESOLUTION,
        "shuffle_caption": False,
        "keep_tokens": 0,
    },
    "datasets": [
        {
            "subsets": [
                {
                    "image_dir": str(Path(DATASET_DIR) / "train_color"),
                    "conditioning_data_dir": str(Path(DATASET_DIR) / "train_bw"),
                    "caption_extension": ".txt",
                    "num_repeats": 5,
                }
            ]
        }
    ],
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
config_path = f"{OUTPUT_DIR}/dataset_config.json"
with open(config_path, "w") as f:
    json.dump(dataset_config, f, indent=2)

# Create blank caption files (no text prompts needed)
color_dir = Path(DATASET_DIR) / "train_color"
if color_dir.exists():
    for img in color_dir.iterdir():
        if img.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            caption_file = img.with_suffix(".txt")
            if not caption_file.exists():
                caption_file.write_text("")

print(f"Config saved to: {config_path}")
print(f"Found {len(list(color_dir.glob('*.png')))} training images")
```

**Cell 5 (Code) — Download base model:**
```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="gsdf/Counterfeit-V3.0",
    filename="Counterfeit-V3.0_fix_fp16.safetensors",
    cache_dir="/kaggle/working/models",
)
print(f"Base model: {model_path}")
```

**Cell 6 (Code) — Launch training:**
```python
!accelerate launch \
    --num_processes=1 \
    --mixed_precision="fp16" \
    /kaggle/working/sd-scripts/train_network.py \
    --pretrained_model_name_or_path="{model_path}" \
    --dataset_config="{config_path}" \
    --output_dir="{OUTPUT_DIR}" \
    --output_name="{LORA_NAME}" \
    --save_model_as="safetensors" \
    --max_train_epochs={EPOCHS} \
    --learning_rate={LEARNING_RATE} \
    --optimizer_type="AdamW8bit" \
    --network_module="networks.lora" \
    --network_dim={NETWORK_DIM} \
    --network_alpha={NETWORK_ALPHA} \
    --train_batch_size={BATCH_SIZE} \
    --resolution="{RESOLUTION},{RESOLUTION}" \
    --mixed_precision="fp16" \
    --save_precision="fp16" \
    --xformers \
    --cache_latents \
    --gradient_checkpointing \
    --seed=42
```

**Cell 7 (Code) — Verify output:**
```python
import os

lora_path = f"{OUTPUT_DIR}/{LORA_NAME}.safetensors"
if os.path.exists(lora_path):
    size_mb = os.path.getsize(lora_path) / (1024 * 1024)
    print(f"LoRA saved: {lora_path} ({size_mb:.1f} MB)")
    print("Download this file and use with: python main.py colorize panel.png ref.png --lora path/to/lora.safetensors")
else:
    print("ERROR: LoRA file not found. Check training logs above.")
    for f in os.listdir(OUTPUT_DIR):
        print(f"  {f}")
```

**Step 2: Commit**

```bash
git add notebooks/kaggle_train.ipynb
git commit -m "feat: add Kaggle LoRA training notebook for style-specific manga colorization"
```

---

## Task 9: Run All Tests & Final Verification

**Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: All tests pass (approximately 18 tests).

**Step 2: Verify CLI help**

```bash
python main.py --help
python main.py colorize --help
python main.py preprocess --help
```

**Step 3: Final commit (if any cleanup needed)**

```bash
git log --oneline
```

Verify commit history looks clean.

---

## Smoke Test (Manual, requires GPU + model downloads)

After all tasks are complete, run this manually on your RTX 4060:

```bash
# 1. Install deps
pip install -e ".[dev]"

# 2. Test preprocessing with any anime image
python main.py preprocess /path/to/colored/images -o test_dataset

# 3. Test inference (will download ~6GB of models on first run)
python main.py colorize test_dataset/train_bw/image.png /path/to/reference.png -o test_output.png --seed 42

# 4. Test with color harmonization
python main.py colorize panel2.png ref.png -o panel2_colored.png --harmonize test_output.png
```
