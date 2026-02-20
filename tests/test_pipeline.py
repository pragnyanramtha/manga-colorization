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
