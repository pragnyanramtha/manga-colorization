# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Always use `uv` for Python operations.

```bash
# Install dependencies
uv sync --extra dev

# Run all tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_preprocess.py -v

# Run a single test by name
uv run pytest tests/test_preprocess.py::test_extract_lines_output_is_grayscale -v

# Run with coverage
uv run pytest tests/ --cov=src

# Run the CLI
uv run manga-colorize
```

## Architecture

This is a reference-based manga colorization pipeline: B&W manga panels → colored images using SD1.5 + ControlNet + IP-Adapter. No text prompts — structure comes from ControlNet (lineart), colors from an IP-Adapter reference image.

### Module layout

- **`src/preprocess.py`** — Generates synthetic B&W manga from colored training images. Pipeline: edge detection (`extract_lines`) → halftone dots (`simulate_screentones`) → paper noise (`add_paper_texture`) → combined via `create_synthetic_manga`. `process_directory` batches paired (B&W, color) training data.
- **`src/postprocess.py`** — `harmonize_colors` matches color distributions across panels using LAB color space: preserves lightness (L), matches color channels (a, b) to a reference histogram.
- **`src/pipeline.py`** — *(not yet implemented)* Core inference class wrapping SD1.5 + ControlNet + IP-Adapter.
- **`main.py`** — *(stub)* CLI entry point (`manga-colorize = "main:main"`).

### Planned pipeline class (Task 6 in `docs/plans/`)

The `ColorizationPipeline` class will wrap:
1. `StableDiffusionControlNetPipeline` with Counterfeit-V3.0 base model
2. ControlNet lineart model (`lllyasviel/sd-controlnet-lineart` or s2 variant)
3. IP-Adapter Plus for reference image colors

**Critical API constraints from research (do not deviate):**
- Counterfeit-V3.0 requires `from_single_file()`, not `from_pretrained()`
- IP-Adapter Plus requires explicit `CLIPVisionModelWithProjection` from `h94/IP-Adapter` subfolder `models/image_encoder`
- The s2 ControlNet variant needs `num_hidden_layers=11` on the CLIP text encoder
- `enable_model_cpu_offload()` MUST be called AFTER `load_ip_adapter()` — reversed order errors
- Weight filename: `ip-adapter-plus_sd15.safetensors`

### Implementation status

Tasks 1–5 complete (project setup, postprocessing, line extraction, screentones, full preprocessing pipeline + 19 passing tests). Tasks 6–9 pending: inference pipeline, CLI, Kaggle training notebook.

See `docs/plans/2026-02-20-manga-colorizer-v1.md` for the full task-by-task plan.
