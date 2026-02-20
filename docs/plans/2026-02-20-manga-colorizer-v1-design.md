# Manga Colorizer V1 — Design Document

## Goal

Build a local, offline-capable AI pipeline that converts B&W manga panels into fully colored images using reference-based style transfer. No text prompts — the user provides a B&W panel and a colored reference image.

## Constraints

- Must run on 8GB VRAM (NVIDIA RTX 4060)
- Must be trainable on Kaggle (2x T4)
- No-prompt architecture: BW panel + reference image only

## Technical Stack

- **Base Model:** Stable Diffusion 1.5
- **Checkpoint:** Counterfeit-V3.0 or AnyLoRA (anime-specific)
- **Structural Control:** ControlNet-v1.1-Lineart-Anime
- **Style/Color Control:** IP-Adapter-Plus-SD15
- **Libraries:** HuggingFace Diffusers, OpenCV, scikit-image

## Modules

### Module A: Data Preprocessing (`src/preprocess.py`)

Converts colored anime images into synthetic manga training pairs.

- **Input:** Folder of colored anime/manga images
- **Output:** Paired BW (input) and color (target) folders
- **Processing:** Line extraction (Canny/HED), screentone simulation (halftone dots, not simple desaturation), paper texture augmentation

### Module B: Inference Pipeline (`src/pipeline.py`)

Core `MangaColorizerPipeline` class.

- Loads SD1.5 + ControlNet + IP-Adapter
- Memory optimization via `enable_model_cpu_offload()`
- `colorize_panel(bw_image, reference_image, character_lora=None)`
- Resizes to SD1.5 native res (512x768 / 768x512)
- ControlNet scale: 1.1, IP-Adapter scale: 0.6
- Hardcoded negative prompt: "monochrome, greyscale, lowres, bad anatomy"

### Module C: Post-Processing (`src/postprocess.py`)

Color consistency via LAB-space histogram matching.

- Transfers L channel from generated image, a/b channels matched to reference
- Prevents color flickering between adjacent panels

## File Structure

```
manga-colorizer/
├── models/                  # Downloaded checkpoints (gitignored)
├── src/
│   ├── preprocess.py        # Synthetic manga data generation
│   ├── pipeline.py          # Core AI pipeline class
│   └── postprocess.py       # Histogram matching utilities
├── notebooks/
│   └── kaggle_train.ipynb   # Kaggle LoRA training notebook
├── requirements.txt
└── main.py                  # CLI entry point
```

## Development Phases

### Phase 1: Local Inference Skeleton

Get the pipeline running on RTX 4060 with pre-trained models. Success: feed a BW panel + reference image, get a colored panel out.

### Phase 2: Kaggle LoRA Training

Train style-specific LoRAs on Kaggle. Success: model colors images using the exact palette of the target manga.
