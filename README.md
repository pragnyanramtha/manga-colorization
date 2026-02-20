# Manga Colorizer V1

AI pipeline for colorizing black & white manga panels using reference-based style transfer. Provide a B&W panel and a colored reference image (character sheet, colored page, etc.) — the model transfers the colors without requiring text prompts.

**Stack:** Stable Diffusion 1.5 + ControlNet (lineart) + IP-Adapter (color transfer)
**Requirements:** NVIDIA GPU with 8GB+ VRAM (tested on RTX 4060)

## Setup

```bash
# Clone and install
git clone <repo-url> manga-colorizer
cd manga-colorizer
pip install -e ".[dev]"
```

Models are downloaded automatically on first run (~6GB total):
- Counterfeit-V3.0 (SD1.5 anime checkpoint)
- ControlNet lineart anime
- IP-Adapter Plus with ViT-H encoder

## Usage

### Colorize a single panel

```bash
python main.py colorize panel_bw.png reference_colored.png -o result.png
```

Options:
```
-o, --output          Output path (default: output.png)
--lora PATH           Path to a style-specific LoRA (.safetensors)
--steps N             Inference steps (default: 20, higher = better quality)
--seed N              Random seed for reproducibility
--harmonize PATH      Match colors to a reference image (e.g., previous panel)
--denoise             Denoise input before colorization (for scanned manga)
--denoise-strength N  Denoising strength (default: 25, higher = more smoothing)
```

Examples:
```bash
# Basic colorization
python main.py colorize page1.png naruto_ref.png -o page1_colored.png

# With LoRA style and reproducible seed
python main.py colorize page1.png ref.png --lora demon_slayer.safetensors --seed 42

# Scanned manga with denoising
python main.py colorize scan_page.png ref.png --denoise --denoise-strength 30

# Match colors to a previously colorized panel
python main.py colorize page2.png ref.png --harmonize page1_colored.png
```

### Batch colorize a directory

Process an entire manga chapter at once:

```bash
python main.py batch ./panels/ reference.png -o ./colored/
```

Options:
```
-o, --output-dir      Output directory (default: output)
--harmonize           Auto-harmonize colors between consecutive panels
--denoise             Denoise all inputs before colorization
--denoise-strength N  Denoising strength (default: 25)
--lora, --steps, --seed  Same as colorize command
```

Example — full chapter with color consistency:
```bash
python main.py batch ./chapter_01/ character_sheet.png -o ./chapter_01_colored/ --harmonize --steps 25
```

Panels are processed in filename order. With `--harmonize`, each panel's colors are matched to the previous panel's output, keeping the palette consistent across pages.

### Generate training data

Create synthetic B&W manga from colored anime images (for LoRA training):

```bash
python main.py preprocess ./colored_images/ -o ./dataset/
```

This produces paired folders:
```
dataset/
  train_bw/     # Synthetic manga (lineart + screentones + paper texture)
  train_color/  # Original colored images
```

### Train a style LoRA (Kaggle)

1. Run `preprocess` on 200+ colored images of your target manga style
2. Upload the `dataset/` folder to Kaggle
3. Open `notebooks/kaggle_train.ipynb` and run all cells
4. Download the resulting `.safetensors` file
5. Use it: `python main.py colorize panel.png ref.png --lora your_style.safetensors`

## Project Structure

```
manga-colorizer/
  src/
    pipeline.py      # Core AI pipeline (SD1.5 + ControlNet + IP-Adapter)
    preprocess.py     # Synthetic manga generation (edges, screentones, texture)
    postprocess.py    # Color harmonization + denoising utilities
  notebooks/
    kaggle_train.ipynb  # LoRA training notebook for Kaggle (2x T4)
  models/              # Auto-downloaded model cache (gitignored)
  main.py              # CLI entry point
```

## Denoising

The `--denoise` flag is for **scanned physical manga** that has compression artifacts, paper grain, or scanner noise. It uses OpenCV non-local means denoising to clean the image before passing it to ControlNet.

**Do not use on clean digital manga** — it will soften fine lines and screentone detail.

Tune `--denoise-strength` based on scan quality:
- `15-20`: Light cleanup (minor JPEG artifacts)
- `25`: Default (moderate scan noise)
- `35-50`: Heavy denoising (poor quality scans)
