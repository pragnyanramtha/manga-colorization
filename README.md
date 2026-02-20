# Manga Colorizer V1

AI pipeline for colorizing black & white manga panels using reference-based style transfer. Provide a B&W panel and a colored reference image — the model transfers the colors without text prompts.

**Stack:** Stable Diffusion 1.5 + ControlNet (lineart) + IP-Adapter (color transfer)
**Requirements:** NVIDIA GPU with 8GB+ VRAM (tested on RTX 4060), Python 3.11+

## Quick Start

```bash
git clone <repo-url> manga-colorizer && cd manga-colorizer
pip install -e ".[dev]"

# Check everything is working
./colorize --check

# Colorize a single panel
./colorize page.png -r reference.png
```

Models are downloaded automatically on first run (~6GB total).

## Usage

### Single panel

```bash
./colorize page.png -r character_sheet.png
./colorize page.png -r ref.png -o result.png
./colorize page.png -r ref.png --lora demon_slayer.safetensors --seed 42
./colorize page.png -r ref.png --denoise                  # scanned manga
./colorize page.png -r ref.png --harmonize-ref prev.png   # match prev panel colors
```

### Batch (whole chapter)

```bash
./colorize -p ./chapter_01/ -r character_sheet.png
./colorize -p ./chapter_01/ -r ref.png -o ./colored/ --harmonize
./colorize -p ./chapter_01/ -r ref.png --denoise --harmonize --steps 25
```

With `--harmonize`, each panel's colors are matched to the previous panel's output, keeping the palette consistent across pages. Panels are processed in filename order.

### Training data

```bash
./colorize -t ./colored_anime_images/
./colorize -t ./colored_images/ -o ./my_dataset/
```

### All options

```
./colorize image.png              Colorize a single B&W panel
./colorize -p DIR                 Batch colorize a directory
./colorize -t DIR                 Generate synthetic training data
./colorize --check                Verify dependencies and GPU

-r, --reference FILE    Colored reference image (required for colorize/batch)
-o, --output PATH       Output file or directory
--lora FILE             LoRA .safetensors for style transfer
--steps N               Inference steps (default: 20)
--seed N                Random seed for reproducibility
--harmonize             Color consistency between consecutive panels (batch)
--harmonize-ref FILE    Match colors to a reference image (single panel)
--denoise               Denoise input before colorization (for scans)
--denoise-strength N    Denoising strength (default: 25)
```

## Training a Style LoRA on Kaggle

Train a LoRA to learn a specific manga's color palette (e.g., "Demon Slayer style", "Chainsaw Man style").

### Step 1: Collect training images

You need **200-500 colored images** from your target manga style. Sources:

| Source | How to get | Notes |
|--------|-----------|-------|
| **Anime screenshots** | VLC > Video > Take Snapshot, or `ffmpeg -i episode.mp4 -vf fps=1 frames/%04d.png` | Best source — clean, high-res, correct colors |
| **Official colored manga** | Digital manga stores (Shonen Jump, Manga Plus) | Some series have official colored editions |
| **Fan colorizations** | Manga coloring communities | Quality varies, curate carefully |
| **Danbooru/Safebooru** | Search `<character> solo colored` | Good for character sheets |

Tips:
- Prefer **high-res** images (1024px+ on longest side)
- Include **variety**: different characters, lighting, backgrounds
- Avoid: text-heavy panels, extreme close-ups, heavily stylized fan art
- More images = better, but 200 is the minimum for decent results

### Step 2: Generate training pairs

```bash
./colorize -t ./my_collected_images/ -o ./dataset/
```

This creates:
```
dataset/
  train_bw/     # Synthetic B&W manga (lineart + screentones + paper texture)
  train_color/  # Original colored images (targets)
```

### Step 3: Upload to Kaggle and train

1. Go to [kaggle.com/datasets/new](https://kaggle.com/datasets/new) and upload your `dataset/` folder
2. Create a new notebook, select **GPU T4 x2** accelerator
3. Upload or paste `notebooks/kaggle_train.ipynb`
4. Edit the config cell:
   ```python
   DATASET_DIR = "/kaggle/input/your-dataset-name"
   LORA_NAME = "my_manga_style"
   ```
5. Run all cells. Training takes ~2-4 hours on 2x T4.
6. Download `lora_output/my_manga_style.safetensors` from the output

### Step 4: Use your LoRA

```bash
./colorize page.png -r ref.png --lora my_manga_style.safetensors
```

### Training tips

- **Overfitting?** Reduce `EPOCHS` (try 5-8) or increase `NETWORK_DIM` to 64
- **Colors not matching?** Add more training images with clear, representative colors
- **Artifacts?** Try `LEARNING_RATE = 5e-5` (lower = safer)
- **Out of memory on Kaggle?** Keep `BATCH_SIZE = 1` and enable `--gradient_checkpointing`

### Recommended datasets to try first

| Style | Where to find images |
|-------|---------------------|
| Demon Slayer | Anime screenshots from episodes + Ufotable promo art |
| Spy x Family | Anime screenshots, clean art style with flat colors |
| Chainsaw Man | MAPPA anime screenshots, distinctive dark palette |
| Dragon Ball | Toei anime screenshots, classic cel-shading look |

## Denoising

The `--denoise` flag is for **scanned physical manga** with compression artifacts, paper grain, or scanner noise. Uses OpenCV non-local means denoising.

**Do not use on clean digital manga** — it softens fine lines.

```
--denoise-strength 15-20    Light (minor JPEG artifacts)
--denoise-strength 25       Default (moderate scan noise)
--denoise-strength 35-50    Heavy (poor quality scans)
```

## Project Structure

```
manga-colorizer/
  colorize              # Main entry point (executable)
  src/
    pipeline.py         # SD1.5 + ControlNet + IP-Adapter pipeline
    preprocess.py       # Synthetic manga generation
    postprocess.py      # Denoising + color harmonization
  notebooks/
    kaggle_train.ipynb  # Kaggle LoRA training notebook
  tests/                # Test suite (pytest)
  models/               # Auto-downloaded model cache (gitignored)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Close other GPU apps. The pipeline uses CPU offloading but needs ~7GB free VRAM. |
| Models downloading slowly | First run downloads ~6GB. Use a HuggingFace token: `huggingface-cli login` |
| Colors look wrong | Try a different reference image. The reference should show the character's canonical colors clearly. |
| Output is blurry | Increase `--steps` to 30-40. |
| Scanned manga looks noisy | Add `--denoise`. |
