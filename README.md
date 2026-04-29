# ColoringTemplateCreator

Converts pop art / bold-outline illustrations into clean coloring templates for iPad coloring apps (built for [Colorflow](https://github.com/prateekranka/colorflow)).

## How It Works

Pop art already has explicit black outlines drawn. Rather than running generic edge detection, this tool **directly extracts those outline pixels** using a luminance threshold combined with an HSV saturation gate — distinguishing true black outlines from dark-colored fills (e.g. deep navy backgrounds).

The result is a clean, high-resolution PNG with black lines on a white (or transparent) background, ready to import into your coloring app.

## Installation

```bash
pip install -r requirements.txt
# or install as a package:
pip install -e .
```

## Usage

### Single image
```bash
python -m coloring_template photo.png
# Output: ./output/photo_coloring.png
```

### Batch processing
```bash
python -m coloring_template *.png -o ./templates/
```

### Custom output directory
```bash
python -m coloring_template art.jpg -o ~/Desktop/coloring/
```

### With preview (side-by-side comparison)
```bash
python -m coloring_template art.png --preview
# Also saves: ./output/art_preview.png
```

### Generate a new coloring template from a prompt
```bash
export ANTHROPIC_API_KEY=...
python -m coloring_template --generate "a fox sitting in autumn leaves"
# Saves PNG, SVG, and validation JSON into ./output
```

### Transparent background (for layered import into Colorflow)
```bash
python -m coloring_template art.png --transparent
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output-dir` | `./output` | Output directory |
| `-s`, `--strategy` | `auto` | Extraction strategy (see below) |
| `-t`, `--threshold` | `60` | Dark pixel threshold 0–255 |
| `--dpi` | `300` | Output DPI metadata |
| `--min-size` | `3000` | Minimum output resolution (longest side in px) |
| `--transparent` | off | Transparent background (RGBA PNG) |
| `--preview` | off | Save side-by-side comparison image |
| `--close-kernel` | `3` | Morphological close kernel size (gap sealing) |
| `--no-smooth` | off | Disable edge smoothing |
| `--generate SUBJECT` | off | Generate a vector-first coloring page from a text prompt |
| `--max-attempts` | `3` | Generation repair attempts |
| `--model` | env/default | Model override for generation |

## Strategies

| Strategy | Best For |
|----------|----------|
| `auto` | Default — analyses image stats to pick the best strategy |
| `dark` | Images with bold black outlines (most pop art) |
| `edge` | Colored outlines or images without clear black lines |
| `canny` | Photographic images or faint edges |
| `combined` | Merges `dark` + `edge` results |

### Threshold tuning (`-t` / `--threshold`)

The threshold controls how dark a pixel must be to count as an outline candidate. It is only used by the `dark`, `combined`, and `auto` strategies.

- **Lower values (30–50)**: Only the boldest, darkest lines are extracted. Fewer false positives from dark backgrounds.
- **Default (60)**: Works well for most pop art with solid black outlines.
- **Higher values (70–100)**: Captures lighter outlines but may pick up shadow regions or dark fills.

## Pipeline

```
Input image
    │
    ▼
1. Load & normalize (upscale if < 3000px)
    │
    ▼
2. Extract line art (strategy-based)
   ├── DarkPixel: threshold + HSV saturation gate
   ├── EdgeDetect: adaptive threshold or Canny
   └── Combined: merge of both
    │
    ▼
3. Cleanup
   ├── Morphological CLOSE (seal outline gaps)
   ├── Remove noise blobs (connected components < min area)
   └── Light Gaussian smooth + re-threshold
    │
    ▼
4. Output PNG
   ├── Black lines on white background (standard)
   └── Black lines on transparent background (--transparent)
```

## Project Structure

```
src/coloring_template/
├── __init__.py
├── __main__.py          # python -m coloring_template entry point
├── cli.py               # argparse CLI
├── pipeline.py          # Orchestrates stages + auto-strategy selection
├── cleanup.py           # Morphological cleanup
├── output.py            # PNG output with DPI
└── strategies/
    ├── base.py          # Abstract base class
    ├── dark_pixel.py    # Primary: threshold + saturation gate
    ├── edge_detect.py   # Fallback: adaptive threshold / Canny
    └── combined.py      # Union of dark + edge
```
