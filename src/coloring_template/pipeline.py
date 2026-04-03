"""Core conversion pipeline: pop art image → coloring template.

Orchestrates the four stages:
  1. Load & normalize
  2. Extract line art (strategy-based)
  3. Cleanup (morphological ops + noise removal)
  4. Save output PNG
"""

import cv2
import numpy as np
from pathlib import Path

from .strategies import (
    DarkPixelStrategy,
    EdgeDetectStrategy,
    CombinedStrategy,
    KMeansSegmentStrategy,
    XDoGStrategy,
)
from .cleanup import clean
from .output import save, save_preview


# ---------------------------------------------------------------------------
# Auto-strategy selection
# ---------------------------------------------------------------------------

def _count_distinct_colors(img_rgb: np.ndarray, sample_size: int = 50000) -> int:
    """Estimate the number of distinct color clusters in an image.

    Uses a quick mini-K-means on a random pixel sample to measure how well
    a small number of clusters explains the image.  Returns the estimated
    number of visually distinct color regions.
    """
    h, w = img_rgb.shape[:2]
    total = h * w

    # Sample pixels for speed
    if total > sample_size:
        indices = np.random.default_rng(42).choice(total, sample_size, replace=False)
        pixels = img_rgb.reshape(-1, 3)[indices].astype(np.float32)
    else:
        pixels = img_rgb.reshape(-1, 3).astype(np.float32)

    # Try K=4 and K=12 — if K=4 already explains the image well (low
    # compactness), the image has few distinct colors (flat palette).
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 2.0)
    compactness_4, _, _ = cv2.kmeans(
        pixels, 4, None, criteria, attempts=2, flags=cv2.KMEANS_PP_CENTERS
    )
    compactness_12, _, _ = cv2.kmeans(
        pixels, 12, None, criteria, attempts=2, flags=cv2.KMEANS_PP_CENTERS
    )

    # Ratio: if doubling K doesn't help much, the image is already flat
    ratio = compactness_4 / max(compactness_12, 1e-6)
    # ratio < 2 means 4 clusters already capture most variance → flat palette
    # ratio > 4 means lots more structure at higher K → complex/photographic
    return ratio


def select_strategy(img_rgb: np.ndarray, threshold: int = 60):
    """Analyse image statistics to choose the best extraction strategy.

    Heuristic (updated with segmentation-aware logic):
      1. If >2% of pixels are dark AND desaturated → strong black outlines present
         → use DarkPixelStrategy (fastest, cleanest for pop art with outlines)
      2. If dark pixels are concentrated in large blobs (fills, not outlines)
         → use EdgeDetectStrategy (dark areas are fills like hair/clothing)
      3. If the image has a flat color palette (few distinct colors, like pop art
         or Matisse-style blocks) → use KMeansSegmentStrategy (guarantees closed
         contours for flood-fill, ideal for flat-color illustrations)
      4. Otherwise → XDoGStrategy (best general-purpose line art extraction)

    Args:
        img_rgb: Input image as HxWx3 RGB array.
        threshold: Dark pixel threshold (passed through to DarkPixelStrategy).

    Returns:
        An instantiated strategy object.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    total = gray.size

    # True outline pixels: dark AND low saturation
    outline_mask = (gray < 50) & (saturation < 60)
    outline_ratio = np.sum(outline_mask) / total

    if outline_ratio > 0.02:
        # Check if dark pixels form thin outlines or large fills.
        # Erode the dark mask — outlines disappear, fills survive.
        dark_binary = (outline_mask.astype(np.uint8) * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        eroded = cv2.erode(dark_binary, kernel, iterations=2)
        surviving_ratio = np.sum(eroded > 0) / total

        if surviving_ratio > 0.005:
            # Significant dark mass survives erosion → these are fills, not just outlines.
            if outline_ratio > 0.04:
                return DarkPixelStrategy(threshold=threshold)
            else:
                return EdgeDetectStrategy(method="adaptive")
        else:
            # Dark pixels are thin lines that disappear under erosion → true outlines
            return DarkPixelStrategy(threshold=threshold)

    # No clear black outlines — check if the image has flat color regions
    # (pop art, Matisse-style, cartoon) where K-means segmentation excels.
    color_complexity = _count_distinct_colors(img_rgb)

    if color_complexity < 3.0:
        # Flat palette — K-means segmentation produces guaranteed closed contours.
        # Estimate optimal K from the complexity ratio.
        k = 6 if color_complexity < 2.0 else 10
        return KMeansSegmentStrategy(k=k)

    # Complex image without outlines — XDoG produces the cleanest line art
    return XDoGStrategy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

STRATEGY_MAP = {
    "dark": lambda t: DarkPixelStrategy(threshold=t),
    "edge": lambda _: EdgeDetectStrategy(method="adaptive"),
    "canny": lambda _: EdgeDetectStrategy(method="canny"),
    "combined": lambda t: CombinedStrategy(threshold=t),
    "kmeans": lambda _: KMeansSegmentStrategy(),
    "xdog": lambda _: XDoGStrategy(),
    "auto": select_strategy,  # called with (img_rgb, threshold)
}


def convert(
    input_path: str | Path,
    output_path: str | Path,
    strategy: str = "auto",
    threshold: int = 60,
    close_kernel_size: int = 3,
    min_component_area: int | None = None,
    smooth: bool = True,
    dpi: int = 300,
    transparent: bool = False,
    min_size: int = 3000,
    preview: bool = False,
    max_gap: int = 10,
) -> Path:
    """Convert a pop art image into a coloring template PNG.

    Args:
        input_path: Path to the input image (JPEG, PNG, WebP, etc.).
        output_path: Path for the output PNG.
        strategy: Extraction strategy: 'auto', 'dark', 'edge', 'canny',
                  'combined', 'kmeans', 'xdog'.
        threshold: Dark pixel threshold (0-255). Only used by dark/combined/auto.
        close_kernel_size: Morphological close kernel size for gap-sealing.
        min_component_area: Minimum blob area to keep (None = auto-scale).
        smooth: Whether to apply edge smoothing after cleanup.
        dpi: Output DPI metadata.
        transparent: If True, background is transparent (RGBA PNG).
        min_size: Minimum pixel length of the longest output side.
        preview: If True, also save a side-by-side preview image.
        max_gap: Maximum pixel distance for endpoint gap bridging (0 = disabled).

    Returns:
        Path to the saved coloring template PNG.

    Raises:
        FileNotFoundError: If input_path does not exist or cannot be read.
        ValueError: If an unknown strategy name is given.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if strategy not in STRATEGY_MAP:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Choose from: {', '.join(STRATEGY_MAP)}"
        )

    # Stage 1: Load & normalize
    img_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Stage 2: Extract line art
    if strategy == "auto":
        strat = select_strategy(img_rgb, threshold)
    else:
        strat = STRATEGY_MAP[strategy](threshold)

    mask = strat.extract(img_rgb)

    # Stage 3: Cleanup
    mask = clean(
        mask,
        close_kernel_size=close_kernel_size,
        min_component_area=min_component_area,
        smooth=smooth,
        bridge_gaps=max_gap > 0,
        max_gap=max_gap,
    )

    # Stage 4: Save output
    result_path = save(mask, output_path, dpi=dpi, transparent=transparent, min_size=min_size)

    # Optional preview
    if preview:
        preview_path = output_path.with_name(output_path.stem + "_preview.png")
        save_preview(img_rgb, mask, preview_path, dpi=min(dpi, 150))

    return result_path


def convert_batch(
    input_paths: list[Path],
    output_dir: Path,
    **kwargs,
) -> list[Path]:
    """Convert multiple images to coloring templates.

    Args:
        input_paths: List of input image paths.
        output_dir: Directory for output files. Created if it doesn't exist.
        **kwargs: Forwarded to convert() for each image.

    Returns:
        List of successfully created output paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    errors = []

    for path in input_paths:
        out_path = output_dir / (path.stem + "_coloring.png")
        try:
            result = convert(path, out_path, **kwargs)
            results.append(result)
            print(f"  OK  {path.name} → {result.name}")
        except Exception as exc:
            errors.append((path, exc))
            print(f"  ERR {path.name}: {exc}")

    if errors:
        print(f"\n{len(errors)} error(s) occurred:")
        for path, exc in errors:
            print(f"  {path.name}: {exc}")

    return results
