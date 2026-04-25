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

from .strategies import DarkPixelStrategy, EdgeDetectStrategy, CombinedStrategy
from .cleanup import clean
from .output import save, save_preview


# ---------------------------------------------------------------------------
# Auto-strategy selection
# ---------------------------------------------------------------------------

def select_strategy(img_rgb: np.ndarray, threshold: int = 80):
    """Analyse image statistics to choose the best extraction strategy.

    Decision tree (in order):
      1. If the image has heavy SOLID DARK FILLS (large painted regions, e.g.
         multi-panel Instagram posts with hair/clothing rendered solid black),
         using DarkPixelStrategy would emit those fills as huge blobs even
         after fill-removal cleanup. Route those images to EdgeDetectStrategy
         which captures only the boundaries between regions.
      2. Else if there is enough dark + desaturated mass to be confident the
         image has hand-drawn black outlines, use DarkPixelStrategy.
      3. Else fall back to EdgeDetectStrategy (no clear outlines).

    The "heavy fills" test uses the FULL dark mask (gray < 50, regardless of
    saturation) eroded by 7px ellipse — this captures both pure-black and
    saturated-dark fills like dark skin tones, deep red lips, navy clothing.

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

    # ---- 1. Heavy-fill detection (any-saturation dark mass) ----
    dark_full = (gray < 50).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    eroded_full = cv2.erode(dark_full, kernel, iterations=2)
    full_fill_ratio = np.sum(eroded_full > 0) / total

    if full_fill_ratio > 0.01:
        # Substantial solid dark areas (regardless of color) → painted fills
        # not outlines → edge detection on region boundaries is better.
        return EdgeDetectStrategy(method="adaptive")

    # ---- 2. True-outline detection (dark AND desaturated) ----
    outline_mask = (gray < 50) & (saturation < 60)
    outline_ratio = np.sum(outline_mask) / total

    if outline_ratio > 0.02:
        # Distinguish thin-outline images from those with smaller saturated fills
        # using the desaturated-only erosion test (preserves earlier behaviour).
        dark_desat = (outline_mask.astype(np.uint8) * 255)
        eroded_desat = cv2.erode(dark_desat, kernel, iterations=2)
        surviving_ratio = np.sum(eroded_desat > 0) / total

        if surviving_ratio > 0.005 and outline_ratio <= 0.04:
            # Some desaturated fills present and outline mass moderate → edges.
            return EdgeDetectStrategy(method="adaptive")
        # Otherwise: bold black outlines dominate → DarkPixel.
        return DarkPixelStrategy(threshold=threshold)

    # ---- 3. Fallback ----
    return EdgeDetectStrategy(method="adaptive")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

STRATEGY_MAP = {
    "dark": lambda t: DarkPixelStrategy(threshold=t),
    "edge": lambda _: EdgeDetectStrategy(method="adaptive"),
    "canny": lambda _: EdgeDetectStrategy(method="canny"),
    "combined": lambda t: CombinedStrategy(threshold=t),
    "auto": select_strategy,  # called with (img_rgb, threshold)
}


def convert(
    input_path: str | Path,
    output_path: str | Path,
    strategy: str = "auto",
    threshold: int = 80,
    close_kernel_size: int = 3,
    min_component_area: int | None = None,
    smooth: bool = True,
    dpi: int = 300,
    transparent: bool = False,
    min_size: int = 3000,
    preview: bool = False,
) -> Path:
    """Convert a pop art image into a coloring template PNG.

    Args:
        input_path: Path to the input image (JPEG, PNG, WebP, etc.).
        output_path: Path for the output PNG.
        strategy: Extraction strategy: 'auto', 'dark', 'edge', 'canny', 'combined'.
        threshold: Dark pixel threshold (0-255). Only used by dark/combined/auto.
        close_kernel_size: Morphological close kernel size for gap-sealing.
        min_component_area: Minimum blob area to keep (None = auto-scale).
        smooth: Whether to apply edge smoothing after cleanup.
        dpi: Output DPI metadata.
        transparent: If True, background is transparent (RGBA PNG).
        min_size: Minimum pixel length of the longest output side.
        preview: If True, also save a side-by-side preview image.

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
