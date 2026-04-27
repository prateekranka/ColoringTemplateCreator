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
from .output import save, save_preview, save_svg


# ---------------------------------------------------------------------------
# Preprocessing — background detection and removal
# ---------------------------------------------------------------------------

def detect_and_remove_dark_background(img_rgb: np.ndarray) -> np.ndarray:
    """Detect if the image has a dominant dark background and replace it with white.

    Pop art with dark backgrounds (e.g., deep blue) causes the pipeline to
    capture the entire background as "outline", producing an inverted result.
    This function detects such backgrounds by analyzing border pixels and
    flood-fills to find the connected background region.

    To avoid artificially inflating the region score (which flood-fills from
    borders to find "outside" area), we dilate the background mask slightly
    before replacement so no outline artifact remains at the boundary.

    Args:
        img_rgb: Input image as HxWx3 RGB array.

    Returns:
        Image with dark background replaced by white, or unchanged if no
        dark background detected.
    """
    h, w = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Collect border pixels (outer 5% frame)
    border_width = max(1, int(min(h, w) * 0.05))
    border_v = np.concatenate([
        hsv[:border_width, :, 2].ravel(),       # top
        hsv[-border_width:, :, 2].ravel(),       # bottom
        hsv[:, :border_width, 2].ravel(),        # left
        hsv[:, -border_width:, 2].ravel(),       # right
    ])
    border_s = np.concatenate([
        hsv[:border_width, :, 1].ravel(),
        hsv[-border_width:, :, 1].ravel(),
        hsv[:, :border_width, 1].ravel(),
        hsv[:, -border_width:, 1].ravel(),
    ])

    median_value = np.median(border_v)
    if median_value > 60:
        # Border is not dark — no dark background to remove
        return img_rgb

    # Skip binary/grayscale images (already-processed coloring outputs).
    # A colored dark background has saturation; a black-and-white image does not.
    median_sat = np.median(border_s)
    if median_sat < 15:
        # Very low saturation border = grayscale/binary image, not colored bg
        return img_rgb

    # Flood-fill from all four corners to find connected dark background.
    # Work on a grayscale version for flood-fill.
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Create a mask for flood-fill (must be 2px larger than image)
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # Flood-fill from border pixels with tight tolerance
    fill_tolerance = (30, 30, 30, 30)  # low/high tolerance for flood-fill
    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]

    # Also seed along borders to handle non-uniform dark backgrounds
    border_seeds = set(corners)
    step = max(1, min(h, w) // 20)
    for x in range(0, w, step):
        border_seeds.add((x, 0))
        border_seeds.add((x, h - 1))
    for y in range(0, h, step):
        border_seeds.add((0, y))
        border_seeds.add((w - 1, y))

    fill_canvas = gray.copy()
    for (sx, sy) in border_seeds:
        if fill_canvas[sy, sx] < 80:  # Only seed from dark pixels
            cv2.floodFill(
                fill_canvas, flood_mask, (sx, sy), 255,
                loDiff=fill_tolerance[:1], upDiff=fill_tolerance[:1],
                flags=cv2.FLOODFILL_FIXED_RANGE,
            )

    # The flood_mask marks filled pixels (1 = background)
    bg_mask = flood_mask[1:-1, 1:-1]  # Remove the 1px border padding

    # Check if background is significant (>20% of image)
    bg_ratio = np.sum(bg_mask > 0) / (h * w)
    if bg_ratio < 0.20:
        return img_rgb

    # Dilate the background mask slightly to eat into the boundary edge,
    # preventing outline artifacts at the background border
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bg_mask_dilated = cv2.dilate(bg_mask, dilate_k, iterations=2)

    # Replace background with white
    result = img_rgb.copy()
    result[bg_mask_dilated > 0] = [255, 255, 255]

    return result


# ---------------------------------------------------------------------------
# Auto-strategy selection
# ---------------------------------------------------------------------------

def select_strategy(img_rgb: np.ndarray, threshold: int = 70):
    """Analyse image statistics to choose the best extraction strategy.

    Heuristic:
      - If image is mostly bright with large dark blobs → dark subject on light bg
        → use EdgeDetectStrategy to capture subject boundaries as lines
      - If >2% of pixels are dark AND desaturated → strong black outlines present
        → use DarkPixelStrategy (fastest, cleanest for pop art)
      - If dark pixels exist but are concentrated in large blobs (fills, not outlines)
        → use EdgeDetectStrategy (dark areas are fills like hair/clothing, not drawn lines)
      - Otherwise → EdgeDetectStrategy (no clear outlines, rely on edge detection)

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

    # ------------------------------------------------------------------
    # Detect dark-subject-on-light-background.
    # When the subject itself is dark (black cat, silhouette, text on
    # bright paper) DarkPixelStrategy treats it as a solid fill and
    # _remove_fill_regions replaces it with a thin contour that often
    # breaks. EdgeDetectStrategy sees the brightness gradient at the
    # subject boundary and produces a natural closed outline.
    # ------------------------------------------------------------------
    bright_ratio = np.sum(gray > 180) / total
    if bright_ratio > 0.60:
        dark_mask = (gray < 100).astype(np.uint8) * 255
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            dark_mask, connectivity=8
        )
        large_dark_area = sum(
            stats[i, cv2.CC_STAT_AREA]
            for i in range(1, num_labels)
            if stats[i, cv2.CC_STAT_AREA] > 0.03 * total
        )
        if large_dark_area > 0.06 * total:
            # Significant dark mass on bright background → subject silhouette
            return EdgeDetectStrategy(method="adaptive")

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

        if surviving_ratio > 0.003:
            # Significant dark mass survives erosion → these are fills, not just outlines.
            # But there may still be some real outlines mixed in with the fills.
            # If outline ratio is high enough, dark pixel strategy with fill removal
            # in cleanup will handle it.
            if outline_ratio > 0.03:
                return DarkPixelStrategy(threshold=threshold)
            else:
                return EdgeDetectStrategy(method="adaptive")
        else:
            # Dark pixels are thin lines that disappear under erosion → true outlines
            return DarkPixelStrategy(threshold=threshold)
    else:
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
    threshold: int = 60,
    close_kernel_size: int = 3,
    min_component_area: int | None = None,
    smooth: bool = True,
    dpi: int = 300,
    transparent: bool = False,
    min_size: int = 3000,
    preview: bool = False,
    svg: bool = False,
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
        svg: If True, also save an SVG version alongside the PNG.

    Returns:
        Path to the saved coloring template PNG (or SVG if svg=True).

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

    # Stage 1.5: Preprocess — detect and remove dark backgrounds
    img_rgb = detect_and_remove_dark_background(img_rgb)

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

    # Optional SVG output
    if svg:
        result_path = save_svg(mask, output_path.with_suffix(".svg"), min_size=min_size)

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
