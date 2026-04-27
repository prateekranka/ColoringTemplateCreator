"""Quality metrics for coloring template masks.

Three metrics, each 0.0–1.0, combined into a composite score (higher = better):

  1. continuity_score  — fraction of outline pixels in large connected components
                         (long unbroken lines score high; fragmented dots score low)
  2. noise_score       — 1 - (noise_blob_count / total_components), penalises speckle
  3. region_score      — fraction of background that is enclosed by outlines
                         (colorable closed regions; higher = more "colorable areas")

composite = 0.4 * continuity + 0.3 * noise + 0.3 * region
"""

import cv2
import numpy as np


def _continuity_score(mask: np.ndarray) -> float:
    """Fraction of outline pixels belonging to large connected components.

    A large component threshold is set at 1% of total image pixels.
    Components below this are treated as fragmented noise strokes.
    """
    total_outline = int(np.sum(mask > 0))
    if total_outline == 0:
        return 0.0

    total_pixels = mask.shape[0] * mask.shape[1]
    large_threshold = max(50, int(total_pixels * 0.001))  # 0.1% of image

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    large_pixels = sum(
        stats[i, cv2.CC_STAT_AREA]
        for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= large_threshold
    )
    return min(1.0, large_pixels / total_outline)


def _noise_score(mask: np.ndarray) -> float:
    """1 - penalised noise ratio.

    Counts blobs smaller than a noise threshold. More tiny blobs = lower score.
    Penalty is capped so a very noisy image scores ~0 but not below.
    """
    total_pixels = mask.shape[0] * mask.shape[1]
    noise_threshold = max(10, int(total_pixels * 0.00005))  # 0.005% of image

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return 1.0  # No components at all (blank mask)

    total_components = num_labels - 1  # Exclude background
    noise_count = sum(
        1 for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] < noise_threshold
    )

    # Penalty: noise_count / total, but saturate so the score can't go negative
    noise_ratio = noise_count / max(1, total_components)
    return max(0.0, 1.0 - noise_ratio)


def _region_score(mask: np.ndarray) -> float:
    """Fraction of background pixels enclosed within outline boundaries.

    Uses flood fill from the image border to find "outside" pixels.
    Everything not reachable from the border and not an outline = enclosed region.
    """
    h, w = mask.shape

    # Binary: outline pixels are foreground
    binary = (mask > 0).astype(np.uint8)

    # Flood fill from all border pixels to find the "outside" background
    flood_seed = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flood_canvas = binary.copy()
    # Fill from top-left corner with value 1
    cv2.floodFill(flood_canvas, flood_seed, (0, 0), 1)

    # Also seed from all four borders to handle irregular borders
    border_seeds = (
        [(0, x) for x in range(w)] +
        [(h - 1, x) for x in range(w)] +
        [(y, 0) for y in range(h)] +
        [(y, w - 1) for y in range(h)]
    )
    for (r, c) in border_seeds:
        if flood_canvas[r, c] == 0:  # Not yet filled and not an outline
            seed_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            cv2.floodFill(flood_canvas, seed_mask, (c, r), 1)

    # Enclosed pixels: background (not outline) AND not reachable from border
    background = binary == 0
    outside = flood_canvas == 1
    enclosed = background & ~outside

    total_background = int(np.sum(background))
    if total_background == 0:
        return 0.0

    enclosed_count = int(np.sum(enclosed))
    return min(1.0, enclosed_count / total_background)


def score(mask: np.ndarray) -> dict:
    """Compute all quality metrics for a binary mask.

    Args:
        mask: Binary mask (uint8) where 255 = outline pixel, 0 = background.

    Returns:
        Dict with keys: continuity, noise, region, composite.
        All values are floats in [0.0, 1.0]; higher is better.
    """
    continuity = _continuity_score(mask)
    noise = _noise_score(mask)
    region = _region_score(mask)
    composite = 0.4 * continuity + 0.3 * noise + 0.3 * region

    return {
        "continuity": round(continuity, 4),
        "noise": round(noise, 4),
        "region": round(region, 4),
        "composite": round(composite, 4),
    }
