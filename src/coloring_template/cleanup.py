"""Morphological cleanup for line art masks.

Cleans up raw extraction masks by:
  1. Removing large solid fill regions that aren't outlines
  2. Closing small gaps in outlines (morphological CLOSE)
  3. Removing isolated noise blobs below a minimum area
  4. Smoothing jagged edges with a light Gaussian pass + re-threshold
  5. Re-removing noise blobs that the smoothing pass may have created
"""

import cv2
import numpy as np


def _remove_fill_regions(mask: np.ndarray, max_fill_ratio: float = 0.02) -> np.ndarray:
    """Remove large solid regions that are fills, not outlines.

    Real outlines are thin lines. Large filled areas (like dark ears on a dog,
    or dark hair fills) should be reduced to just their boundary edges.

    Strategy: erode the mask aggressively. Anything that survives heavy erosion
    is a thick/solid region (a fill). Subtract those interiors but keep their
    edges by taking just the boundary contour.

    Args:
        mask: Binary mask (uint8) where 255 = outline pixel.
        max_fill_ratio: Maximum ratio of image area a single component can occupy
                        before being treated as a fill region. Default 0.02 (2%).

    Returns:
        Mask with fill interiors replaced by boundary outlines.
    """
    total_pixels = mask.shape[0] * mask.shape[1]
    max_area = int(total_pixels * max_fill_ratio)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    result = mask.copy()

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < max_area:
            continue

        # This component is suspiciously large — check if it's a solid fill
        # by measuring its "compactness" (area / convex hull area).
        # Also check aspect ratio — long thin lines can be large but aren't fills.
        comp_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue

        # Solidity: how much of the convex hull is filled
        solidity = area / hull_area

        # Bounding box aspect ratio
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        bbox_area = bw * bh
        fill_ratio = area / bbox_area if bbox_area > 0 else 0

        # A solid fill region has high solidity AND high bounding-box fill ratio
        # Thin outlines/lines have low fill ratio even if they span large areas
        if solidity > 0.7 and fill_ratio > 0.5:
            # Replace with just the boundary contour (3px thick)
            result[labels == i] = 0
            cv2.drawContours(result, [cnt], -1, 255, 3)

    return result


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop connected components smaller than min_area pixels."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def clean(
    mask: np.ndarray,
    close_kernel_size: int = 5,
    min_component_area: int | None = None,
    smooth: bool = True,
    remove_fills: bool = False,
    max_fill_ratio: float = 0.02,
) -> np.ndarray:
    """Clean a binary extraction mask for use as a coloring template.

    Args:
        mask: Binary mask (uint8) where 255 = outline pixel.
        close_kernel_size: Size of the structuring element for morphological CLOSE.
                           Larger values seal wider gaps but may merge nearby lines.
                           Default 5 (was 3): seals 1-2px gaps that would leak fill.
        min_component_area: Minimum area in pixels for a connected component to be kept.
                            Components smaller than this are treated as noise and removed.
                            If None, auto-calculated as 0.005% of total image area
                            (scales naturally with resolution).
        smooth: If True, apply a light Gaussian blur then re-threshold to smooth
                jagged/aliased edges. Recommended for flood-fill coloring apps.
        remove_fills: If True, detect and convert large solid fill regions to just
                      their boundary outlines. Fixes dark ears/hair being solid black.
        max_fill_ratio: Maximum ratio of image area for a single connected component
                        before it's treated as a fill region. Default 0.02 (2%).

    Returns:
        Cleaned binary mask (uint8), same shape as input.
    """
    # 1. Remove large solid fills (convert to boundary outlines)
    if remove_fills:
        mask = _remove_fill_regions(mask, max_fill_ratio=max_fill_ratio)

    # 2. Morphological CLOSE: seal small gaps in outlines
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
    )
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 3. Remove noise blobs smaller than min_component_area
    if min_component_area is None:
        total_pixels = mask.shape[0] * mask.shape[1]
        min_component_area = max(80, int(total_pixels * 0.00005))

    cleaned = _remove_small_components(closed, min_component_area)

    # 4. Light Gaussian blur + re-threshold to smooth jagged edges.
    #    This can re-introduce tiny artefact components from antialiased
    #    pixels, so we re-run the size filter immediately after.
    if smooth:
        blurred = cv2.GaussianBlur(cleaned, (3, 3), 0.8)
        _, cleaned = cv2.threshold(blurred, 85, 255, cv2.THRESH_BINARY)
        # 5. Second-pass component filter to remove smoothing artefacts.
        cleaned = _remove_small_components(cleaned, min_component_area)

    return cleaned
