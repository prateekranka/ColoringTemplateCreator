"""Morphological cleanup for line art masks.

Cleans up raw extraction masks by:
  1. Removing large solid fill regions that aren't outlines
  2. Closing small gaps in outlines (multi-pass morphological approach)
  3. Bridging nearby contour endpoints to seal larger gaps
  4. Removing isolated noise blobs below a minimum area
  5. Smoothing jagged edges with a light Gaussian pass + re-threshold
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
        if solidity > 0.5 and fill_ratio > 0.3:
            # Replace with just the boundary contour (3px thick)
            result[labels == i] = 0
            cv2.drawContours(result, [cnt], -1, 255, 3)

    return result


def _bridge_contour_gaps(mask: np.ndarray, max_gap: int = 10) -> np.ndarray:
    """Bridge small gaps between nearby contour endpoints.

    Finds line endpoints (pixels with only one neighbor) and connects pairs
    that are within max_gap pixels of each other.  This seals gaps that
    morphological CLOSE alone cannot handle without over-thickening lines.

    Inspired by the "trapped-ball" concept from cartoon segmentation research
    (Zhang et al., IEEE TVCG 2009) — regions where a ball gets trapped become
    fillable segments.  This is a lightweight approximation: instead of
    simulating ball physics, we directly connect nearby endpoints.

    Args:
        mask: Binary mask (uint8) where 255 = outline pixel.
        max_gap: Maximum distance in pixels between endpoints to bridge.

    Returns:
        Mask with gaps bridged by drawn lines.
    """
    # Find endpoints using hit-or-miss with endpoint kernels.
    # An endpoint is a foreground pixel with exactly one foreground neighbor.
    # We use a thinned version to find true endpoints.
    thinned = cv2.ximgproc.thinning(mask) if hasattr(cv2, 'ximgproc') else mask

    # Count neighbors for each foreground pixel
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbor_count = cv2.filter2D((thinned > 0).astype(np.uint8), -1, kernel)

    # Endpoints: foreground pixels with exactly 1 neighbor
    endpoints = (thinned > 0) & (neighbor_count == 1)
    ey, ex = np.where(endpoints)

    if len(ey) < 2:
        return mask

    result = mask.copy()

    # For each endpoint, find the nearest other endpoint within max_gap
    # and draw a line between them (greedy matching).
    coords = np.column_stack((ex, ey))
    used = set()

    for i in range(len(coords)):
        if i in used:
            continue
        pt1 = coords[i]
        best_j = -1
        best_dist = max_gap + 1

        for j in range(i + 1, len(coords)):
            if j in used:
                continue
            dist = np.sqrt((coords[j][0] - pt1[0]) ** 2 + (coords[j][1] - pt1[1]) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_j = j

        if best_j >= 0 and best_dist <= max_gap:
            pt2 = coords[best_j]
            cv2.line(result, tuple(pt1), tuple(pt2), 255, 1)
            used.add(i)
            used.add(best_j)

    return result


def clean(
    mask: np.ndarray,
    close_kernel_size: int = 3,
    min_component_area: int | None = None,
    smooth: bool = True,
    remove_fills: bool = True,
    max_fill_ratio: float = 0.02,
    bridge_gaps: bool = True,
    max_gap: int = 10,
) -> np.ndarray:
    """Clean a binary extraction mask for use as a coloring template.

    Args:
        mask: Binary mask (uint8) where 255 = outline pixel.
        close_kernel_size: Size of the structuring element for morphological CLOSE.
                           Larger values seal wider gaps but may merge nearby lines.
        min_component_area: Minimum area in pixels for a connected component to be kept.
                            Components smaller than this are treated as noise and removed.
                            If None, auto-calculated as 0.0005% of total image area
                            (scales naturally with resolution).
        smooth: If True, apply a light Gaussian blur then re-threshold to smooth
                jagged/aliased edges. Recommended for flood-fill coloring apps.
        remove_fills: If True, detect and convert large solid fill regions to just
                      their boundary outlines. Fixes dark ears/hair being solid black.
        max_fill_ratio: Maximum ratio of image area for a single connected component
                        before it's treated as a fill region. Default 0.02 (2%).
        bridge_gaps: If True, connect nearby contour endpoints to seal gaps that
                     morphological CLOSE cannot reach without over-thickening.
        max_gap: Maximum pixel distance between endpoints to bridge (default 10).

    Returns:
        Cleaned binary mask (uint8), same shape as input.
    """
    # 1. Remove large solid fills (convert to boundary outlines)
    if remove_fills:
        mask = _remove_fill_regions(mask, max_fill_ratio=max_fill_ratio)

    # 2. Morphological CLOSE: seal small gaps in outlines
    #    Use a two-pass approach: first a small kernel for tight gaps,
    #    then the user-specified kernel for broader sealing.
    if close_kernel_size >= 3:
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, small_kernel)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
    )
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 3. Bridge larger gaps by connecting nearby contour endpoints
    if bridge_gaps:
        closed = _bridge_contour_gaps(closed, max_gap=max_gap)

    # 4. Remove noise blobs smaller than min_component_area
    if min_component_area is None:
        total_pixels = mask.shape[0] * mask.shape[1]
        min_component_area = max(30, int(total_pixels * 0.000005))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    cleaned = np.zeros_like(closed)
    for i in range(1, num_labels):  # label 0 is background
        if stats[i, cv2.CC_STAT_AREA] >= min_component_area:
            cleaned[labels == i] = 255

    # 5. Light Gaussian blur + re-threshold to smooth jagged edges
    if smooth:
        blurred = cv2.GaussianBlur(cleaned, (3, 3), 0.8)
        _, cleaned = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)

    return cleaned
