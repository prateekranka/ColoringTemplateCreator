"""Morphological cleanup for line art masks.

Cleans up raw extraction masks by:
  1. Closing small gaps in outlines (morphological CLOSE)
  2. Removing isolated noise blobs below a minimum area
  3. Smoothing jagged edges with a light Gaussian pass + re-threshold
"""

import cv2
import numpy as np


def clean(
    mask: np.ndarray,
    close_kernel_size: int = 3,
    min_component_area: int | None = None,
    smooth: bool = True,
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

    Returns:
        Cleaned binary mask (uint8), same shape as input.
    """
    # 1. Morphological CLOSE: seal small gaps in outlines
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)
    )
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 2. Remove noise blobs smaller than min_component_area
    if min_component_area is None:
        total_pixels = mask.shape[0] * mask.shape[1]
        min_component_area = max(30, int(total_pixels * 0.000005))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    cleaned = np.zeros_like(closed)
    for i in range(1, num_labels):  # label 0 is background
        if stats[i, cv2.CC_STAT_AREA] >= min_component_area:
            cleaned[labels == i] = 255

    # 3. Light Gaussian blur + re-threshold to smooth jagged edges
    if smooth:
        blurred = cv2.GaussianBlur(cleaned, (3, 3), 0.8)
        _, cleaned = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)

    return cleaned
