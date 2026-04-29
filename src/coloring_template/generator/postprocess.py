"""Raster cleanup for image-model coloring templates."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..output import save, save_svg


def raster_to_mask(
    input_path: str | Path,
    *,
    threshold: int = 210,
    min_component_area: int | None = None,
) -> np.ndarray:
    """Convert generated raster line art into a binary outline mask.

    The returned mask follows the package convention: 255 means outline pixel,
    0 means background.
    """
    input_path = Path(input_path)
    gray = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    if min_component_area is None:
        total = mask.shape[0] * mask.shape[1]
        min_component_area = max(25, int(total * 0.00002))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_component_area:
            cleaned[labels == i] = 255

    return cleaned


def clean_generated_raster(
    input_path: str | Path,
    output_path: str | Path,
    *,
    threshold: int = 210,
    transparent: bool = False,
    min_size: int = 3000,
    dpi: int = 300,
    svg: bool = False,
) -> Path:
    """Save a generated raster as a pure black/white coloring template."""
    output_path = Path(output_path)
    mask = raster_to_mask(input_path, threshold=threshold)
    result = save(mask, output_path, dpi=dpi, transparent=transparent, min_size=min_size)
    if svg:
        save_svg(mask, output_path.with_suffix(".svg"), min_size=min_size)
    return result
