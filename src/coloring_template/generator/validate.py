"""Validation metrics for generated coloring templates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ValidationResult:
    """Topology and density checks for a rendered coloring template."""

    ok: bool
    feedback: list[str]
    black_density: float
    enclosed_regions: int
    tiny_regions: int
    component_count: int
    content_coverage: float


def _foreground_mask(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read rendered PNG: {image_path}")
    return (img < 180).astype(np.uint8) * 255


def _enclosed_region_stats(mask: np.ndarray) -> tuple[int, int]:
    background = cv2.bitwise_not(mask)
    h, w = background.shape
    flood_canvas = background.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood_canvas, flood_mask, (0, 0), 128)
    enclosed = (flood_canvas == 255).astype(np.uint8) * 255

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(enclosed, connectivity=8)
    region_count = 0
    tiny_count = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 100:
            continue
        region_count += 1
        if area < 900:
            tiny_count += 1
    return region_count, tiny_count


def _content_coverage(mask: np.ndarray) -> float:
    points = cv2.findNonZero(mask)
    if points is None:
        return 0.0
    x, y, w, h = cv2.boundingRect(points)
    total = mask.shape[0] * mask.shape[1]
    return (w * h) / total


def validate_png(
    image_path: str | Path,
    *,
    min_density: float = 0.04,
    max_density: float = 0.12,
    min_regions: int = 25,
    max_tiny_regions: int = 30,
) -> ValidationResult:
    """Validate a rendered PNG as a coloring app template."""
    image_path = Path(image_path)
    mask = _foreground_mask(image_path)
    total = mask.shape[0] * mask.shape[1]
    black_density = cv2.countNonZero(mask) / total

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    component_count = sum(
        1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= 25
    )
    enclosed_regions, tiny_regions = _enclosed_region_stats(mask)
    coverage = _content_coverage(mask)

    feedback = []
    if black_density < min_density:
        feedback.append(
            f"black line density is too low ({black_density:.1%}); add more enclosed detail and stronger strokes"
        )
    if black_density > max_density:
        feedback.append(
            f"black line density is too high ({black_density:.1%}); remove dense marks and avoid black blobs"
        )
    if enclosed_regions < min_regions:
        feedback.append(
            f"only {enclosed_regions} enclosed colorable regions detected; create at least {min_regions} closed regions"
        )
    if tiny_regions > max_tiny_regions:
        feedback.append(
            f"{tiny_regions} tiny regions detected; merge or enlarge small details"
        )
    if component_count > 350:
        feedback.append(
            f"{component_count} separate line components detected; connect related outlines into cleaner paths"
        )
    if coverage < 0.40:
        feedback.append(
            f"content covers only {coverage:.1%} of the page; make the subject and decorations larger"
        )

    return ValidationResult(
        ok=not feedback,
        feedback=feedback,
        black_density=black_density,
        enclosed_regions=enclosed_regions,
        tiny_regions=tiny_regions,
        component_count=component_count,
        content_coverage=coverage,
    )
