"""K-means segmentation strategy - produces guaranteed closed contours.

Based on the insight that segmentation methods are superior to edge detection
for coloring books because they guarantee closed contours defining fillable
regions.  Every pixel belongs to exactly one cluster, so boundaries between
clusters are inherently closed — flood-fill will never leak.

Pipeline:
  1. Bilateral filter to smooth texture/halftone while preserving edges
  2. K-means clustering on pixel colors to reduce to K flat regions
  3. Boundary extraction by marking pixels adjacent to a different cluster
  4. Optional morphological thinning for consistent line weight
"""

import cv2
import numpy as np

from .base import BaseStrategy


class KMeansSegmentStrategy(BaseStrategy):
    """Extracts coloring outlines via K-means color segmentation.

    Best for:
      - Pop art with flat color regions and limited palettes
      - Illustrations where flood-fill must not leak (closed contours guaranteed)
      - Images without clear black outlines (e.g. Matisse-style color blocks)

    The number of clusters K should roughly match the number of distinct colors
    in the image.  For pop art, K=6-10 works well.
    """

    def __init__(self, k: int = 8, blur_strength: int = 9, line_thickness: int = 2):
        """
        Args:
            k: Number of color clusters (default 8). More clusters = more detail
               in the outlines. Fewer clusters = simpler, bolder regions.
            blur_strength: Diameter for bilateral filter preprocessing.
                          Higher = more smoothing of texture/halftone dots.
            line_thickness: Thickness of the extracted boundary lines in pixels.
                           1 = thin crisp lines, 2-3 = bolder coloring-book lines.
        """
        self.k = k
        self.blur_strength = blur_strength
        self.line_thickness = line_thickness

    def extract(self, img_rgb: np.ndarray) -> np.ndarray:
        h, w = img_rgb.shape[:2]

        # --- Step 1: Pre-filter to remove texture/halftone while keeping edges ---
        smoothed = cv2.bilateralFilter(img_rgb, self.blur_strength, 75, 75)
        smoothed = cv2.bilateralFilter(smoothed, self.blur_strength, 75, 75)

        # --- Step 2: K-means clustering in L*a*b* color space ---
        # L*a*b* is perceptually uniform, so K-means distances match human
        # perception of color difference better than RGB.
        lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB)
        pixels = lab.reshape(-1, 3).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, _ = cv2.kmeans(
            pixels, self.k, None, criteria, attempts=3, flags=cv2.KMEANS_PP_CENTERS
        )
        label_map = labels.reshape(h, w)

        # --- Step 3: Extract boundaries between differently-labeled pixels ---
        # A pixel is a boundary pixel if any of its 4-connected neighbors
        # belongs to a different cluster.
        boundary = np.zeros((h, w), dtype=np.uint8)

        # Shift-and-compare for each direction (vectorized, no loops)
        boundary[:-1, :] |= (label_map[:-1, :] != label_map[1:, :]).astype(np.uint8)
        boundary[1:, :]  |= (label_map[1:, :] != label_map[:-1, :]).astype(np.uint8)
        boundary[:, :-1] |= (label_map[:, :-1] != label_map[:, 1:]).astype(np.uint8)
        boundary[:, 1:]  |= (label_map[:, 1:] != label_map[:, :-1]).astype(np.uint8)

        mask = boundary * 255

        # --- Step 4: Thicken lines if requested ---
        if self.line_thickness > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.line_thickness, self.line_thickness),
            )
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask
