"""Edge detection strategy - fallback for images without clear black outlines."""

import cv2
import numpy as np

from .base import BaseStrategy


class EdgeDetectStrategy(BaseStrategy):
    """Extracts line art via edge detection.

    Used when outlines are not pure black (e.g. colored outlines, light backgrounds,
    or illustrations drawn on colored paper). Two methods are available:

    - 'adaptive': Adaptive thresholding - handles varying lighting/contrast and
                  naturally captures the full width of existing outlines. Best for
                  most illustration-style images.
    - 'canny': Canny edge detection with bilateral pre-filter - better at finding
               faint edges but produces thinner single-pixel lines. Good for photos.
    """

    def __init__(self, method: str = "adaptive", canny_low: int = 30, canny_high: int = 100):
        """
        Args:
            method: 'adaptive' or 'canny'.
            canny_low: Lower threshold for Canny hysteresis (used if method='canny').
            canny_high: Upper threshold for Canny hysteresis (used if method='canny').
        """
        if method not in ("adaptive", "canny"):
            raise ValueError(f"method must be 'adaptive' or 'canny', got {method!r}")
        self.method = method
        self.canny_low = canny_low
        self.canny_high = canny_high

    def extract(self, img_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Heavy bilateral filtering to smooth out brush texture / halftone
        # while preserving the major color region boundaries.
        # Apply twice for stronger smoothing on painterly images.
        smoothed = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        smoothed = cv2.bilateralFilter(smoothed, d=9, sigmaColor=75, sigmaSpace=75)

        if self.method == "adaptive":
            return self._adaptive(smoothed)
        return self._canny(smoothed)

    def _adaptive(self, gray: np.ndarray) -> np.ndarray:
        # blockSize=21 (larger than before) + C=5 to focus on strong boundaries
        # and ignore subtle texture. Larger block = less sensitive to local noise.
        mask = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21,
            C=5,
        )
        return mask

    def _canny(self, gray: np.ndarray) -> np.ndarray:
        # Higher Canny thresholds to only detect strong edges (major shape boundaries)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        # Dilate to thicken single-pixel Canny edges to a more paintable width
        kernel = np.ones((2, 2), np.uint8)
        thick = cv2.dilate(edges, kernel, iterations=1)
        return thick
