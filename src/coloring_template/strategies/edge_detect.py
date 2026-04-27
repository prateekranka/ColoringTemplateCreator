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
        # Apply four times for stronger smoothing on geometric/painterly images.
        smoothed = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        smoothed = cv2.bilateralFilter(smoothed, d=9, sigmaColor=75, sigmaSpace=75)
        smoothed = cv2.bilateralFilter(smoothed, d=9, sigmaColor=75, sigmaSpace=75)
        smoothed = cv2.bilateralFilter(smoothed, d=9, sigmaColor=75, sigmaSpace=75)

        if self.method == "adaptive":
            grayscale_mask = self._adaptive(smoothed)
        else:
            grayscale_mask = self._canny(smoothed)

        # Detect color boundaries invisible in grayscale using LAB color space
        color_edges = self._lab_color_edges(img_rgb)
        if color_edges is not None:
            grayscale_mask = cv2.bitwise_or(grayscale_mask, color_edges)

        return grayscale_mask

    def _lab_color_edges(self, img_rgb: np.ndarray) -> np.ndarray | None:
        """Detect edges from color boundaries using LAB color space.

        Adjacent regions with different hues but similar luminance are invisible
        in grayscale. LAB A and B channels encode color information independent
        of lightness, so gradients in these channels reveal color boundaries.

        Returns:
            Binary mask of color edges, or None if no significant color edges found.
        """
        # Convert to LAB
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

        # Bilateral filter on each LAB channel to smooth texture while
        # preserving color region boundaries
        lab_smoothed = np.zeros_like(img_lab)
        for i in range(3):
            lab_smoothed[:, :, i] = cv2.bilateralFilter(
                img_lab[:, :, i], d=9, sigmaColor=75, sigmaSpace=75
            )
            lab_smoothed[:, :, i] = cv2.bilateralFilter(
                lab_smoothed[:, :, i], d=9, sigmaColor=75, sigmaSpace=75
            )

        # Compute gradient magnitude on A and B channels (color channels)
        # using Scharr operator for better rotational symmetry
        a_channel = lab_smoothed[:, :, 1].astype(np.float32)
        b_channel = lab_smoothed[:, :, 2].astype(np.float32)

        a_grad_x = cv2.Scharr(a_channel, cv2.CV_32F, 1, 0)
        a_grad_y = cv2.Scharr(a_channel, cv2.CV_32F, 0, 1)
        a_mag = np.sqrt(a_grad_x**2 + a_grad_y**2)

        b_grad_x = cv2.Scharr(b_channel, cv2.CV_32F, 1, 0)
        b_grad_y = cv2.Scharr(b_channel, cv2.CV_32F, 0, 1)
        b_mag = np.sqrt(b_grad_x**2 + b_grad_y**2)

        # Combine: max of A and B gradient magnitudes
        color_mag = np.maximum(a_mag, b_mag)

        # Normalize to 0-255
        max_val = color_mag.max()
        if max_val < 1.0:
            return None  # No significant color variation
        color_mag = (color_mag / max_val * 255).astype(np.uint8)

        # Threshold: only keep strong color boundaries
        # Use a high threshold to avoid picking up subtle gradients
        _, color_edges = cv2.threshold(color_mag, 60, 255, cv2.THRESH_BINARY)

        # Check if the color edges are meaningful (not just noise)
        edge_ratio = np.sum(color_edges > 0) / color_edges.size
        if edge_ratio > 0.10:
            # Too many edges = probably noise/texture, not real boundaries
            # Raise threshold aggressively
            _, color_edges = cv2.threshold(color_mag, 120, 255, cv2.THRESH_BINARY)
            edge_ratio = np.sum(color_edges > 0) / color_edges.size
            if edge_ratio > 0.10:
                return None  # Still too noisy, skip color edges entirely

        return color_edges

    def _adaptive(self, gray: np.ndarray) -> np.ndarray:
        mask = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=41,
            C=6,
        )
        return mask

    def _canny(self, gray: np.ndarray) -> np.ndarray:
        # Higher Canny thresholds to only detect strong edges (major shape boundaries)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        # Dilate to thicken single-pixel Canny edges to a more paintable width
        kernel = np.ones((2, 2), np.uint8)
        thick = cv2.dilate(edges, kernel, iterations=1)
        return thick
