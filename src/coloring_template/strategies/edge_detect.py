"""Edge detection strategy - fallback for images without clear black outlines."""

import cv2
import numpy as np

from .base import BaseStrategy


class EdgeDetectStrategy(BaseStrategy):
    """Extracts line art via edge detection.

    Used when outlines are not pure black (e.g. colored outlines, light backgrounds,
    or illustrations drawn on colored paper). Two methods are available:

    - 'adaptive': Adaptive thresholding - handles varying lighting/contrast and
                  naturally captures the full width of existing outlines.
    - 'canny': Canny edge detection with bilateral pre-filter.
    """

    def __init__(self, method: str = "adaptive", canny_low: int = 30, canny_high: int = 100,
                 bilateral_d: int = 9, bilateral_sigmaColor: int = 75,
                 bilateral_sigmaSpace: int = 75, bilateral_passes: int = 4,
                 adaptive_blockSize: int = 41, adaptive_C: int = 6,
                 lab_d: int = 9, lab_sigmaColor: int = 75, lab_sigmaSpace: int = 75,
                 lab_passes: int = 2, lab_init_thresh: int = 60,
                 lab_high_thresh: int = 120, lab_edge_ratio: float = 0.10,
                 lab_fallback_ratio: float = 0.15):
        if method not in ("adaptive", "canny"):
            raise ValueError(f"method must be 'adaptive' or 'canny', got {method!r}")
        self.method = method
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.bilateral_d = bilateral_d
        self.bilateral_sigmaColor = bilateral_sigmaColor
        self.bilateral_sigmaSpace = bilateral_sigmaSpace
        self.bilateral_passes = bilateral_passes
        self.adaptive_blockSize = adaptive_blockSize
        self.adaptive_C = adaptive_C
        self.lab_d = lab_d
        self.lab_sigmaColor = lab_sigmaColor
        self.lab_sigmaSpace = lab_sigmaSpace
        self.lab_passes = lab_passes
        self.lab_init_thresh = lab_init_thresh
        self.lab_high_thresh = lab_high_thresh
        self.lab_edge_ratio = lab_edge_ratio
        self.lab_fallback_ratio = lab_fallback_ratio

    def extract(self, img_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        smoothed = gray
        for _ in range(self.bilateral_passes):
            smoothed = cv2.bilateralFilter(
                smoothed, d=self.bilateral_d,
                sigmaColor=self.bilateral_sigmaColor,
                sigmaSpace=self.bilateral_sigmaSpace
            )

        if self.method == "adaptive":
            grayscale_mask = self._adaptive(smoothed)
        else:
            grayscale_mask = self._canny(smoothed)

        color_edges = self._lab_color_edges(img_rgb)
        if color_edges is not None:
            grayscale_mask = cv2.bitwise_or(grayscale_mask, color_edges)

        return grayscale_mask

    def _lab_color_edges(self, img_rgb: np.ndarray) -> np.ndarray | None:
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

        lab_smoothed = np.zeros_like(img_lab)
        for i in range(3):
            ch = img_lab[:, :, i]
            for _ in range(self.lab_passes):
                ch = cv2.bilateralFilter(
                    ch, d=self.lab_d,
                    sigmaColor=self.lab_sigmaColor,
                    sigmaSpace=self.lab_sigmaSpace
                )
            lab_smoothed[:, :, i] = ch

        a_channel = lab_smoothed[:, :, 1].astype(np.float32)
        b_channel = lab_smoothed[:, :, 2].astype(np.float32)

        a_grad_x = cv2.Scharr(a_channel, cv2.CV_32F, 1, 0)
        a_grad_y = cv2.Scharr(a_channel, cv2.CV_32F, 0, 1)
        a_mag = np.sqrt(a_grad_x**2 + a_grad_y**2)

        b_grad_x = cv2.Scharr(b_channel, cv2.CV_32F, 1, 0)
        b_grad_y = cv2.Scharr(b_channel, cv2.CV_32F, 0, 1)
        b_mag = np.sqrt(b_grad_x**2 + b_grad_y**2)

        color_mag = np.maximum(a_mag, b_mag)

        max_val = color_mag.max()
        if max_val < 1.0:
            return None
        color_mag = (color_mag / max_val * 255).astype(np.uint8)

        _, color_edges = cv2.threshold(color_mag, self.lab_init_thresh, 255, cv2.THRESH_BINARY)

        edge_ratio = np.sum(color_edges > 0) / color_edges.size
        if edge_ratio > self.lab_edge_ratio:
            _, color_edges = cv2.threshold(color_mag, self.lab_high_thresh, 255, cv2.THRESH_BINARY)
            edge_ratio = np.sum(color_edges > 0) / color_edges.size
            if edge_ratio > self.lab_fallback_ratio:
                return None

        return color_edges

    def _adaptive(self, gray: np.ndarray) -> np.ndarray:
        mask = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=self.adaptive_blockSize,
            C=self.adaptive_C,
        )
        return mask

    def _canny(self, gray: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        kernel = np.ones((2, 2), np.uint8)
        thick = cv2.dilate(edges, kernel, iterations=1)
        return thick
