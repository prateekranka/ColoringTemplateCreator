"""Dark pixel extraction strategy - primary strategy for pop art with bold black outlines."""

import cv2
import numpy as np

from .base import BaseStrategy


class DarkPixelStrategy(BaseStrategy):
    """Extracts line art by identifying dark, desaturated pixels (true black outlines).

    Pop art typically has explicit bold black outlines already drawn. This strategy
    directly captures those outline pixels using a luminance threshold combined with
    an HSV saturation gate that distinguishes black outlines (low saturation, low value)
    from dark-colored fills like deep navy or dark purple (high saturation, low value).
    """

    def __init__(self, threshold: int = 80, saturation_gate: bool = True,
                 saturation_threshold: int = 100, very_dark_divisor: int = 2):
        self.threshold = threshold
        self.saturation_gate = saturation_gate
        self.saturation_threshold = saturation_threshold
        self.very_dark_divisor = very_dark_divisor

    def extract(self, img_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        if not self.saturation_gate:
            _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY_INV)
            return mask

        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]

        dark_and_desaturated = (gray < self.threshold) & (saturation < self.saturation_threshold)
        very_dark = gray < (self.threshold // self.very_dark_divisor)

        mask = np.where(dark_and_desaturated | very_dark, 255, 0).astype(np.uint8)
        return mask
