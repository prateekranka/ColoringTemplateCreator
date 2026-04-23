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

    def __init__(
        self,
        threshold: int = 60,
        saturation_gate: bool = True,
        saturation_cutoff: int = 80,
    ):
        """
        Args:
            threshold: Grayscale threshold 0-255. Pixels below this are considered
                       outline candidates. Default 60 works well for bold black outlines.
                       Lower values = stricter (only boldest lines); higher = more inclusive.
            saturation_gate: If True, uses HSV saturation to reject dark-but-colored pixels
                             so dark blue/purple fills don't become part of the outline.
            saturation_cutoff: HSV saturation threshold (0-255) applied when
                               saturation_gate is True. Pixels must have saturation
                               below this to count as outline. Lower = stricter
                               rejection of colored pixels.
        """
        self.threshold = threshold
        self.saturation_gate = saturation_gate
        self.saturation_cutoff = saturation_cutoff

    def extract(self, img_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        if not self.saturation_gate:
            _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY_INV)
            return mask

        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]

        # True black outlines: low brightness AND low saturation
        dark_and_desaturated = (gray < self.threshold) & (saturation < self.saturation_cutoff)

        # Very dark pixels are outline regardless of saturation (e.g. near-pure-black)
        very_dark = gray < (self.threshold // 2)

        mask = np.where(dark_and_desaturated | very_dark, 255, 0).astype(np.uint8)
        return mask
