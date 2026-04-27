"""Combined strategy - merges dark pixel and edge detection results."""

import cv2
import numpy as np

from .base import BaseStrategy
from .dark_pixel import DarkPixelStrategy
from .edge_detect import EdgeDetectStrategy


class CombinedStrategy(BaseStrategy):
    """Runs both DarkPixel and EdgeDetect strategies and merges their masks with OR.

    Useful as an 'auto' fallback when there are dark outlines present but also
    some colored/lighter lines that pure dark-pixel extraction would miss.
    The union of both masks is then passed through a heavier cleanup step.
    """

    def __init__(self, threshold: int = 60):
        """
        Args:
            threshold: Dark pixel threshold passed to DarkPixelStrategy.
        """
        self._dark = DarkPixelStrategy(threshold=threshold)
        self._edge = EdgeDetectStrategy(method="adaptive")

    def extract(self, img_rgb: np.ndarray) -> np.ndarray:
        mask_dark = self._dark.extract(img_rgb).astype(np.float32) / 255.0
        mask_edge = self._edge.extract(img_rgb).astype(np.float32) / 255.0
        blended = np.clip(mask_dark * 0.7 + mask_edge * 0.3, 0.0, 1.0)
        _, combined = cv2.threshold((blended * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        return combined
