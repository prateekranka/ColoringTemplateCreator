"""Abstract base class for line extraction strategies."""

from abc import ABC, abstractmethod
import numpy as np


class BaseStrategy(ABC):
    """Base class for all line extraction strategies.

    Each strategy takes an RGB image and returns a binary mask where
    255 = outline pixel, 0 = background.
    """

    @abstractmethod
    def extract(self, img_rgb: np.ndarray) -> np.ndarray:
        """Extract line art from an RGB image.

        Args:
            img_rgb: Input image as an HxWx3 numpy array in RGB format.

        Returns:
            Binary mask as an HxW numpy array (uint8), where 255 = outline.
        """
        ...
