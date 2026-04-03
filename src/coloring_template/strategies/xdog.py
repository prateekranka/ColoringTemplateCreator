"""XDoG (Extended Difference of Gaussians) strategy.

Developed by Winnemöller et al. (Computers & Graphics, 2012), XDoG extends the
standard Difference of Gaussians with soft thresholding to produce clean
black-and-white line art that looks hand-drawn.  With high φ values the output
closely resembles ink-on-paper illustration outlines — ideal for coloring books.

Parameters:
  σ  (sigma)     - base Gaussian scale (controls detail level)
  k              - ratio between the two Gaussian scales (σ₂ = k·σ)
  p              - sharpness multiplier for the DoG
  φ  (phi)       - threshold steepness (higher = crisper black/white)
  ε  (epsilon)   - threshold level (controls how much is kept as line)
"""

import cv2
import numpy as np

from .base import BaseStrategy


class XDoGStrategy(BaseStrategy):
    """Produces clean line art via Extended Difference of Gaussians.

    Best for:
      - Any image where you want hand-drawn-looking outlines
      - Images with subtle edges that Canny misses
      - Producing variable line weight (thicker at stronger edges)
    """

    def __init__(
        self,
        sigma: float = 0.5,
        k: float = 1.6,
        p: float = 20.0,
        phi: float = 10.0,
        epsilon: float = 0.01,
    ):
        """
        Args:
            sigma: Base Gaussian blur scale. Smaller = finer detail. 0.3-1.0 typical.
            k: Scale ratio between the two Gaussians. Standard DoG uses ~1.6.
            p: Sharpness multiplier. Higher values emphasize edges more strongly.
            phi: Threshold steepness. Higher = harder black/white transition.
                 Values 1-5 give soft pencil look, 10+ gives ink look.
            epsilon: Threshold level. Lower = more lines kept. Range ~-0.5 to 0.5.
        """
        self.sigma = sigma
        self.k = k
        self.p = p
        self.phi = phi
        self.epsilon = epsilon

    def extract(self, img_rgb: np.ndarray) -> np.ndarray:
        # Convert to grayscale float [0, 1]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0

        # Compute kernel sizes (must be odd and positive)
        ksize1 = self._sigma_to_ksize(self.sigma)
        ksize2 = self._sigma_to_ksize(self.sigma * self.k)

        # Two Gaussian blurs at different scales
        g1 = cv2.GaussianBlur(gray, (ksize1, ksize1), self.sigma)
        g2 = cv2.GaussianBlur(gray, (ksize2, ksize2), self.sigma * self.k)

        # Extended Difference of Gaussians
        # D(x) = G_σ(x) - τ · G_{kσ}(x)  where τ = 1 + p
        dog = g1 - (1.0 + self.p) * g2 + self.p * g1
        # Simplifies to: (1+p)*g1 - (1+p)*g2 = (1+p)*(g1 - g2)
        # But the standard XDoG formulation is:
        # D(x) = (1+p) * G_σ - p * G_{kσ}
        dog = (1.0 + self.p) * g1 - self.p * g2

        # Soft thresholding with tanh
        # T(x) = 1               if D(x) >= ε
        #       = 1 + tanh(φ·(D(x) - ε))   otherwise
        result = np.where(
            dog >= self.epsilon,
            1.0,
            1.0 + np.tanh(self.phi * (dog - self.epsilon)),
        )

        # Invert (we want lines=255, background=0) and convert to uint8
        # result is ~1.0 for background, ~0.0 for edges
        mask = ((1.0 - result) * 255).clip(0, 255).astype(np.uint8)

        # Binary threshold to get clean black/white
        _, mask = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)

        return mask

    @staticmethod
    def _sigma_to_ksize(sigma: float) -> int:
        """Convert sigma to an appropriate odd kernel size."""
        ksize = int(np.ceil(sigma * 6)) | 1  # ensure odd
        return max(ksize, 3)
