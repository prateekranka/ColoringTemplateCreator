from .base import BaseStrategy
from .dark_pixel import DarkPixelStrategy
from .edge_detect import EdgeDetectStrategy
from .combined import CombinedStrategy
from .kmeans_segment import KMeansSegmentStrategy
from .xdog import XDoGStrategy

__all__ = [
    "BaseStrategy",
    "DarkPixelStrategy",
    "EdgeDetectStrategy",
    "CombinedStrategy",
    "KMeansSegmentStrategy",
    "XDoGStrategy",
]
