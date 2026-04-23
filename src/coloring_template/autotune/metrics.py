"""Quality metrics for scoring coloring-template candidates.

Operates on a binary mask (uint8, 255 = line, 0 = background) and produces a
`Score` with six sub-metrics plus a weighted total. Metrics capture the visual
qualities of gold-standard coloring-book line art:

- Detail density in a target band (not too sparse, not too dense)
- Lines are actually thin outlines (not solid fills)
- Regions are closed enough to flood-fill
- No speckle / noise blobs
- Lines are crisp (few stray skeleton endpoints)
- Output is pure binary (no anti-aliased grays)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np
from skimage.morphology import skeletonize


# Weights sum to 1.0. Band and fill dominate because those are the two most
# visible failure modes (wireframe / solid blob).
WEIGHTS = {
    "band": 0.25,
    "fill": 0.25,
    "closed": 0.20,
    "noise": 0.15,
    "crisp": 0.10,
    "pure": 0.05,
}

# Detail-density target: final mask should have this fraction of pixels "on".
BAND_LOW = 0.02
BAND_HIGH = 0.08
BAND_FALLOFF = 0.02  # triangular falloff outside the band

# Fill-penalty calibration (skeleton:line-pixel ratio).
FILL_LOW = 0.15   # below this = fat solid region
FILL_HIGH = 0.35  # above this = already thin outline

# Noise detection: components with fewer than this many pixels are speckle.
NOISE_MIN_AREA = 20

# Crispness: fraction of skeleton pixels that are endpoints; lower is better.
CRISP_MAX_ENDPOINT_RATIO = 0.05

# If input is large, downsample for scoring only (final render is full-res).
SCORE_MAX_SIDE = 1500


@dataclass(frozen=True)
class Score:
    total: float
    band: float
    fill: float
    closed: float
    noise: float
    crisp: float
    pure: float

    def to_dict(self) -> dict:
        return asdict(self)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _maybe_downsample(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    longest = max(h, w)
    if longest <= SCORE_MAX_SIDE:
        return mask
    scale = SCORE_MAX_SIDE / longest
    new_w, new_h = int(w * scale), int(h * scale)
    # INTER_NEAREST preserves binary character so skeletonize behaves sanely.
    small = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return small


def _band_score(edge_ratio: float) -> float:
    if BAND_LOW <= edge_ratio <= BAND_HIGH:
        return 1.0
    if edge_ratio < BAND_LOW:
        return _clamp(1.0 - (BAND_LOW - edge_ratio) / BAND_FALLOFF)
    return _clamp(1.0 - (edge_ratio - BAND_HIGH) / BAND_FALLOFF)


def _fill_score(mask_bool: np.ndarray) -> tuple[float, np.ndarray]:
    """Return fill score plus the skeleton (reused downstream for crispness)."""
    line_pixels = int(mask_bool.sum())
    if line_pixels == 0:
        # Empty mask → treat as maximally bad fill signal.
        return 0.0, np.zeros_like(mask_bool, dtype=bool)
    skel = skeletonize(mask_bool)
    skel_pixels = int(skel.sum())
    ratio = skel_pixels / line_pixels
    score = _clamp((ratio - FILL_LOW) / (FILL_HIGH - FILL_LOW))
    return score, skel


def _closed_score(mask: np.ndarray) -> float:
    """Fraction of background regions that are fully enclosed by line art."""
    inv = cv2.bitwise_not(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    if num_labels <= 1:
        return 0.0
    h, w = mask.shape
    closed = 0
    total = 0
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        total += 1
        touches_border = x == 0 or y == 0 or (x + cw) >= w or (y + ch) >= h
        if not touches_border:
            closed += 1
    if total == 0:
        return 0.0
    return closed / total


def _noise_score(mask: np.ndarray) -> float:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return 1.0  # nothing to be noisy about
    total = num_labels - 1
    small = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < NOISE_MIN_AREA:
            small += 1
    return 1.0 - small / total


# Kernel used to count 8-neighbours in the skeleton.
_NEIGHBOR_KERNEL = np.array(
    [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
    dtype=np.uint8,
)


def _crisp_score(skel: np.ndarray) -> float:
    skel_pixels = int(skel.sum())
    if skel_pixels == 0:
        return 0.0
    skel_u8 = skel.astype(np.uint8)
    neighbours = cv2.filter2D(skel_u8, ddepth=cv2.CV_8U, kernel=_NEIGHBOR_KERNEL)
    endpoints = int(((skel_u8 == 1) & (neighbours == 1)).sum())
    endpoint_ratio = endpoints / skel_pixels
    return 1.0 - _clamp(endpoint_ratio / CRISP_MAX_ENDPOINT_RATIO)


def _pure_score(mask: np.ndarray) -> float:
    # Pipeline re-thresholds so this should always be ~1.0; still worth checking.
    unique_vals = np.unique(mask)
    if set(unique_vals.tolist()) <= {0, 255}:
        return 1.0
    # Fraction of pixels that are strictly 0 or 255.
    pure = int(((mask == 0) | (mask == 255)).sum())
    return pure / mask.size


def score_mask(mask: np.ndarray) -> Score:
    """Score a binary mask against the gold-standard coloring-template rubric.

    Args:
        mask: Binary mask (uint8), 255 = line, 0 = background. 2-D array.

    Returns:
        A Score with the weighted total and every sub-metric.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")

    small = _maybe_downsample(mask)
    mask_bool = small >= 128

    edge_ratio = float(mask_bool.sum()) / mask_bool.size
    s_band = _band_score(edge_ratio)
    s_fill, skel = _fill_score(mask_bool)
    s_closed = _closed_score(small)
    s_noise = _noise_score(small)
    s_crisp = _crisp_score(skel)
    s_pure = _pure_score(mask)

    total = (
        WEIGHTS["band"] * s_band
        + WEIGHTS["fill"] * s_fill
        + WEIGHTS["closed"] * s_closed
        + WEIGHTS["noise"] * s_noise
        + WEIGHTS["crisp"] * s_crisp
        + WEIGHTS["pure"] * s_pure
    )

    return Score(
        total=float(total),
        band=float(s_band),
        fill=float(s_fill),
        closed=float(s_closed),
        noise=float(s_noise),
        crisp=float(s_crisp),
        pure=float(s_pure),
    )
