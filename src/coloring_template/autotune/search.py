"""Autoresearch parameter search for coloring-template extraction.

Samples candidate parameter sets, runs each through the DarkPixel + cleanup
pipeline, scores the resulting mask against gold-standard quality metrics,
and returns the best candidate.

Used by `pipeline.convert(..., auto_tune=True)` to self-tune per image.
"""

from __future__ import annotations

import multiprocessing as mp
import random
import time
from dataclasses import dataclass, asdict, field
from typing import Iterable, Optional

import cv2
import numpy as np

from ..cleanup import clean
from ..strategies.dark_pixel import DarkPixelStrategy
from .metrics import Score, score_mask


# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------

THRESHOLD_CHOICES = (45, 60, 75, 90)
SATURATION_CUTOFF_CHOICES = (60, 80, 100)
CLOSE_KERNEL_CHOICES = (3, 5, 7)
# None = let clean() auto-scale from image area.
MIN_COMPONENT_AREA_CHOICES: tuple = (None, 50, 200)
MAX_FILL_RATIO_CHOICES = (0.015, 0.02, 0.03)
LINE_THICKNESS_CHOICES = (0, 1)  # dilate iterations with 3x3 ellipse


@dataclass(frozen=True)
class ParamSet:
    threshold: int
    saturation_cutoff: int
    close_kernel_size: int
    min_component_area: Optional[int]
    max_fill_ratio: float
    line_thickness: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Candidate:
    index: int
    params: ParamSet
    score: Score
    elapsed_ms: float
    mask: np.ndarray = field(repr=False)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

def sample_candidates(n: int, seed: int) -> list[ParamSet]:
    """Stratified random sampling.

    We stratify on threshold so each of the 4 thresholds gets at least n/4
    samples. Within each stratum, other parameters are chosen uniformly at
    random. Deterministic given `seed`.
    """
    if n <= 0:
        raise ValueError("budget must be positive")

    rng = random.Random(seed)
    thresholds = list(THRESHOLD_CHOICES)
    rng.shuffle(thresholds)

    # Distribute n samples across the thresholds as evenly as possible.
    base = n // len(thresholds)
    extra = n % len(thresholds)
    per_threshold = [base + (1 if i < extra else 0) for i in range(len(thresholds))]

    samples: list[ParamSet] = []
    for thr, count in zip(thresholds, per_threshold):
        for _ in range(count):
            samples.append(
                ParamSet(
                    threshold=thr,
                    saturation_cutoff=rng.choice(SATURATION_CUTOFF_CHOICES),
                    close_kernel_size=rng.choice(CLOSE_KERNEL_CHOICES),
                    min_component_area=rng.choice(MIN_COMPONENT_AREA_CHOICES),
                    max_fill_ratio=rng.choice(MAX_FILL_RATIO_CHOICES),
                    line_thickness=rng.choice(LINE_THICKNESS_CHOICES),
                )
            )

    rng.shuffle(samples)
    return samples


# ---------------------------------------------------------------------------
# Candidate runner (module-level for multiprocessing spawn compat)
# ---------------------------------------------------------------------------

# Used by worker processes; populated by `_worker_init`.
_WORKER_IMG: Optional[np.ndarray] = None


def _worker_init(img_rgb: np.ndarray) -> None:
    global _WORKER_IMG
    _WORKER_IMG = img_rgb


def run_candidate(img_rgb: np.ndarray, params: ParamSet) -> np.ndarray:
    """Run the full extract → clean → (optional) dilate pipeline for a candidate.

    Returns the final binary mask (uint8, 255 = line).
    """
    strategy = DarkPixelStrategy(
        threshold=params.threshold,
        saturation_gate=True,
        saturation_cutoff=params.saturation_cutoff,
    )
    raw = strategy.extract(img_rgb)
    cleaned = clean(
        raw,
        close_kernel_size=params.close_kernel_size,
        min_component_area=params.min_component_area,
        smooth=True,
        remove_fills=True,
        max_fill_ratio=params.max_fill_ratio,
    )
    if params.line_thickness > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.dilate(cleaned, kernel, iterations=params.line_thickness)
    return cleaned


def _evaluate(params: ParamSet) -> tuple[ParamSet, Score, float, np.ndarray]:
    """Worker entry point. Uses the image held in the worker global."""
    if _WORKER_IMG is None:
        raise RuntimeError("worker not initialised — _WORKER_IMG is None")
    start = time.perf_counter()
    mask = run_candidate(_WORKER_IMG, params)
    score = score_mask(mask)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return params, score, elapsed_ms, mask


def _evaluate_sequential(img_rgb: np.ndarray, params: ParamSet) -> tuple[ParamSet, Score, float, np.ndarray]:
    start = time.perf_counter()
    mask = run_candidate(img_rgb, params)
    score = score_mask(mask)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return params, score, elapsed_ms, mask


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def auto_tune(
    img_rgb: np.ndarray,
    budget: int = 20,
    workers: Optional[int] = None,
    seed: Optional[int] = None,
    early_stop: float = 0.92,
    on_candidate=None,
) -> tuple[np.ndarray, ParamSet, Score, list[Candidate]]:
    """Search parameter space and return the best candidate's mask.

    Args:
        img_rgb: Input RGB image.
        budget: Number of candidates to try.
        workers: Parallel worker count. None = min(cpu_count(), 4).
                 Set to 1 to force sequential execution (required for early
                 stop to actually save work).
        seed: Random seed for candidate sampling. None = non-deterministic.
        early_stop: If a candidate scores at least this high, stop searching.
                    Only effective when workers == 1.
        on_candidate: Optional callback `(Candidate) -> None` invoked as each
                      candidate finishes. Used by the logger.

    Returns:
        A tuple `(best_mask, best_params, best_score, all_candidates)`.
    """
    if seed is None:
        seed = random.randrange(1 << 30)

    if workers is None:
        workers = min(mp.cpu_count(), 4)
    workers = max(1, workers)

    candidates = sample_candidates(budget, seed)
    results: list[Candidate] = []
    best: Optional[Candidate] = None

    if workers == 1:
        # Sequential — supports early stop.
        for idx, params in enumerate(candidates):
            p, sc, ms, mask = _evaluate_sequential(img_rgb, params)
            cand = Candidate(index=idx, params=p, score=sc, elapsed_ms=ms, mask=mask)
            results.append(cand)
            if on_candidate is not None:
                on_candidate(cand)
            if best is None or cand.score.total > best.score.total:
                best = cand
            if cand.score.total >= early_stop:
                break
    else:
        # Parallel — no mid-run early stop (would require shared state).
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(img_rgb,),
        ) as pool:
            for idx, (p, sc, ms, mask) in enumerate(
                pool.imap_unordered(_evaluate, candidates)
            ):
                cand = Candidate(index=idx, params=p, score=sc, elapsed_ms=ms, mask=mask)
                results.append(cand)
                if on_candidate is not None:
                    on_candidate(cand)
                if best is None or cand.score.total > best.score.total:
                    best = cand

    if best is None:
        raise RuntimeError("auto_tune produced no candidates (budget too small?)")

    return best.mask, best.params, best.score, results
