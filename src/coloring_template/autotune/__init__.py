"""Autoresearch: per-image parameter search for coloring-template extraction."""

from .metrics import Score, score_mask
from .search import Candidate, ParamSet, auto_tune, run_candidate, sample_candidates
from .logbook import ExperimentLogger

__all__ = [
    "Score",
    "score_mask",
    "ParamSet",
    "Candidate",
    "auto_tune",
    "run_candidate",
    "sample_candidates",
    "ExperimentLogger",
]
