"""Experiment logging for autoresearch runs.

Writes three artefacts per run, all under an experiments directory:

  1. `<image-stem>_<timestamp>.jsonl`   — one line per candidate tried
  2. `<image-stem>_winner.json`         — most recent winning params + score
  3. `runs.csv`                          — rolling one-row-per-run summary

These files are committed to the repo so parameter-space exploration is
reviewable in code review and reproducible across runs.
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .search import Candidate, ParamSet
from .metrics import Score


RUNS_CSV_FIELDS = [
    "timestamp",
    "image",
    "budget",
    "seed",
    "winner_score",
    "winner_params_json",
    "git_sha",
]


def _git_sha() -> Optional[str]:
    """Return the current HEAD sha, or None if we aren't in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


class ExperimentLogger:
    """Context manager that logs every candidate from an autoresearch run.

    Usage::

        with ExperimentLogger(image_path, budget, seed, experiments_dir) as log:
            mask, params, score, _ = auto_tune(
                img, budget=budget, seed=seed, on_candidate=log.record,
            )
            log.finalise(params, score)
    """

    def __init__(
        self,
        image_path: Path,
        budget: int,
        seed: int,
        experiments_dir: Path,
    ):
        self.image_path = Path(image_path)
        self.budget = budget
        self.seed = seed
        self.experiments_dir = Path(experiments_dir)
        self.timestamp = time.strftime("%Y%m%dT%H%M%S")
        self.git_sha = _git_sha()
        self._fh = None
        self._closed = False

        stem = self.image_path.stem.replace(" ", "_").replace("(", "").replace(")", "")
        self.jsonl_path = self.experiments_dir / f"{stem}_{self.timestamp}.jsonl"
        self.winner_path = self.experiments_dir / f"{stem}_winner.json"
        self.runs_csv_path = self.experiments_dir / "runs.csv"

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> "ExperimentLogger":
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self._fh = self.jsonl_path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._close()

    def _close(self) -> None:
        if self._closed:
            return
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        self._closed = True

    # -- per-candidate ------------------------------------------------------

    def record(self, candidate: Candidate) -> None:
        if self._fh is None:
            raise RuntimeError("ExperimentLogger not entered as a context manager")
        record = {
            "image": str(self.image_path),
            "candidate_index": candidate.index,
            "params": candidate.params.to_dict(),
            "score": candidate.score.to_dict(),
            "elapsed_ms": candidate.elapsed_ms,
            "seed": self.seed,
            "budget": self.budget,
            "git_sha": self.git_sha,
        }
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    # -- end-of-run ---------------------------------------------------------

    def finalise(self, winner_params: ParamSet, winner_score: Score) -> None:
        """Write winner-summary JSON and append a row to runs.csv."""
        winner_payload = {
            "image": str(self.image_path),
            "timestamp": self.timestamp,
            "seed": self.seed,
            "budget": self.budget,
            "params": winner_params.to_dict(),
            "score": winner_score.to_dict(),
            "git_sha": self.git_sha,
        }
        self.winner_path.write_text(
            json.dumps(winner_payload, indent=2) + "\n", encoding="utf-8"
        )

        row = {
            "timestamp": self.timestamp,
            "image": str(self.image_path),
            "budget": self.budget,
            "seed": self.seed,
            "winner_score": f"{winner_score.total:.4f}",
            "winner_params_json": json.dumps(winner_params.to_dict()),
            "git_sha": self.git_sha or "",
        }
        self._append_runs_csv(row)

    def _append_runs_csv(self, row: dict) -> None:
        new_file = not self.runs_csv_path.exists()
        with self.runs_csv_path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=RUNS_CSV_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
