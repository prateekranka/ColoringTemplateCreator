# Experiments

This directory is the autoresearch logbook. Every time the tool runs in
auto-tune mode (the default), it writes three artefacts here so parameter
exploration is reproducible and reviewable in code review.

## Files

### `<image-stem>_<timestamp>.jsonl`
One line per candidate parameter set evaluated in the run. Each line is a JSON
object:

```json
{
  "image": "/abs/path/to/input.png",
  "candidate_index": 7,
  "params": {
    "threshold": 60,
    "saturation_cutoff": 80,
    "close_kernel_size": 5,
    "min_component_area": null,
    "max_fill_ratio": 0.02,
    "line_thickness": 0
  },
  "score": {
    "total": 0.8421,
    "band": 1.0,
    "fill": 0.76,
    "closed": 0.71,
    "noise": 0.94,
    "crisp": 0.88,
    "pure": 1.0
  },
  "elapsed_ms": 812.4,
  "seed": 1749834021,
  "budget": 20,
  "git_sha": "abc123..."
}
```

### `<image-stem>_winner.json`
The single winning candidate for that image — latest run overwrites. Useful
for diffing winning params as the scoring rubric evolves.

### `runs.csv`
Rolling master log, one row per run. Load in pandas for analysis across many
images and many scoring-weight tweaks:

```python
import pandas as pd
df = pd.read_csv("experiments/runs.csv")
df["params"] = df["winner_params_json"].apply(json.loads)
```

## Controls

- `--experiments-dir PATH` — relocate this directory.
- `--no-log` — skip writing entirely (for CI / benchmarking).
- `--search-budget N` — number of candidates per run (default 20).

## Scoring rubric

Defined in `src/coloring_template/autotune/metrics.py`:

| Metric | Weight | What it measures |
|---|---|---|
| `band`   | 0.25 | Edge-pixel ratio inside target band [0.02, 0.08] |
| `fill`   | 0.25 | Skeleton-to-line ratio (catches solid blobs) |
| `closed` | 0.20 | Fraction of bg regions fully enclosed (flood-fill friendly) |
| `noise`  | 0.15 | Absence of speckle (components < 20 px) |
| `crisp`  | 0.10 | Skeleton endpoint ratio (lower = cleaner lines) |
| `pure`   | 0.05 | Binary purity — no anti-alias grays |

Score in `[0, 1]`; `total >= 0.92` triggers early-stop in sequential search.
