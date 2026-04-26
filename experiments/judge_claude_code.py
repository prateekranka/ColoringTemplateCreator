"""LLM visual judge for coloring template quality — Claude Code CLI edition.

Drop-in replacement for judge.py that uses the `claude` CLI subprocess instead
of the Anthropic API. Works without ANTHROPIC_API_KEY; uses the Claude.ai
subscription authenticated in your Claude Code CLI installation.

Public API (identical to judge.py):
    score_images(output_paths, gold_dir) -> (list[float], float)
"""

import re
import subprocess
import sys
from pathlib import Path

MAX_GOLD_IMAGES = 4

JUDGE_PROMPT_TEMPLATE = """\
You are an expert judge evaluating coloring book page quality.

First, read these gold-standard reference images to understand the target quality:
{gold_paths}

Then read this candidate coloring template:
{output_path}

Score the CANDIDATE from 0 to 10 using this rubric:
  10 — Perfect. Indistinguishable from the references. Clean 1-2px-wide black
       outlines, all major regions fully enclosed (flood-fill without leaking),
       virtually no noise blobs, outline density roughly 2-8%.
  8-9 — Professional quality. Minor imperfections: a few small gaps or noise
        blobs, but overall structure is clean and every main region is bounded.
  6-7 — Acceptable. Recognizable as a coloring template but with notable flaws:
        some regions not fully closed, moderate noise, or lines too thick/thin.
  4-5 — Poor. Outline structure present but significantly degraded: large gaps,
        heavy texture noise, solid fill areas, or density far outside range.
  0-3 — Broken. Mostly black, mostly white, unrecognizable, or extreme noise.

Reply with a single integer 0-10 and NOTHING ELSE."""


def _load_gold_images(gold_dir: Path) -> list[Path]:
    if not gold_dir.is_dir():
        return []
    candidates: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        candidates.extend(gold_dir.glob(ext))
    candidates = [p for p in candidates if p.is_file()]
    candidates.sort()
    return candidates[:MAX_GOLD_IMAGES]


def _check_claude_cli() -> None:
    result = subprocess.run(["claude", "--version"], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "`claude` CLI not found. Install Claude Code: https://claude.ai/code"
        )


def _parse_score(text: str) -> float:
    match = re.search(r'\b(10|[0-9])\b', text.strip())
    return float(int(match.group(1))) if match else 0.0


def _score_one(output_path: Path, gold_images: list[Path]) -> float:
    if gold_images:
        gold_paths_str = "\n".join(f"  {p}" for p in gold_images)
    else:
        gold_paths_str = "  (none available — judge on general quality criteria)"

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        gold_paths=gold_paths_str,
        output_path=output_path,
    )

    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--allowedTools", "Read", "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=90,
        )
        raw = result.stdout.strip()
        score = _parse_score(raw)
        print(
            f"  judge_cc({output_path.name}): {raw!r} -> {score:.1f}",
            file=sys.stderr,
        )
        return score
    except subprocess.TimeoutExpired:
        print(f"  judge_cc timeout for {output_path.name}", file=sys.stderr)
        return 0.0
    except Exception as exc:
        print(f"  judge_cc error for {output_path.name}: {exc}", file=sys.stderr)
        return 0.0


def score_images(
    output_paths: list[Path],
    gold_dir: Path,
) -> tuple[list[float], float]:
    """Score pipeline output images against gold-standard exemplars.

    Uses `claude` CLI subprocess — no ANTHROPIC_API_KEY required.

    Args:
        output_paths: Paths to coloring template PNGs to evaluate.
        gold_dir: Directory containing gold-standard reference PNGs.

    Returns:
        (per_image_scores, mean_score) — scores are 0.0-10.0.
    """
    if not output_paths:
        return [], 0.0

    _check_claude_cli()
    gold_images = _load_gold_images(gold_dir)

    if not gold_images:
        print(
            f"WARNING: no gold-standard images found in {gold_dir}. "
            "Scoring without exemplars.",
            file=sys.stderr,
        )

    per_image_scores = [_score_one(p, gold_images) for p in output_paths]
    mean_score = sum(per_image_scores) / len(per_image_scores)
    return per_image_scores, mean_score
