#!/usr/bin/env python3
"""Trial runner for the generator autoresearch loop.

Randomly samples 4 subjects, generates SVG coloring templates via Claude,
scores them against gold standards, and prints:

    score: X.XX

to stdout. Per-image details go to stderr.

Usage:
    python experiments/generator/gen_trial.py
    python experiments/generator/gen_trial.py --seed 123
"""

import argparse
import random
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

import generate  # noqa: E402 (sibling in generator/)
import judge     # noqa: E402 (parent experiments/)
from subjects import SUBJECTS  # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "output"
GOLD_DIR   = ROOT / "experiments" / "gold_standard"
N_SAMPLES  = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one generator trial")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for subject sampling (default: 42)")
    return p.parse_args()


def subject_to_slug(subject: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", subject.lower()).strip("_")[:50]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    selected = random.sample(SUBJECTS, min(N_SAMPLES, len(SUBJECTS)))

    print(f"Generator trial  seed={args.seed}", file=sys.stderr)
    print(f"Subjects: {selected}", file=sys.stderr)
    print(file=sys.stderr)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    failed: list[str] = []

    for subject in selected:
        slug = subject_to_slug(subject)
        out_path = OUTPUT_DIR / f"{slug}.png"
        try:
            generate.generate_template(subject, out_path)
            output_paths.append(out_path)
            print(f"  OK   {subject} -> {out_path.name}", file=sys.stderr)
        except Exception as exc:
            failed.append(subject)
            print(f"  FAIL {subject}: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    if not output_paths:
        print("ERROR: all subjects failed to generate", file=sys.stderr)
        sys.exit(1)

    print(file=sys.stderr)
    print("Calling LLM judge...", file=sys.stderr)

    try:
        per_image_scores, mean_score = judge.score_images(output_paths, GOLD_DIR)
    except Exception as exc:
        print(f"ERROR: judge failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    print(file=sys.stderr)
    print(f"{'Subject':<45} {'Score':>6}", file=sys.stderr)
    print("-" * 53, file=sys.stderr)
    for path, sc in zip(output_paths, per_image_scores):
        print(f"  {path.stem:<43} {sc:>5.1f}", file=sys.stderr)
    for subj in failed:
        print(f"  {subject_to_slug(subj):<43} {'FAIL':>5}", file=sys.stderr)
    print("-" * 53, file=sys.stderr)
    print(f"  {'MEAN':<43} {mean_score:>5.2f}", file=sys.stderr)

    # Only stdout line — parsed by gen_loop.py
    print(f"score: {mean_score:.2f}")


if __name__ == "__main__":
    main()
