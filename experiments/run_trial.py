#!/usr/bin/env python3
"""Trial runner for the autoresearch loop.

Processes all example images through the full coloring-template pipeline,
calls the LLM judge, and prints the mean score to stdout as:

    score: X.XX

A per-image table is printed to stderr.

Usage:
    python experiments/run_trial.py
    python experiments/run_trial.py --strategy dark --threshold 50
"""

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# experiments/ is already on sys.path (Python adds the script's dir automatically)
from coloring_template.pipeline import convert  # noqa: E402
import judge                                    # noqa: E402

EXAMPLES_DIR  = ROOT / "examples"
TRIAL_OUT_DIR = ROOT / "experiments" / "trial_output"
GOLD_DIR      = ROOT / "experiments" / "gold_standard"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one autoresearch trial")
    p.add_argument("--strategy", default="auto",
                   choices=["auto", "dark", "edge", "canny", "combined"])
    p.add_argument("--threshold", type=int, default=60)
    return p.parse_args()


def collect_images() -> list[Path]:
    images: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(EXAMPLES_DIR.glob(ext))
    images = [p for p in images if not p.name.startswith("create_")]
    return sorted(images)


def run_pipeline(image_path: Path, strategy: str, threshold: int) -> Path:
    TRIAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRIAL_OUT_DIR / f"{image_path.stem}_output.png"
    convert(
        input_path=image_path,
        output_path=out_path,
        strategy=strategy,
        threshold=threshold,
    )
    return out_path


def main() -> None:
    args = parse_args()
    images = collect_images()

    if not images:
        print("ERROR: no images found in examples/", file=sys.stderr)
        sys.exit(1)

    print(f"Trial: strategy={args.strategy}  threshold={args.threshold}", file=sys.stderr)
    print(f"Images: {[p.name for p in images]}", file=sys.stderr)
    print(file=sys.stderr)

    output_paths: list[Path] = []
    failed: list[str] = []

    for img in images:
        try:
            out = run_pipeline(img, args.strategy, args.threshold)
            output_paths.append(out)
            print(f"  OK   {img.name} -> {out.name}", file=sys.stderr)
        except Exception as exc:
            failed.append(img.name)
            print(f"  FAIL {img.name}: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    if not output_paths:
        print("ERROR: all images failed", file=sys.stderr)
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
    print(f"{'Image':<40} {'Score':>6}", file=sys.stderr)
    print("-" * 48, file=sys.stderr)
    for path, sc in zip(output_paths, per_image_scores):
        print(f"  {path.stem:<38} {sc:>5.1f}", file=sys.stderr)
    for name in failed:
        print(f"  {name:<38} {'FAIL':>5}", file=sys.stderr)
    print("-" * 48, file=sys.stderr)
    print(f"  {'MEAN':<38} {mean_score:>5.2f}", file=sys.stderr)

    # Only stdout line — parsed by loop.py
    print(f"score: {mean_score:.2f}")


if __name__ == "__main__":
    main()
