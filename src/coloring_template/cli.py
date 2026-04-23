"""Command-line interface for ColoringTemplateCreator.

Usage examples:
    # Auto-tune (default). Runs parameter search, writes PNG + experiment log:
    python -m coloring_template photo.png

    # Auto-tune with a bigger search and both PNG and SVG output:
    python -m coloring_template art.jpg --search-budget 60 --format both

    # Batch with explicit output dir:
    python -m coloring_template *.png -o ./templates/

    # Legacy fixed-parameter pipeline:
    python -m coloring_template art.jpg --no-auto-tune --strategy dark --threshold 70

    # Transparent background for layered import into Colorflow:
    python -m coloring_template art.png --transparent

    # Save side-by-side preview to check quality:
    python -m coloring_template art.png --preview
"""

from __future__ import annotations

import argparse
import glob as _glob
import sys
from pathlib import Path

from .pipeline import STRATEGY_MAP, VALID_FORMATS, convert, convert_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coloring_template",
        description=(
            "Convert illustrations into coloring templates for iPad coloring apps "
            "(e.g. Colorflow). By default, runs an autoresearch parameter search "
            "per image and writes an experiment log under ./experiments/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "input",
        nargs="+",
        metavar="INPUT",
        help="Input image file(s). Accepts paths and glob patterns.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="./output",
        metavar="DIR",
        help="Output directory (default: ./output).",
    )

    # --- Auto-tune controls ------------------------------------------------
    tune_group = parser.add_argument_group("auto-tune (default)")
    tune_group.add_argument(
        "--auto-tune",
        dest="auto_tune",
        action="store_true",
        default=True,
        help="Run autoresearch parameter search (default: ON).",
    )
    tune_group.add_argument(
        "--no-auto-tune",
        dest="auto_tune",
        action="store_false",
        help="Disable autoresearch; use the fixed-parameter legacy pipeline.",
    )
    tune_group.add_argument(
        "--search-budget",
        type=int,
        default=20,
        metavar="N",
        help="Number of candidate parameter sets to evaluate (default: 20).",
    )
    tune_group.add_argument(
        "--experiments-dir",
        default="./experiments",
        metavar="DIR",
        help="Where to write experiment logs (default: ./experiments).",
    )
    tune_group.add_argument(
        "--no-log",
        dest="log_experiments",
        action="store_false",
        default=True,
        help="Skip writing experiment logs.",
    )
    tune_group.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Parallel workers for the search (default: auto).",
    )

    # --- Output format -----------------------------------------------------
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=list(VALID_FORMATS),
        default="png",
        help="Output format (default: png). Use 'both' for PNG + SVG.",
    )

    # --- Legacy pipeline (used when --no-auto-tune) ------------------------
    legacy_group = parser.add_argument_group("legacy pipeline (with --no-auto-tune)")
    legacy_group.add_argument(
        "-s", "--strategy",
        choices=list(STRATEGY_MAP.keys()),
        default="auto",
        help="Extraction strategy (default: auto).",
    )
    legacy_group.add_argument(
        "-t", "--threshold",
        type=int,
        default=60,
        metavar="N",
        help="Dark pixel threshold 0-255 (default: 60).",
    )
    legacy_group.add_argument(
        "--close-kernel",
        type=int,
        default=3,
        metavar="N",
        dest="close_kernel_size",
        help="Morphological close kernel size (default: 3).",
    )
    legacy_group.add_argument(
        "--no-smooth",
        action="store_false",
        dest="smooth",
        default=True,
        help="Disable edge smoothing.",
    )

    # --- Shared output options --------------------------------------------
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        metavar="N",
        help="Output DPI metadata (default: 300).",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=3000,
        metavar="PX",
        help="Minimum pixel length of the longest side (default: 3000).",
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Output as RGBA PNG with transparent background.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Also save a side-by-side original/template comparison PNG.",
    )

    return parser


def resolve_inputs(raw_inputs: list[str]) -> list[Path]:
    """Expand glob patterns and collect unique input paths."""
    paths = []
    for item in raw_inputs:
        expanded = _glob.glob(item, recursive=True)
        if expanded:
            paths.extend(Path(p) for p in expanded)
        else:
            paths.append(Path(item))

    seen = set()
    unique = []
    for p in paths:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(p)
    return unique


def _build_kwargs(args: argparse.Namespace) -> dict:
    return dict(
        auto_tune=args.auto_tune,
        search_budget=args.search_budget,
        output_format=args.output_format,
        experiments_dir=args.experiments_dir,
        log_experiments=args.log_experiments,
        workers=args.workers,
        strategy=args.strategy,
        threshold=args.threshold,
        close_kernel_size=args.close_kernel_size,
        smooth=args.smooth,
        dpi=args.dpi,
        transparent=args.transparent,
        min_size=args.min_size,
        preview=args.preview,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    inputs = resolve_inputs(args.input)
    output_dir = Path(args.output_dir)

    kwargs = _build_kwargs(args)

    if len(inputs) == 1:
        path = inputs[0]
        out_path = output_dir / (path.stem + "_coloring.png")
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            saved = convert(path, out_path, **kwargs)
            for p in saved:
                print(f"Saved: {p}")
            return 0
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"Error processing {path.name}: {exc}", file=sys.stderr)
            return 1

    print(f"Processing {len(inputs)} image(s) → {output_dir}/")
    results = convert_batch(inputs, output_dir, **kwargs)
    # results is a flat list of saved paths, potentially > len(inputs) with --format both.
    # Treat any input that produced zero outputs as a failure.
    produced_stems = {p.stem for p in results}
    failed = sum(
        1 for p in inputs if (p.stem + "_coloring") not in produced_stems
    )
    print(f"\nDone: {len(inputs) - failed} succeeded, {failed} failed.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
