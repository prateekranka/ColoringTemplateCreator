"""Command-line interface for ColoringTemplateCreator.

Usage examples:
    # Single file (auto strategy):
    python -m coloring_template photo.png

    # Batch with explicit output dir:
    python -m coloring_template *.png -o ./templates/

    # Use dark pixel strategy with custom threshold:
    python -m coloring_template art.jpg --strategy dark --threshold 70

    # Transparent background for layered import into Colorflow:
    python -m coloring_template art.png --transparent

    # Save side-by-side preview to check quality:
    python -m coloring_template art.png --preview
"""

import argparse
import glob as _glob
import sys
from pathlib import Path

from .pipeline import convert, convert_batch, STRATEGY_MAP


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coloring_template",
        description=(
            "Convert pop art / illustrations into coloring templates "
            "for iPad coloring apps (e.g. Colorflow)."
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
        help="Output directory (default: ./output). Created if it doesn't exist.",
    )
    parser.add_argument(
        "-s", "--strategy",
        choices=list(STRATEGY_MAP.keys()),
        default="auto",
        help=(
            "Extraction strategy (default: auto).\n"
            "  auto     - Analyses the image and picks the best strategy\n"
            "  dark     - Direct dark-pixel extraction, best for bold black outlines\n"
            "  edge     - Adaptive threshold, best for colored/light outlines\n"
            "  canny    - Canny edge detection with bilateral filter\n"
            "  combined - Merges dark and edge results"
        ),
    )
    parser.add_argument(
        "-t", "--threshold",
        type=int,
        default=60,
        metavar="N",
        help=(
            "Dark pixel threshold 0-255 (default: 60). "
            "Used by 'dark', 'combined', and 'auto' strategies. "
            "Lower = stricter (only boldest lines); higher = more inclusive."
        ),
    )
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
        help=(
            "Minimum pixel length of the longest side (default: 3000). "
            "Smaller images are upscaled to this size."
        ),
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Output as RGBA PNG with transparent background (lines only).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Also save a side-by-side original/template comparison PNG.",
    )
    parser.add_argument(
        "--close-kernel",
        type=int,
        default=3,
        metavar="N",
        dest="close_kernel_size",
        help="Morphological close kernel size for gap-sealing (default: 3).",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_false",
        dest="smooth",
        help="Disable edge smoothing (keep raw binary edges).",
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
            paths.append(Path(item))  # Let pipeline raise FileNotFoundError

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in paths:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(p)
    return unique


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    inputs = resolve_inputs(args.input)
    output_dir = Path(args.output_dir)

    common_kwargs = dict(
        strategy=args.strategy,
        threshold=args.threshold,
        close_kernel_size=args.close_kernel_size,
        smooth=args.smooth,
        dpi=args.dpi,
        transparent=args.transparent,
        min_size=args.min_size,
        preview=args.preview,
    )

    if len(inputs) == 1:
        # Single file: output directly into output_dir with _coloring suffix
        path = inputs[0]
        out_path = output_dir / (path.stem + "_coloring.png")
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = convert(path, out_path, **common_kwargs)
            print(f"Saved: {result}")
            return 0
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"Error processing {path.name}: {exc}", file=sys.stderr)
            return 1
    else:
        # Batch mode
        print(f"Processing {len(inputs)} image(s) → {output_dir}/")
        results = convert_batch(inputs, output_dir, **common_kwargs)
        failed = len(inputs) - len(results)
        print(f"\nDone: {len(results)} succeeded, {failed} failed.")
        return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
