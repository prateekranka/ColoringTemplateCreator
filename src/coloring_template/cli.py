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
        nargs="*",
        metavar="INPUT",
        help="Input image file(s). Accepts paths and glob patterns.",
    )
    parser.add_argument(
        "--generate",
        metavar="SUBJECT",
        help=(
            "Generate a new vector-first coloring template from a subject prompt "
            "instead of extracting outlines from an input image."
        ),
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
    parser.add_argument(
        "--svg",
        action="store_true",
        help="Output as SVG with traced vector paths (black lines on white background).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        metavar="N",
        help="Generation repair attempts for --generate mode (default: 3).",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        help=(
            "Model override for --generate mode. Defaults to COLORING_TEMPLATE_MODEL "
            "or the package default."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        metavar="N",
        help="Token budget for SVG generation in --generate mode (default: 8192).",
    )
    parser.add_argument(
        "--generate-provider",
        choices=["svg", "openai-image"],
        default="svg",
        help=(
            "Generation backend for --generate mode. 'svg' uses the vector LLM "
            "provider; 'openai-image' uses the OpenAI Image API."
        ),
    )
    parser.add_argument(
        "--image-size",
        default="1024x1536",
        metavar="SIZE",
        help="Image API size for --generate-provider openai-image (default: 1024x1536).",
    )
    parser.add_argument(
        "--image-quality",
        default="medium",
        choices=["low", "medium", "high", "auto"],
        help="Image API quality for --generate-provider openai-image (default: medium).",
    )
    parser.add_argument(
        "--image-background",
        default="opaque",
        choices=["opaque", "transparent", "auto"],
        help="Image API background mode for openai-image (default: opaque).",
    )
    parser.add_argument(
        "--raster-threshold",
        type=int,
        default=210,
        metavar="N",
        dest="raster_threshold",
        help="Binarization threshold for generated raster cleanup (default: 210).",
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

    output_dir = Path(args.output_dir)

    if args.generate:
        if args.input:
            parser.error("--generate cannot be combined with input image paths")
        if args.max_attempts < 1:
            parser.error("--max-attempts must be at least 1")

        output_dir.mkdir(parents=True, exist_ok=True)
        stem = "_".join(args.generate.lower().split())[:80] or "generated"
        safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
        out_path = output_dir / f"{safe_stem}_coloring.png"
        try:
            if args.generate_provider == "openai-image":
                from .generator import generate_openai_image_template

                result_path, report = generate_openai_image_template(
                    args.generate,
                    out_path,
                    model=args.model,
                    size=args.image_size,
                    quality=args.image_quality,
                    background=args.image_background,
                    threshold=args.raster_threshold,
                    transparent=args.transparent,
                    min_size=args.min_size,
                    dpi=args.dpi,
                    svg=args.svg,
                )
            else:
                from .generator import generate_template

                result_path, report = generate_template(
                    args.generate,
                    out_path,
                    max_attempts=args.max_attempts,
                    model=args.model,
                    max_tokens=args.max_tokens,
                )
            status = "passed" if report.ok else "needs review"
            print(f"Saved: {result_path}")
            print(
                "Validation: "
                f"{status}; density={report.black_density:.1%}, "
                f"regions={report.enclosed_regions}, "
                f"tiny_regions={report.tiny_regions}, "
                f"components={report.component_count}"
            )
            if report.feedback:
                print("Feedback:")
                for item in report.feedback:
                    print(f"  - {item}")
            return 0 if report.ok else 2
        except Exception as exc:
            print(f"Error generating template: {exc}", file=sys.stderr)
            return 1

    if not args.input:
        parser.error("provide input image path(s), or use --generate SUBJECT")

    inputs = resolve_inputs(args.input)

    common_kwargs = dict(
        strategy=args.strategy,
        threshold=args.threshold,
        close_kernel_size=args.close_kernel_size,
        smooth=args.smooth,
        dpi=args.dpi,
        transparent=args.transparent,
        min_size=args.min_size,
        preview=args.preview,
        svg=args.svg,
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
