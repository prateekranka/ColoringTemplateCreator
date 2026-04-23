"""Core conversion pipeline: pop art image → coloring template.

Two modes:

- **Auto-tune mode (default)**: runs an autoresearch parameter search over the
  DarkPixel + cleanup pipeline, scores each candidate against gold-standard
  quality metrics, and keeps the best. Writes a full experiment log under
  `./experiments/` unless disabled. Only sensible for illustrations with
  explicit black outlines.

- **Legacy mode (`auto_tune=False`)**: the original fixed-parameter pipeline,
  preserved for back-compat — load → strategy → cleanup → save.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

from .autotune import ExperimentLogger, auto_tune as run_auto_tune
from .cleanup import clean
from .output import save, save_preview
from .output_svg import save_svg
from .strategies import CombinedStrategy, DarkPixelStrategy, EdgeDetectStrategy


# ---------------------------------------------------------------------------
# Auto-strategy selection (legacy mode)
# ---------------------------------------------------------------------------

def select_strategy(img_rgb: np.ndarray, threshold: int = 60):
    """Analyse image statistics to choose the best extraction strategy.

    Used only in legacy (`auto_tune=False`) mode. Auto-tune mode always uses
    DarkPixelStrategy since the intended input is illustrations with explicit
    black outlines.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    total = gray.size

    outline_mask = (gray < 50) & (saturation < 60)
    outline_ratio = np.sum(outline_mask) / total

    if outline_ratio > 0.02:
        dark_binary = (outline_mask.astype(np.uint8) * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        eroded = cv2.erode(dark_binary, kernel, iterations=2)
        surviving_ratio = np.sum(eroded > 0) / total

        if surviving_ratio > 0.005:
            if outline_ratio > 0.04:
                return DarkPixelStrategy(threshold=threshold)
            return EdgeDetectStrategy(method="adaptive")
        return DarkPixelStrategy(threshold=threshold)
    return EdgeDetectStrategy(method="adaptive")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

STRATEGY_MAP = {
    "dark": lambda t: DarkPixelStrategy(threshold=t),
    "edge": lambda _: EdgeDetectStrategy(method="adaptive"),
    "canny": lambda _: EdgeDetectStrategy(method="canny"),
    "combined": lambda t: CombinedStrategy(threshold=t),
    "auto": select_strategy,
}

VALID_FORMATS = ("png", "svg", "both")


def _seed_for_path(path: Path) -> int:
    """Deterministic per-image seed (same image → same search)."""
    return abs(hash(str(path.resolve()))) & 0x7FFFFFFF


def convert(
    input_path: str | Path,
    output_path: str | Path,
    *,
    # Auto-tune mode (default)
    auto_tune: bool = True,
    search_budget: int = 20,
    output_format: str = "png",
    experiments_dir: str | Path = "./experiments",
    log_experiments: bool = True,
    workers: int | None = None,
    # Legacy (auto_tune=False) mode
    strategy: str = "auto",
    threshold: int = 60,
    close_kernel_size: int = 3,
    min_component_area: int | None = None,
    smooth: bool = True,
    # Shared
    dpi: int = 300,
    transparent: bool = False,
    min_size: int = 3000,
    preview: bool = False,
) -> list[Path]:
    """Convert an illustration to a coloring template.

    Args:
        input_path: Input image (PNG/JPG/WebP).
        output_path: Output PNG path. The SVG (if requested) is written with
                     the same stem and a `.svg` extension.
        auto_tune: If True (default), run autoresearch to pick parameters
                   per-image. If False, use the fixed-parameter legacy path.
        search_budget: Number of candidates to try in auto-tune mode.
        output_format: 'png', 'svg', or 'both'.
        experiments_dir: Where to write experiment logs (auto-tune mode only).
        log_experiments: If False, skip writing experiment logs.
        workers: Parallel workers for auto-tune; None = auto-detect.
        strategy, threshold, close_kernel_size, min_component_area, smooth:
            Legacy-mode parameters, ignored when auto_tune=True.
        dpi, transparent, min_size, preview: PNG output options.

    Returns:
        List of saved paths (PNG, SVG, or both, depending on output_format).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if output_format not in VALID_FORMATS:
        raise ValueError(
            f"output_format must be one of {VALID_FORMATS}, got {output_format!r}"
        )

    # Stage 1: Load & normalize
    img_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Stage 2 + 3: produce the cleaned binary mask.
    if auto_tune:
        mask = _run_auto_tune(
            img_rgb=img_rgb,
            input_path=input_path,
            budget=search_budget,
            workers=workers,
            experiments_dir=Path(experiments_dir),
            log_experiments=log_experiments,
        )
    else:
        mask = _run_legacy(
            img_rgb=img_rgb,
            strategy=strategy,
            threshold=threshold,
            close_kernel_size=close_kernel_size,
            min_component_area=min_component_area,
            smooth=smooth,
        )

    # Stage 4: Save output(s)
    saved: list[Path] = []
    if output_format in ("png", "both"):
        png_path = save(
            mask, output_path, dpi=dpi, transparent=transparent, min_size=min_size
        )
        saved.append(png_path)

    if output_format in ("svg", "both"):
        svg_path = output_path.with_suffix(".svg")
        saved.append(save_svg(mask, svg_path))

    if preview:
        preview_path = output_path.with_name(output_path.stem + "_preview.png")
        save_preview(img_rgb, mask, preview_path, dpi=min(dpi, 150))

    return saved


def _run_auto_tune(
    img_rgb: np.ndarray,
    input_path: Path,
    budget: int,
    workers: int | None,
    experiments_dir: Path,
    log_experiments: bool,
) -> np.ndarray:
    seed = _seed_for_path(input_path)
    if log_experiments:
        with ExperimentLogger(
            image_path=input_path,
            budget=budget,
            seed=seed,
            experiments_dir=experiments_dir,
        ) as logger:
            mask, params, score, _ = run_auto_tune(
                img_rgb,
                budget=budget,
                workers=workers,
                seed=seed,
                on_candidate=logger.record,
            )
            logger.finalise(params, score)
    else:
        mask, params, score, _ = run_auto_tune(
            img_rgb, budget=budget, workers=workers, seed=seed,
        )

    print(
        f"  autotune: score={score.total:.3f} "
        f"(band={score.band:.2f} fill={score.fill:.2f} closed={score.closed:.2f} "
        f"noise={score.noise:.2f} crisp={score.crisp:.2f}) "
        f"params={params.to_dict()}"
    )
    return mask


def _run_legacy(
    img_rgb: np.ndarray,
    strategy: str,
    threshold: int,
    close_kernel_size: int,
    min_component_area: int | None,
    smooth: bool,
) -> np.ndarray:
    if strategy not in STRATEGY_MAP:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Choose from: {', '.join(STRATEGY_MAP)}"
        )
    if strategy == "auto":
        strat = select_strategy(img_rgb, threshold)
    else:
        strat = STRATEGY_MAP[strategy](threshold)

    mask = strat.extract(img_rgb)
    return clean(
        mask,
        close_kernel_size=close_kernel_size,
        min_component_area=min_component_area,
        smooth=smooth,
    )


def convert_batch(
    input_paths: list[Path],
    output_dir: Path,
    **kwargs,
) -> list[Path]:
    """Convert multiple images. Returns a flat list of every saved path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[Path] = []
    errors = []

    for path in input_paths:
        out_path = output_dir / (path.stem + "_coloring.png")
        try:
            saved = convert(path, out_path, **kwargs)
            results.extend(saved)
            print(f"  OK  {path.name} → {', '.join(p.name for p in saved)}")
        except Exception as exc:
            errors.append((path, exc))
            print(f"  ERR {path.name}: {exc}")

    if errors:
        print(f"\n{len(errors)} error(s) occurred:")
        for path, exc in errors:
            print(f"  {path.name}: {exc}")

    return results
