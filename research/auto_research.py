"""Auto Research System — evaluates all strategies on all example images.

For each image:
  - Runs all five strategies: auto, dark, edge, canny, combined
  - Computes quality metrics on the resulting outline mask
  - Saves output PNGs to research/output/<image_name>/
  - Prints a ranked report

Metrics
-------
outline_density   : fraction of pixels that are outline (want ~2–10%)
thinness          : fraction of outline pixels that are thin lines
                    (measured by how much the mask shrinks under erosion)
num_components    : number of connected outline components (fewer = cleaner)
closure_score     : fraction of outline pixels that form closed-region boundaries
                    (estimated via flood-fill reachability from edges)
composite         : weighted score used for ranking (higher = better)
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Make the src package importable when run from anywhere
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coloring_template.pipeline import convert, select_strategy, STRATEGY_MAP
from coloring_template.strategies import DarkPixelStrategy, EdgeDetectStrategy, CombinedStrategy
from coloring_template.cleanup import clean

OUTPUT_DIR = Path(__file__).parent / "output"
EXAMPLES_DIR = ROOT / "examples"
THRESHOLD = 60


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(mask: np.ndarray) -> dict:
    """Compute quality metrics for an outline mask (uint8, 0/255)."""
    h, w = mask.shape
    total = h * w
    outline_pixels = int(np.sum(mask > 0))
    outline_density = outline_pixels / total

    # Thinness: erode with 3×3 kernel; thin lines disappear, fills survive.
    # Thinness score = fraction of outline pixels that disappear under erosion.
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask, kernel3, iterations=1)
    surviving = int(np.sum(eroded > 0))
    thinness = 1.0 - (surviving / outline_pixels) if outline_pixels > 0 else 0.0

    # Connected components (background = label 0, ignore it)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    num_components = num_labels - 1

    # Closure score: estimate how well outlines form closed regions.
    # Flood-fill from all four borders on an inverted mask; if a white region is
    # reachable from the border it is an *open* region (not enclosed).
    # Closure = fraction of non-outline area that is enclosed (not reachable from border).
    inv = cv2.bitwise_not(mask)
    flood = inv.copy()
    # OpenCV floodFill needs a mask one pixel larger on each side
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    # Flood from corners/edges (sample 200 border points)
    border_pts = (
        [(0, y) for y in range(0, h, max(1, h // 50))]
        + [(w - 1, y) for y in range(0, h, max(1, h // 50))]
        + [(x, 0) for x in range(0, w, max(1, w // 50))]
        + [(x, h - 1) for x in range(0, w, max(1, w // 50))]
    )
    for x, y in border_pts:
        if flood[y, x] == 255:
            cv2.floodFill(flood, flood_mask, (x, y), 128)
    open_pixels = int(np.sum(flood == 128))
    total_white = int(np.sum(inv > 0))
    enclosed_pixels = max(0, total_white - open_pixels)
    closure_score = enclosed_pixels / total_white if total_white > 0 else 0.0

    # Composite score: favour moderate density, high thinness, closure, few components.
    # Density penalty: best around 3–8%, heavily penalise >15% or <0.5%.
    density_score = 1.0 - abs(outline_density - 0.05) / 0.10
    density_score = max(0.0, min(1.0, density_score))

    # Component penalty: fewer is better (normalised against 500 as "messy" baseline)
    component_score = max(0.0, 1.0 - num_components / 500)

    composite = (
        0.30 * density_score
        + 0.30 * thinness
        + 0.25 * closure_score
        + 0.15 * component_score
    )

    return {
        "outline_density": outline_density,
        "thinness": thinness,
        "num_components": num_components,
        "closure_score": closure_score,
        "composite": composite,
    }


# ---------------------------------------------------------------------------
# Per-image research
# ---------------------------------------------------------------------------

def research_image(image_path: Path) -> dict:
    """Run all strategies on one image and collect metrics."""
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Which strategy does auto select?
    auto_strat = select_strategy(img_rgb, THRESHOLD)
    auto_name = type(auto_strat).__name__

    image_out_dir = OUTPUT_DIR / image_path.stem
    image_out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    strategy_objects = {
        "dark":     DarkPixelStrategy(threshold=THRESHOLD),
        "edge":     EdgeDetectStrategy(method="adaptive"),
        "canny":    EdgeDetectStrategy(method="canny"),
        "combined": CombinedStrategy(threshold=THRESHOLD),
        "auto":     auto_strat,
    }

    for name, strat in strategy_objects.items():
        t0 = time.perf_counter()
        raw_mask = strat.extract(img_rgb)
        cleaned = clean(raw_mask, close_kernel_size=3, smooth=True)
        elapsed = time.perf_counter() - t0

        metrics = compute_metrics(cleaned)
        metrics["elapsed_s"] = elapsed

        # Save output image
        out_file = image_out_dir / f"{name}.png"
        cv2.imwrite(str(out_file), cv2.bitwise_not(cleaned))   # white bg, black lines

        results[name] = metrics

    return {
        "image": image_path.name,
        "auto_selected": auto_name,
        "strategies": results,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

HEADER = (
    f"{'Strategy':<10}  {'Density':>8}  {'Thinness':>9}  "
    f"{'Closure':>8}  {'#Comps':>7}  {'Composite':>9}  {'ms':>6}"
)
SEP = "-" * len(HEADER)


def print_report(data: dict) -> None:
    image = data["image"]
    auto = data["auto_selected"]
    strats = data["strategies"]

    print(f"\n{'='*60}")
    print(f"  Image : {image}")
    print(f"  Auto selected : {auto}")
    print(f"{'='*60}")
    print(HEADER)
    print(SEP)

    ranked = sorted(strats.items(), key=lambda kv: kv[1]["composite"], reverse=True)
    for name, m in ranked:
        marker = " *" if name == "auto" else "  "
        print(
            f"{name:<10}{marker}"
            f"  {m['outline_density']:>7.2%}"
            f"  {m['thinness']:>8.2%}"
            f"  {m['closure_score']:>7.2%}"
            f"  {m['num_components']:>7d}"
            f"  {m['composite']:>9.4f}"
            f"  {m['elapsed_s']*1000:>5.0f}"
        )

    best = ranked[0][0]
    print(SEP)
    print(f"  Best strategy : {best}  (composite {ranked[0][1]['composite']:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    image_paths = sorted(EXAMPLES_DIR.glob("*.png")) + sorted(EXAMPLES_DIR.glob("*.jpg"))
    image_paths = [p for p in image_paths if not p.name.startswith("create_")]

    if not image_paths:
        print("No images found in examples/")
        sys.exit(1)

    print(f"Auto Research System")
    print(f"====================")
    print(f"Images  : {len(image_paths)}")
    print(f"Output  : {OUTPUT_DIR}")

    all_results = []
    for img_path in image_paths:
        print(f"\nProcessing: {img_path.name} ...", end=" ", flush=True)
        result = research_image(img_path)
        print("done")
        all_results.append(result)
        print_report(result)

    # Summary table
    print(f"\n\n{'='*60}")
    print("  SUMMARY — Auto strategy selection accuracy")
    print(f"{'='*60}")
    strategy_class_map = {
        "DarkPixelStrategy": "dark",
        "EdgeDetectStrategy": "edge",   # includes canny
        "CombinedStrategy": "combined",
    }

    for r in all_results:
        auto_class = r["auto_selected"]
        auto_logical = strategy_class_map.get(auto_class, auto_class)
        strats = r["strategies"]
        ranked = sorted(strats.items(), key=lambda kv: kv[1]["composite"], reverse=True)
        best = ranked[0][0]
        match = "✓" if auto_logical == best or (
            auto_class == "EdgeDetectStrategy" and best in ("edge", "canny")
        ) else "✗"
        print(
            f"  {r['image']:<35}  auto→{auto_logical:<8}  best={best:<8}  {match}"
        )

    print(f"\nOutputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
