"""SVG output for coloring templates.

Primary: `potrace` binary via subprocess — produces clean Bezier paths from
a pure binary mask. This is the right tool because the input is already
bilevel, so potrace just vectorises the shapes.

Fallback: `svgwrite` + cv2 contours — jaggier output, but pure Python so the
feature stays usable without a system dependency.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


log = logging.getLogger(__name__)

POTRACE_BIN = "potrace"
HAVE_POTRACE = shutil.which(POTRACE_BIN) is not None


def save_svg(mask: np.ndarray, output_path: str | Path) -> Path:
    """Convert a binary line mask to SVG and save it.

    Args:
        mask: Binary mask (uint8) where 255 = line, 0 = background.
        output_path: Destination .svg file.

    Returns:
        Path to the saved SVG.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if HAVE_POTRACE:
        try:
            return _save_with_potrace(mask, output_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("potrace failed (%s); falling back to contour SVG", exc)

    return _save_with_contours(mask, output_path)


# ---------------------------------------------------------------------------
# potrace path
# ---------------------------------------------------------------------------

def _save_with_potrace(mask: np.ndarray, output_path: Path) -> Path:
    """Pipe a PBM of the mask into potrace, capture SVG on stdout."""
    pbm_bytes = _mask_to_pbm(mask)
    result = subprocess.run(
        [POTRACE_BIN, "-b", "svg", "--tight", "-o", "-", "-"],
        input=pbm_bytes,
        capture_output=True,
        check=True,
    )
    output_path.write_bytes(result.stdout)
    return output_path


def _mask_to_pbm(mask: np.ndarray) -> bytes:
    """Encode a binary mask as a PBM (P4) byte string.

    Potrace treats black (bit=1) as foreground, which matches our convention
    (255 = line). We pack bits MSB-first, 8 pixels per byte.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")

    h, w = mask.shape
    # Binarise defensively; anything >= 128 is "on".
    bits = (mask >= 128).astype(np.uint8)

    # Pack to bytes; np.packbits packs MSB-first which PBM expects.
    packed = np.packbits(bits, axis=1)
    header = f"P4\n{w} {h}\n".encode("ascii")
    return header + packed.tobytes()


# ---------------------------------------------------------------------------
# Fallback: cv2 contours → svgwrite
# ---------------------------------------------------------------------------

def _save_with_contours(mask: np.ndarray, output_path: Path) -> Path:
    import svgwrite  # local import so the dep is only required on fallback

    h, w = mask.shape[:2]
    dwg = svgwrite.Drawing(str(output_path), size=(w, h), profile="full")
    dwg.viewbox(0, 0, w, h)
    dwg.add(dwg.rect(insert=(0, 0), size=(w, h), fill="white"))

    contours, _ = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS
    )
    for contour in contours:
        if len(contour) < 2:
            continue
        pts = contour.reshape(-1, 2)
        path_data = "M " + " L ".join(f"{x},{y}" for x, y in pts) + " Z"
        dwg.add(dwg.path(d=path_data, fill="black", stroke="none"))

    dwg.save()
    return output_path
