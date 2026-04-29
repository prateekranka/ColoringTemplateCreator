"""Render sanitized SVG templates."""

from __future__ import annotations

import subprocess
import tempfile
import shutil
from pathlib import Path


def render_svg_to_png(svg_text: str, output_path: str | Path, size: int = 2048) -> Path:
    """Rasterize SVG text to a PNG file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import cairosvg
    except ImportError:
        cairosvg = None

    if cairosvg is not None:
        cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"),
            write_to=str(output_path),
            output_width=size,
            output_height=size,
        )
        return output_path

    if shutil.which("inkscape") is None:
        raise RuntimeError(
            "SVG rendering requires either the 'cairosvg' Python package or "
            "the 'inkscape' command-line tool."
        )

    with tempfile.NamedTemporaryFile(suffix=".svg", mode="w", delete=False) as handle:
        handle.write(svg_text)
        tmp_svg = Path(handle.name)
    try:
        subprocess.run(
            ["inkscape", "--export-type=png", f"--export-filename={output_path}", str(tmp_svg)],
            check=True,
            capture_output=True,
        )
    finally:
        tmp_svg.unlink(missing_ok=True)
    return output_path
