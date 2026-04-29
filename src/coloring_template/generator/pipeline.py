"""End-to-end prompt-to-template generation pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .render import render_svg_to_png
from .svg_generate import generate_svg
from .svg_sanitize import sanitize_svg
from .validate import ValidationResult, validate_png


def generate_template(
    subject: str,
    output_path: str | Path,
    *,
    max_attempts: int = 3,
    model: str | None = None,
    max_tokens: int = 8192,
    save_report: bool = True,
) -> tuple[Path, ValidationResult]:
    """Generate, sanitize, render, validate, and retry a coloring template.

    Args:
        subject: Text prompt for the desired coloring page.
        output_path: Final PNG path.
        max_attempts: Number of generate/repair attempts before returning best effort.
        model: Optional model override for the generator client.
        max_tokens: Token budget for SVG generation.
        save_report: Save a JSON validation report next to the PNG.

    Returns:
        Tuple of final PNG path and its validation result.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feedback: list[str] | None = None
    last_svg: str | None = None
    last_result: ValidationResult | None = None
    last_error: Exception | None = None

    for _attempt in range(1, max_attempts + 1):
        try:
            raw_svg = generate_svg(subject, feedback, model=model, max_tokens=max_tokens)
            svg = sanitize_svg(raw_svg)
            with tempfile.TemporaryDirectory() as tmpdir:
                candidate_png = Path(tmpdir) / "candidate.png"
                render_svg_to_png(svg, candidate_png)
                result = validate_png(candidate_png)

            last_svg = svg
            last_result = result
            feedback = result.feedback
            if result.ok:
                break
        except Exception as exc:
            last_error = exc
            feedback = [f"previous SVG failed production validation: {exc}"]

    if last_svg is None or last_result is None:
        detail = f": {last_error}" if last_error is not None else ""
        raise RuntimeError(f"Could not generate a valid SVG candidate{detail}") from last_error

    svg_path = output_path.with_suffix(".svg")
    svg_path.write_text(last_svg, encoding="utf-8")
    render_svg_to_png(last_svg, output_path)

    if save_report:
        report_path = output_path.with_suffix(".json")
        report_path.write_text(
            json.dumps(
                {
                    "subject": subject,
                    "ok": last_result.ok,
                    "feedback": last_result.feedback,
                    "black_density": last_result.black_density,
                    "enclosed_regions": last_result.enclosed_regions,
                    "tiny_regions": last_result.tiny_regions,
                    "component_count": last_result.component_count,
                    "content_coverage": last_result.content_coverage,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return output_path, last_result
