"""Claude SVG coloring template generator.

This is the ONLY file the autoresearch agent may edit.
Tune SYSTEM_PROMPT and GENERATION_PARAMS to improve judge scores.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Tune these — the agent edits this file to improve scores
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert coloring book illustrator. Generate SVG vector art suitable for a children's coloring book page.

Rules:
- Output ONLY the raw SVG tag and its contents. No markdown, no explanation, no code fences.
- Canvas: viewBox="0 0 2048 2048" width="2048" height="2048"
- Background: a white rectangle <rect width="2048" height="2048" fill="white"/>
- All lines: stroke="black" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"
- Draw clean, closed outline paths for all major elements (body, head, limbs, petals, leaves, etc.)
- Include simple interior detail lines (eyes, fur texture, petal veins, feather lines) — sparse but present
- Target 3-8% black pixel density when rasterized to PNG
- NO gradients, NO colors, NO solid fills — outlines only
- Style: friendly, simple, suitable for young children aged 4-10
- Make all regions large enough to color with a finger on an iPad
- Use smooth bezier curves (C, S commands) for organic shapes; straight lines (L) for geometric shapes
- Every closed region should be a proper closed path (end with Z)
- Include at least 8 distinct colorable regions"""

GENERATION_PARAMS = {
    "model": "claude-opus-4-7",
    "max_tokens": 4096,
    "temperature": 1.0,
}

# ---------------------------------------------------------------------------
# Implementation — agent should not need to change below this line,
# but may if needed to support new GENERATION_PARAMS keys
# ---------------------------------------------------------------------------

try:
    import cairosvg
    _HAVE_CAIROSVG = True
except ImportError:
    _HAVE_CAIROSVG = False


def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set")
    return anthropic.Anthropic(api_key=api_key)


def generate_svg(subject: str) -> str:
    """Call Claude to generate an SVG string for the given subject."""
    client = _get_client()
    response = client.messages.create(
        model=GENERATION_PARAMS["model"],
        max_tokens=GENERATION_PARAMS["max_tokens"],
        temperature=GENERATION_PARAMS.get("temperature", 1.0),
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Draw a coloring book page of: {subject}"},
        ],
    )
    raw = response.content[0].text if response.content else ""

    # Strip markdown code fences if the model includes them despite instructions
    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw.strip())

    if not raw.strip().startswith("<svg"):
        raise ValueError(f"Model did not return an SVG (got: {raw[:200]!r})")
    return raw.strip()


def svg_to_png(svg_text: str, output_path: Path) -> Path:
    """Rasterize an SVG string to a PNG file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if _HAVE_CAIROSVG:
        cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"),
            write_to=str(output_path),
            output_width=2048,
            output_height=2048,
        )
    else:
        # Fallback: write SVG to temp file and call inkscape
        with tempfile.NamedTemporaryFile(suffix=".svg", mode="w", delete=False) as f:
            f.write(svg_text)
            tmp_svg = f.name
        try:
            subprocess.run(
                ["inkscape", "--export-type=png", f"--export-filename={output_path}", tmp_svg],
                check=True, capture_output=True,
            )
        finally:
            Path(tmp_svg).unlink(missing_ok=True)

    return output_path


def generate_template(subject: str, output_path: Path) -> Path:
    """Generate a coloring template PNG for the given subject.

    Also saves the raw SVG alongside the PNG for inspection.
    Returns the path to the PNG.
    """
    output_path = Path(output_path)
    svg_text = generate_svg(subject)

    # Save SVG for debugging
    svg_path = output_path.with_suffix(".svg")
    svg_path.write_text(svg_text, encoding="utf-8")

    return svg_to_png(svg_text, output_path)
