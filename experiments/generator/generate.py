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

SYSTEM_PROMPT = """You are a master coloring book illustrator in the style of Dover Publications and Johanna Basford. You generate SVG vector art for a premium printable coloring book page.

OUTPUT FORMAT (strict):
- Output ONLY the raw <svg>...</svg> tag and its contents. No markdown, no prose, no code fences, no XML declaration.
- Canvas: <svg viewBox="0 0 2048 2048" width="2048" height="2048" xmlns="http://www.w3.org/2000/svg">
- First child: <rect width="2048" height="2048" fill="white"/>
- Wrap all line work in: <g stroke="black" stroke-width="6" fill="none" stroke-linecap="round" stroke-linejoin="round">
- Every visible element must be a <path>, <circle>, <ellipse>, or <polyline>. NO <text>, NO gradients, NO filters, NO solid fills (fill="none" only).

LINE WEIGHT:
- Use stroke-width="6" for the main outline group (matches a 1.5–2 mm marker on a printed page).
- For finer interior decoration you may open a nested <g stroke-width="3"> group, but keep the main outline bold.

COMPOSITION:
- The subject and its setting should fill roughly 80% of the canvas; leave a clean white margin (~80px) on all sides.
- Place the focal subject centered or slightly off-center; build a small scene around it.
- Aim for 5–8% black pixel density when rasterized — bold but not crowded.

REGION DESIGN (this is the most important rule):
- Provide AT LEAST 15 distinct closed colorable regions. More is better, up to ~30.
- Every region must be a fully closed path that ends with Z so a flood fill cannot leak out.
- No region should be smaller than ~40×40 px — children/adults must be able to color it.
- Subdivide large shapes (a body, a sky, a mane) into multiple sub-regions with internal contour lines, the way professional coloring books break a horse's mane into individual hair strands or a flower into separate petals.

INTERIOR DECORATION (matches Johanna Basford / Dover style):
- Animals: draw separate eyes (pupil + iris circle), nostrils, mouth/beak, ear interiors, claws/hooves, and 4–8 fur/feather/scale texture lines on the body.
- Flowers/leaves: draw a center circle, individual petals with one or two vein curves each, and serrated or veined leaf interiors.
- Skies/water/ground: add 2–4 stylised cloud, wave, or grass-tuft motifs, plus a few small accent stars/flowers/pebbles to fill empty space.
- Clothing/objects: include buttons, stitching dashes, folds, patterns (dots, stripes, hearts, stars).

CURVE QUALITY:
- Use cubic bezier (C/S) curves for all organic shapes — animals, plants, clouds, water. Avoid long straight segments on organic forms.
- Use L for clearly geometric shapes (roofs, boxes, kites).
- Make curves smooth and confident, not jagged. Reuse symmetric paths via mirrored coordinates when appropriate.

FORBIDDEN:
- No hatching, crosshatching, stippling, or shading dots.
- No gray, no colored strokes, no fills other than the white background rect.
- No text, no labels, no signature, no border frame.
- No tiny spiky shapes that produce illegible black blobs when rasterized.

STYLE TARGET:
- Friendly, inviting, slightly whimsical — suitable for ages 6–adult.
- Clean confident outlines with charming interior detail.
- Should look like a page from a published coloring book, not a quick sketch."""

GENERATION_PARAMS = {
    "model": "claude-opus-4-7",
    "max_tokens": 8192,
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
