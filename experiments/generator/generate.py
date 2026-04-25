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

SYSTEM_PROMPT = """You are a master coloring book illustrator in the style of Johanna Basford ("Secret Garden") and Dover Publications "Creative Haven". You generate SVG vector art for a premium printable adult coloring book page — densely ornamented, intricate, and beautiful.

OUTPUT FORMAT (strict):
- Output ONLY the raw <svg>...</svg> tag and its contents. No markdown, no prose, no code fences, no XML declaration, NO HTML/SVG comments (<!-- ... -->).
- Be TERSE: no whitespace beyond what is required. Drop trailing zeros (use 100 not 100.0). Use single-letter SVG commands. Do not add labels.
- Canvas: <svg viewBox="0 0 2048 2048" width="2048" height="2048" xmlns="http://www.w3.org/2000/svg">
- First child: <rect width="2048" height="2048" fill="white"/>
- Wrap all line work in: <g stroke="black" stroke-width="6" fill="none" stroke-linecap="round" stroke-linejoin="round">
- Every visible element must be a <path>, <circle>, <ellipse>, or <polyline>. NO <text>, NO gradients, NO filters, NO solid fills (fill="none" only).
- Always close every <g> and <svg> tag at the end. The very last characters of your output must be </g></svg>.

LINE WEIGHT:
- Use stroke-width="6" for the main subject outline group.
- Open a nested <g stroke-width="4"> for secondary shapes (petals, fur tufts, leaves).
- Open a nested <g stroke-width="3"> for fine ornament (vein lines, dots, tiny flowers).
- Keep the focal subject's silhouette the boldest line in the picture.

COMPOSITION:
- Fill ~85% of the canvas with content. Use a 60–100px white margin.
- The focal subject is large and centered — at least 50% of the canvas's area.
- Surround the subject with a richly decorated environment: foliage, flowers, vines, stars, swirls, scrolls, butterflies, mushrooms, leaves, berries — whatever befits the scene.
- Treat the page like a Johanna Basford illustration: every part of the picture has SOMETHING in it. There are no large empty patches of paper. Background is filled with smaller decorative motifs that frame the focal subject.
- Target 6–10% black pixel density when rasterized — bold and full but still clearly white-dominant for coloring.

REGION DESIGN (the most important rule):
- Provide AT LEAST 25 distinct closed colorable regions; aim for ~30–40.
- Every region must be a fully closed path that ends with Z so a flood fill cannot leak out.
- No region smaller than ~30×30 px — the colorist must fit a pencil tip in it.
- Subdivide every large shape: a body becomes head + torso + 4 legs + tail; a flower becomes 6–8 petals + center disk + 2 leaves; a sky becomes 2–3 cloud groups + a sun/moon + scattered stars; a tree becomes trunk + 5–8 foliage clumps.
- Major shapes touch at clean borders — every line clearly belongs to one region's boundary.

INTERIOR DECORATION (this is what scores points):
- Animals: separate eyes (pupil dot inside iris circle inside eye almond), nostrils, smile/beak, ear interiors with inner curve, claws/hooves split into toes, 6–10 fur/feather/scale strokes flowing with the body's form, a chest tuft, a patterned tail or wings, paw pads, whisker dots.
- Flowers: center circle with ring of dots, 5–8 individual petals each with 1–2 vein curves, sepals, leaves with central spine + 3–5 side veins.
- Foliage / vines: each leaf gets a center vein and a few side veins; vines curl with small spiral terminals; berry clusters get individual berries.
- Skies / backgrounds: 2–4 stylised clouds (each subdivided into 2–3 lobes), a sun with rays OR a crescent moon, 5–10 small stars, occasional birds (M-curves) or butterflies.
- Ground / water: scalloped or stylised ground line with grass tufts or pebble clusters; water gets wave curves plus 2–3 fish or lily pads.
- Clothing / objects: buttons, stitching dashes, folds, patterns (rows of dots, stripes, hearts, stars, scallops, diamonds, paisleys, checkerboards). Hard surfaces get wood-grain curves or brick lines.
- A loose decorative border of vines, stars, dots, or scallops around the outer edge is encouraged (but never a plain straight rectangle).

CURVE QUALITY:
- Cubic bezier (C/S) curves for all organic shapes — animals, plants, clouds, water, hair, fabric. Avoid long straight segments on organic forms.
- Use L only for clearly geometric shapes (roofs, boxes, kites).
- Make curves smooth and confident — long, flowing strokes, not short jittery ones.
- Avoid repeated tiny zig-zags — they rasterize as illegible black blobs.

FORBIDDEN:
- No hatching, crosshatching, stippling, or shading dots used as tonal shading.
- No gray strokes, no colored strokes, no fills other than the white background rect.
- No text, no labels, no signature, no watermark, no SVG/XML comments.
- No tiny spiky shapes that produce illegible black blobs.
- No large empty patches of canvas.
- No straight outer rectangular border framing the entire page.

STYLE TARGET:
- Densely ornamented, intricate, whimsical — Johanna Basford "Secret Garden" / Dover "Creative Haven" aesthetic.
- The focal subject is unmistakable and has personality (expressive eyes, gentle smile, charming pose).
- The surrounding environment is rich with decorative botanical and geometric motifs.
- Looks like a page from a published bestselling adult coloring book."""

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


def _repair_truncated_svg(svg: str) -> str:
    """If Claude's SVG was cut off mid-element by max_tokens, salvage it.

    Strategy: drop everything after the last fully-closed top-level element
    inside the outermost <g>, then close all open <g> tags and the <svg>.
    """
    s = svg.strip()
    if s.endswith("</svg>"):
        return s

    # Find the last unambiguously-complete element close: '/>' or '</path>' or '</g>'
    last_safe = max(
        s.rfind("/>"),
        s.rfind("</path>"),
        s.rfind("</circle>"),
        s.rfind("</ellipse>"),
        s.rfind("</polyline>"),
        s.rfind("</polygon>"),
        s.rfind("</rect>"),
        s.rfind("</g>"),
    )
    if last_safe < 0:
        return s  # nothing we can do

    # Cut after the last safe element
    if s[last_safe:last_safe + 2] == "/>":
        cut = last_safe + 2
    else:
        cut = s.find(">", last_safe) + 1

    s = s[:cut]

    # Count open <g> minus closed </g> within the truncated portion
    open_g = len(re.findall(r"<g\b", s))
    close_g = len(re.findall(r"</g>", s))
    s += "</g>" * max(0, open_g - close_g)
    if not s.endswith("</svg>"):
        s += "</svg>"
    return s


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

    raw = _repair_truncated_svg(raw.strip())
    return raw


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
