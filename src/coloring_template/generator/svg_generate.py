"""Model client for SVG generation."""

from __future__ import annotations

import os
import re

from .prompt import SYSTEM_PROMPT, user_prompt


DEFAULT_MODEL = "claude-opus-4-7"


def _extract_svg(text: str) -> str:
    """Extract an SVG document from a model response."""
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text.strip())

    match = re.search(r"<svg\b.*?</svg>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"Model did not return an SVG document: {text[:200]!r}")
    return match.group(0).strip()


def generate_svg(
    subject: str,
    feedback: list[str] | None = None,
    *,
    model: str | None = None,
    max_tokens: int = 8192,
) -> str:
    """Generate raw SVG text for a coloring template subject."""
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "Prompt-to-template generation requires the 'anthropic' package. "
            "Install project dependencies with `pip install -e .`."
        ) from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model or os.environ.get("COLORING_TEMPLATE_MODEL", DEFAULT_MODEL),
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt(subject, feedback)}],
    )
    raw = response.content[0].text if response.content else ""
    return _extract_svg(raw)
