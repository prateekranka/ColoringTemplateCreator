"""OpenAI Image API provider for programmatic coloring-page generation."""

from __future__ import annotations

import base64
import os
from pathlib import Path

from .postprocess import clean_generated_raster
from .prompt import image_prompt
from .validate import ValidationResult, validate_png


DEFAULT_IMAGE_MODEL = "gpt-image-2"


def generate_openai_image_template(
    subject: str,
    output_path: str | Path,
    *,
    model: str | None = None,
    size: str = "1024x1536",
    quality: str = "medium",
    output_format: str = "png",
    background: str = "opaque",
    threshold: int = 210,
    transparent: bool = False,
    min_size: int = 3000,
    dpi: int = 300,
    svg: bool = False,
    audience: str = "kids ages 5-8",
    difficulty: str = "simple",
    save_raw: bool = True,
) -> tuple[Path, ValidationResult]:
    """Generate a coloring template using OpenAI's Image API.

    The raw model image is saved next to the cleaned template with a `_raw`
    suffix when `save_raw` is true.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI image generation requires the 'openai' package. "
            "Install project dependencies with `pip install -e .`."
        ) from exc

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI()
    prompt = image_prompt(subject, audience=audience, difficulty=difficulty)
    result = client.images.generate(
        model=model or os.environ.get("COLORING_TEMPLATE_IMAGE_MODEL", DEFAULT_IMAGE_MODEL),
        prompt=prompt,
        size=size,
        quality=quality,
        output_format=output_format,
        background=background,
    )

    image_base64 = result.data[0].b64_json
    raw_bytes = base64.b64decode(image_base64)
    raw_path = output_path.with_name(f"{output_path.stem}_raw.{output_format}")
    if save_raw:
        raw_path.write_bytes(raw_bytes)
    else:
        raw_path = output_path.with_name(f"{output_path.stem}_raw_temp.{output_format}")
        raw_path.write_bytes(raw_bytes)

    clean_generated_raster(
        raw_path,
        output_path,
        threshold=threshold,
        transparent=transparent,
        min_size=min_size,
        dpi=dpi,
        svg=svg,
    )
    if not save_raw:
        raw_path.unlink(missing_ok=True)

    report = validate_png(output_path)
    return output_path, report
