"""Prompt-to-template generation pipeline.

This package generates vector-first coloring templates, then validates the
rendered result as a flood-fillable coloring asset.
"""

from .pipeline import generate_template
from .openai_image import generate_openai_image_template
from .validate import ValidationResult

__all__ = ["ValidationResult", "generate_openai_image_template", "generate_template"]
