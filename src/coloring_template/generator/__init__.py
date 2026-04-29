"""Prompt-to-template generation pipeline.

This package generates vector-first coloring templates, then validates the
rendered result as a flood-fillable coloring asset.
"""

from .pipeline import generate_template
from .validate import ValidationResult

__all__ = ["ValidationResult", "generate_template"]
