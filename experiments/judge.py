"""LLM visual judge for coloring template quality.

Uses Claude Haiku to score output images against gold-standard exemplars.
Each output image is scored 0-10 in a single API call that shows the output
alongside all gold-standard reference images.

Public API:
    score_images(output_paths, gold_dir) -> (list[float], float)
"""

import base64
import os
import re
import sys
from pathlib import Path

import anthropic

JUDGE_MODEL = "claude-haiku-4-5-20251001"
MAX_GOLD_IMAGES = 4

JUDGE_SYSTEM_PROMPT = """You are an expert judge evaluating coloring book page quality.
You will be shown one candidate coloring template and one or more gold-standard reference images.
Your job is to assign the candidate a score from 0 to 10 (integers only) based on how closely
it matches the quality and style of the reference images.

Scoring rubric:
  10 — Perfect. Indistinguishable in quality from the references. Clean, 1-2px-wide black
       outlines, all major regions are fully enclosed (suitable for flood-fill coloring),
       virtually no noise blobs, outline density visually similar to the references (~2-8%).
  8-9 — Professional quality. Minor imperfections: a few small gaps or tiny noise blobs,
        but the overall structure is clean and every main coloring region is bounded.
  6-7 — Acceptable. Recognizable as a coloring template but with notable flaws: some regions
        not fully closed, moderate noise, or lines too thick/thin vs references.
  4-5 — Poor. Outline structure present but significantly degraded: large gaps, heavy texture
        noise, solid fill areas that should be empty, or density far outside reference range.
  0-3 — Broken. Output is mostly black, mostly white, unrecognizable, or extreme noise makes
        it unusable as a coloring template.

Reply with a single integer and nothing else. No explanation."""


def _encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _image_media_type(path: Path) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(path.suffix.lower(), "image/png")


def _load_gold_images(gold_dir: Path) -> list[Path]:
    if not gold_dir.is_dir():
        return []
    candidates: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        candidates.extend(gold_dir.glob(ext))
    candidates = [p for p in candidates if p.is_file()]
    candidates.sort()
    return candidates[:MAX_GOLD_IMAGES]


def _build_message_content(output_image: Path, gold_images: list[Path]) -> list[dict]:
    content: list[dict] = []

    if gold_images:
        content.append({
            "type": "text",
            "text": (
                f"Here {'are' if len(gold_images) > 1 else 'is'} "
                f"{len(gold_images)} gold-standard reference coloring "
                f"template{'s' if len(gold_images) > 1 else ''} showing the target quality:"
            ),
        })
        for i, gold_path in enumerate(gold_images):
            block: dict = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _image_media_type(gold_path),
                    "data": _encode_image(gold_path),
                },
            }
            # Cache on the last gold image so the entire prefix is cacheable
            if i == len(gold_images) - 1:
                block["cache_control"] = {"type": "ephemeral"}
            content.append(block)
    else:
        content.append({
            "type": "text",
            "text": (
                "No gold-standard reference images are available. "
                "Score the candidate based on general coloring-book quality criteria."
            ),
        })

    content.append({
        "type": "text",
        "text": "Now here is the candidate coloring template to score:",
    })
    content.append({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": _image_media_type(output_image),
            "data": _encode_image(output_image),
        },
    })
    content.append({
        "type": "text",
        "text": "Score this candidate from 0 to 10 (integer only). Reply with the integer and nothing else.",
    })

    return content


def _parse_score(response_text: str) -> float:
    match = re.search(r'\b(10|[0-9])\b', response_text.strip())
    if match:
        return float(int(match.group(1)))
    return 0.0


def score_images(
    output_paths: list[Path],
    gold_dir: Path,
) -> tuple[list[float], float]:
    """Score pipeline output images against gold-standard exemplars.

    Makes one API call per output image. Gold images are prompt-cached.

    Args:
        output_paths: Paths to coloring template PNGs to evaluate.
        gold_dir: Directory containing gold-standard reference PNGs.

    Returns:
        (per_image_scores, mean_score) — scores are 0.0-10.0.
    """
    if not output_paths:
        return [], 0.0

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=api_key)
    gold_images = _load_gold_images(gold_dir)

    if not gold_images:
        print(
            f"WARNING: no gold-standard images found in {gold_dir}. "
            "Scoring without exemplars.",
            file=sys.stderr,
        )

    per_image_scores: list[float] = []

    for output_path in output_paths:
        try:
            content = _build_message_content(output_path, gold_images)
            response = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=16,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            raw_text = response.content[0].text if response.content else ""
            score = _parse_score(raw_text)
            print(
                f"  judge({output_path.name}): {raw_text!r} -> {score:.1f}",
                file=sys.stderr,
            )
        except anthropic.APIError as exc:
            print(f"  judge API error for {output_path.name}: {exc}", file=sys.stderr)
            score = 0.0
        except Exception as exc:
            print(f"  judge unexpected error for {output_path.name}: {exc}", file=sys.stderr)
            score = 0.0

        per_image_scores.append(score)

    mean_score = sum(per_image_scores) / len(per_image_scores) if per_image_scores else 0.0
    return per_image_scores, mean_score
