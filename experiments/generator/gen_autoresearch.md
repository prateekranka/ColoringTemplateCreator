# Generator Autoresearch — Claude SVG Coloring Template (2026-04-25)

You are an autonomous research agent improving a Claude-powered coloring book template
generator. Instead of extracting outlines from images, this system asks Claude Opus to
generate SVG vector art directly — black outlines on white, no fills — then rasterizes
to PNG and scores against professional coloring book gold standards.

Your job: edit the generator's system prompt and parameters to make Claude produce better
SVG illustrations. You operate in a tight loop: propose a change → run the trial →
inspect the score → commit or revert → repeat. NEVER pause or ask for permission.

---

## Why SVG generation

The image-extraction pipeline (src/coloring_template/) plateaued at 3.69/10 because the
pop-art inputs lack the hand-drawn interior decoration present in professional coloring
books (eyes, fur texture, petal veins, feather lines). Claude generating SVG directly can
create that decoration from scratch.

---

## What you may and may NOT edit

ALLOWED — exactly ONE file:
  - `experiments/generator/generate.py`

This file contains:
  - `SYSTEM_PROMPT` — the instruction Claude receives; your main lever
  - `GENERATION_PARAMS` — model, temperature, max_tokens
  - `generate_svg()`, `svg_to_png()`, `generate_template()` — implementation

FORBIDDEN — everything else:
  - `experiments/generator/gen_trial.py`
  - `experiments/generator/gen_loop.py`
  - `experiments/generator/subjects.py`
  - `experiments/generator/gen_autoresearch.md`
  - `experiments/generator/gen_results.tsv`
  - Anything under `src/`, `examples/`, `experiments/gold_standard/`

---

## Evaluation command

    python experiments/generator/gen_trial.py

Randomly samples 4 subjects from `subjects.py`, generates SVG → PNG for each,
scores each PNG against the gold-standard coloring book references, prints a
table to stderr, and prints exactly one line to stdout:

    score: X.XX

Mean score is 0.00–10.00. Higher is better. Parse that line.

---

## Git workflow

KEEP (score improved or equal to previous best):
  1. git add experiments/generator/generate.py
  2. git commit -m "<description> (score: X.XX)"
  3. Append TSV row to experiments/generator/gen_results.tsv (status=keep)

DISCARD (score regressed):
  1. git reset --hard HEAD
  2. Append TSV row (status=discard)

CRASH (gen_trial.py exited non-zero or no score line):
  1. git reset --hard HEAD
  2. Append TSV row (status=crash, score=N/A)
  3. Read the error, diagnose before next attempt

---

## gen_results.tsv row format

Tab-separated:
  <git_short_hash>  <score>  <status>  <description>  <ISO8601_UTC_timestamp>

---

## Scoring target

Aim for 7+. The scale:
  - 0–3  Broken (blank, scribble, no recognizable subject)
  - 4–5  Recognizable but sparse or malformed (missing limbs, no detail)
  - 6–7  Good coloring template — clean lines, closed regions, some interior detail
  - 8–9  Professional quality — matches gold standard style closely
  - 10   Indistinguishable from the gold standard references

Gold standard target: clean 1-2px black outlines, all major regions enclosed (flood-fill
without leaking), decorative interior detail (fur lines, petal veins, eye details),
friendly and inviting, 2-8% black pixel density.

---

## What to tune in SYSTEM_PROMPT

1. **Detail level instructions** — tell Claude exactly how many colorable regions to
   include (e.g., "at least 10 distinct regions"), how much interior detail (e.g.,
   "3-5 interior detail lines per major element")
2. **Path quality hints** — "use smooth bezier curves", "close all paths with Z",
   "avoid straight lines for organic shapes"
3. **Density target** — if score is low because output is sparse or too dense,
   adjust the % target
4. **Style reference** — try describing the gold standard aesthetic more precisely
   ("similar to Dover coloring books", "Johanna Basford style")
5. **Forbidden elements** — add explicit prohibitions ("no hatching", "no crosshatch",
   "no stippling")
6. **Output format strictness** — if the model adds markdown or preamble, add
   stronger "output SVG only" instructions
7. **Composition hints** — "subject should fill 70% of the canvas", "leave white
   margins on all sides"

## What to tune in GENERATION_PARAMS

- `temperature` — 0.5 for more consistent output, 1.0 for more varied/creative
- `max_tokens` — increase to 8192 if complex SVG is being truncated
- `model` — do not change; claude-opus-4-7 is required

---

## Hard constraints

- NEVER run `git push`
- NEVER modify any file other than `experiments/generator/generate.py`
- NEVER run `pip install`, `conda install`
- NEVER run `rm -rf`
- ALWAYS run the trial before committing
- ALWAYS append to gen_results.tsv after every trial
- Make ONE logical change per trial
- Run indefinitely until interrupted
