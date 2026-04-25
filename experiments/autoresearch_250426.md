# Autoresearch — Coloring Template Optimizer (2026-04-25)

Applies Karpathy's autoresearch pattern (github.com/karpathy/autoresearch) to
the ColoringTemplateCreator OpenCV pipeline. An LLM agent edits source code,
runs a fixed evaluation command, scores output quality via an LLM visual judge
against gold-standard coloring-book screenshots, and hill-climbs via git.

---

## Why This Approach

### LLM judge over numeric metrics

The prior research run (`research/auto_research.py`) computed a numeric composite
(outline density + thinness + closure score + component count). It consistently
ranked `canny` best — yet the auto selector chose `dark` or `edge`. The numeric
metrics miss what matters: does the output look like a professional coloring page?

Canny can score high on thinness while producing a jagged, underconnected skeleton
that bleeds color when flood-filled. A human (or a vision LLM) would immediately
reject it. The LLM judge looks at what a human looks at: cleanliness, closed
regions, appropriate line weight, density similar to the gold references.

### Edit-any-code scope (not just parameter tuning)

The searchable parameter space is small (threshold, kernel size, ±5 more). A
numeric optimizer (Optuna, random search) would exhaust the useful range within
~100 trials and plateau. The real gains come from structural changes: tweaking
the bilateral filter sigma that controls how much texture is blurred before
adaptive thresholding, adjusting the morphological close kernel shape, adding a
skeletonization pass, or rewriting the auto-strategy selector heuristic entirely.
An LLM agent that reads the current code, sees what scores it got, and reasons
about WHY the output looks wrong is the right tool for this open-ended space.

### Headless Anthropic API loop, not Optuna

Bayesian optimizers model a smooth numeric loss surface. Our loss is:
  - partly numeric (score 0-10)
  - partly categorical (which strategy class to use)
  - partly structural (code edits that change the computation graph)

An LLM agent naturally handles all three, produces a rationale for each change,
and can interpret visual feedback ("lines too thick in the hair region") in a way
no optimizer can.

### claude-haiku-4-5 for judge

Cheap, fast, and vision-capable. Each trial makes 4 judge API calls (one per test
image). At ~1-2k input tokens per call after caching the gold images, a trial
costs <$0.01. This allows hundreds of overnight iterations without cost concern.

### claude-opus-4-7 for agent

The agent must read 400-line Python files, identify the relevant 5-line block,
reason about its effect on the visual output, and write a targeted fix. This
requires strong code + reasoning capability. Opus is the right choice; do not
downgrade to Sonnet or Haiku for the agent.

### results.tsv over results.json

Flat TSV is grep-able (`grep keep experiments/results.tsv`), diffable, and
human-readable at a glance. Matches Karpathy's original format. JSON would be
easier to parse programmatically but harder to scan at 2am when debugging.

### Prompt caching for gold images and system prompt

The gold-standard reference images are the same in every judge API call during a
run. Without caching, they'd be billed repeatedly. `cache_control: ephemeral`
(5-minute TTL) on the last gold image block caches the entire prefix, cutting
judge costs by ~70% for runs with 4+ trials. The system prompt (this file) is
also cached in loop.py for the same reason.

### Bash blocklist + write_file path restriction

The agent must not be able to:
- Delete files (`rm -rf`) — catastrophic, hard to recover
- Push to remote (`git push`) — publishes half-baked experiments
- Install packages (`pip install`) — silently mutates the shared environment
- Overwrite loop.py/run_trial.py/judge.py — would break the harness silently

`write_file` is path-prefix-checked at the tool level (not just by instruction)
because the agent might misread the instructions under pressure. Bash can READ
experiments/ freely (needed for `git status`, `cat results.tsv`) but git push
is blocked unconditionally.

---

## Planned Experiment Order

Start from baseline, then try changes roughly in this order of expected impact:

1. **Auto-strategy thresholds** (`pipeline.py:select_strategy`)
   - The current `outline_ratio > 0.02` and `surviving_ratio > 0.005` thresholds
     were hand-picked. Adjust to better separate pop-art outlines from fills.

2. **EdgeDetect bilateral filter** (`strategies/edge_detect.py`)
   - `d=9, sigmaColor=75, sigmaSpace=75` — larger sigmaColor blurs more texture;
     smaller d is faster. The adaptive `blockSize=21, C=5` may need tuning per
     image style. Try blockSize ∈ {11, 15, 21, 31} and C ∈ {3, 5, 8}.

3. **DarkPixelStrategy saturation gate** (`strategies/dark_pixel.py`)
   - `saturation < 80` gate; `very_dark = gray < threshold//2` fallback.
     Adjust the saturation boundary and the very-dark threshold.

4. **Cleanup morphological ops** (`cleanup.py`)
   - close_kernel_size, min_component_area, max_fill_ratio, smooth on/off.
   - Try MORPH_RECT vs MORPH_ELLIPSE kernels.

5. **New post-processing: skeletonization**
   - After cleanup, apply Zhang-Suen thinning or OpenCV's `ximgproc.thinning`
     to reduce thick blobs to 1-2px lines. Add as an optional step in cleanup.py.

6. **New strategy class** (`strategies/`)
   - E.g. a "stroke" strategy that uses Laplacian of Gaussian or DoG to find
     outline boundaries specifically, tuned for the 2px-line pop-art style.

7. **Combined strategy weighting**
   - `combined.py` currently does binary OR. Try weighted alpha-blending of the
     two masks at different ratios.

---

## Agent Instructions

You are an autonomous research agent improving the coloring-book template pipeline
in this repository. You operate in a tight loop: propose a code change → run the
trial → inspect the score → commit or revert → repeat. You MUST NEVER pause,
ask for confirmation, or wait for human input.

### Repository layout

- `src/coloring_template/` — ALL production code. You may edit any file here.
- `examples/*.png`, `examples/*.jpg` — Fixed test inputs. Do NOT modify.
- `experiments/` — Research infrastructure. Do NOT modify any file here.
- `experiments/gold_standard/` — Reference PNGs showing the target aesthetic.
  Read these to understand what "good" looks like. Do not modify.
- `research/auto_research.py` — Prior benchmarking run. Read-only reference.

### What you may and may NOT edit

ALLOWED — any file under `src/coloring_template/`:
  - pipeline.py, cleanup.py
  - strategies/dark_pixel.py, strategies/edge_detect.py
  - strategies/combined.py, strategies/base.py, strategies/__init__.py
  - New .py files you add inside src/coloring_template/ or its subdirectories

FORBIDDEN — anything outside src/coloring_template/:
  - experiments/, examples/, research/, pyproject.toml, requirements.txt

### Evaluation command

    python experiments/run_trial.py

Processes all 4 example images through the full pipeline and asks Claude Haiku
to score each output against the gold-standard exemplars. Prints a summary table
to stderr. Prints exactly one line to stdout:

    score: X.XX

where X.XX is the mean judge score (0.00–10.00). Higher is better. Parse that
line to get the numeric score.

### Git workflow

After every trial, either KEEP or DISCARD:

KEEP (score improved or equal to the previous best):
  1. git add -u src/coloring_template/
  2. git commit -m "<short description> (score: X.XX)"
  3. Append one TSV row to experiments/results.tsv (status=keep)

DISCARD (score regressed):
  1. git reset --hard HEAD
  2. Append one TSV row to experiments/results.tsv (status=discard)

CRASH (run_trial.py exited non-zero or produced no score line):
  1. git reset --hard HEAD
  2. Append one TSV row (status=crash, score=N/A, description=error summary)
  3. Read the error, fix it before trying the next change

### results.tsv row format

Tab-separated, append one row per trial:

  <git_short_hash>  <score>  <status>  <description>  <ISO8601_UTC_timestamp>

- git_short_hash: `git rev-parse --short HEAD`
- score: float to 2 decimal places, or "N/A" for crash
- status: keep | discard | crash
- description: one sentence
- timestamp: e.g. 2026-04-25T22:00:00Z

### Scoring target

Aim for 8+. The scale:
  - 0–3  Broken (all black, all white, or severe noise)
  - 4–5  Partial outlines with heavy noise or broken regions
  - 6–7  Recognizable template but with gaps or excess texture
  - 8–9  Professional — suitable for iPad coloring apps
  - 10   Perfect — indistinguishable from the gold-standard exemplars

Gold-standard target aesthetic: clean 1-2px black outlines, all major regions
fully enclosed (flood-fill without leaking), minimal noise blobs, outline density
roughly 2-8% of all pixels.

### Hard constraints

- NEVER run `git push`
- NEVER modify any file under `experiments/`
- NEVER run `pip install`, `conda install`, or any package manager
- NEVER run `rm -rf`
- ALWAYS run the trial before committing
- ALWAYS append to results.tsv after every trial
- Make ONE logical change per trial so causality is clear
- The loop runs indefinitely — stop only if explicitly interrupted
