# Gold Standard Reference Images

Place reference PNG screenshots of high-quality coloring book pages here.
These are used as visual exemplars by the LLM judge in `experiments/judge.py`.

Requirements:
- Format: PNG or JPG
- Style: black outlines on white background, similar to professional coloring books
- Content: any subject is fine (the style is what matters, not the subject)
- Up to 4 images are loaded per judge call; additional images are ignored

The autoresearch agent reads this directory but never modifies it.

## How to add gold standards

Save 1-4 screenshots from your coloring app (e.g. Colorflow) that show the
target template quality. Crop to just the template (no UI chrome). Name them
anything — e.g. `reference_01.png`, `reference_02.png`.

Then run a smoke test:

    ANTHROPIC_API_KEY=sk-... python experiments/run_trial.py
