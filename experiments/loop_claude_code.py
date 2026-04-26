#!/usr/bin/env python3
"""Headless autoresearch agent loop — Claude Code CLI edition.

Drop-in replacement for loop.py that uses the `claude` CLI subprocess instead
of the Anthropic API directly. Works without ANTHROPIC_API_KEY; uses the
Claude.ai subscription authenticated in your Claude Code CLI installation.

Each outer iteration calls `claude -p` once to run ONE research trial (propose
a change → run_trial_claude_code.py → score → commit/revert → append to
results.tsv). The outer Python loop drives the iteration count, mirroring the
structure of loop.py.

Usage:
    python experiments/loop_claude_code.py
    python experiments/loop_claude_code.py --max-iterations 900
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT            = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"
RESULTS_TSV     = EXPERIMENTS_DIR / "results.tsv"
PROGRAM_MD      = EXPERIMENTS_DIR / "autoresearch_250426.md"


def check_claude_cli() -> None:
    result = subprocess.run(["claude", "--version"], capture_output=True)
    if result.returncode != 0:
        sys.exit(
            "ERROR: `claude` CLI not found.\n"
            "Install Claude Code and log in: https://claude.ai/code"
        )


def read_recent_results(n: int = 10) -> str:
    """Return the last n rows of results.tsv as context for the agent."""
    if not RESULTS_TSV.exists():
        return "(no results yet)"
    lines = RESULTS_TSV.read_text().splitlines()
    recent = lines[-n:] if len(lines) >= n else lines
    return "\n".join(recent) if recent else "(no results yet)"


def run_one_iteration(iteration: int, max_iterations: int) -> bool:
    """Run one research trial via the claude CLI.

    Returns True if the subprocess exited cleanly, False on error.
    """
    recent = read_recent_results()
    remaining = max_iterations - iteration + 1 if max_iterations else "unlimited"

    task = (
        "Read experiments/autoresearch_250426.md for your full agent instructions.\n\n"
        "IMPORTANT CHANGE: use `python experiments/run_trial_claude_code.py` as the "
        "evaluation command (not run_trial.py). This version uses the Claude Code CLI "
        "for judging — no ANTHROPIC_API_KEY needed.\n\n"
        f"This is iteration {iteration}"
        + (f" of {max_iterations}" if max_iterations else "")
        + f". Remaining iterations: {remaining}.\n\n"
        "Recent results.tsv rows (latest at bottom):\n"
        f"{recent}\n\n"
        "Run EXACTLY ONE trial:\n"
        "  1. Propose one code change based on the history above\n"
        "  2. Apply it to src/coloring_template/\n"
        "  3. Run: python experiments/run_trial_claude_code.py\n"
        "  4. Parse the score from stdout\n"
        "  5. KEEP (score improved) → git add -u src/ && git commit -m '...(score: X.XX)'\n"
        "     DISCARD (score regressed) → git reset --hard HEAD\n"
        "  6. Append one TSV row to experiments/results.tsv\n"
        "  7. Stop — do NOT run another trial.\n\n"
        "All hard constraints from autoresearch_250426.md apply: no git push, "
        "no rm -rf, no pip install, only edit src/coloring_template/."
    )

    cmd = [
        "claude",
        "-p", task,
        "--allowedTools", "Bash,Read,Write",
        "--permission-mode", "bypassPermissions",
    ]

    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode == 0


def run_loop(max_iterations: int) -> None:
    check_claude_cli()

    label = f"{max_iterations} iterations" if max_iterations else "unlimited"
    print(f"Starting autoresearch via Claude Code CLI ({label})")
    print("Press Ctrl+C to stop.\n", flush=True)

    iteration = 0
    while True:
        iteration += 1
        if max_iterations and iteration > max_iterations:
            print(f"\n[loop] Reached max_iterations={max_iterations}. Stopping.")
            break

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"\n{'='*60}", flush=True)
        print(f"[loop] Iteration {iteration}"
              + (f"/{max_iterations}" if max_iterations else "")
              + f"  {ts}", flush=True)
        print(f"{'='*60}", flush=True)

        ok = run_one_iteration(iteration, max_iterations)
        if not ok:
            print(f"[loop] claude exited with error on iteration {iteration}. "
                  "Waiting 10s before retrying...", flush=True)
            time.sleep(10)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Autoresearch agent loop via Claude Code CLI (no API key needed)"
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        metavar="N",
        help="Stop after N iterations (0 = unlimited, default: 0)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_loop(args.max_iterations)
