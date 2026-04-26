#!/usr/bin/env python3
"""Headless autoresearch agent loop — Claude Code CLI edition.

Drop-in replacement for loop.py that uses the `claude` CLI subprocess instead
of the Anthropic API directly. Works without ANTHROPIC_API_KEY; uses the
Claude.ai subscription authenticated in your Claude Code CLI installation.

The `claude` process manages its own internal agentic tool-use loop (Bash,
Read, Write), so this script is much simpler than loop.py — it just launches
the agent and streams its output.

Usage:
    python experiments/loop_claude_code.py
    python experiments/loop_claude_code.py --max-turns 30
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT            = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"
PROGRAM_MD      = EXPERIMENTS_DIR / "autoresearch_250426.md"


def check_claude_cli() -> None:
    result = subprocess.run(["claude", "--version"], capture_output=True)
    if result.returncode != 0:
        sys.exit(
            "ERROR: `claude` CLI not found.\n"
            "Install Claude Code and log in: https://claude.ai/code"
        )


def run_loop(max_turns: int) -> None:
    check_claude_cli()

    task = (
        "Read experiments/autoresearch_250426.md for your full agent instructions.\n\n"
        "Then begin the autoresearch loop exactly as described in that file, with "
        "one important change:\n\n"
        "  Use `python experiments/run_trial_claude_code.py` as the evaluation "
        "command instead of `python experiments/run_trial.py`. This version uses "
        "the Claude Code CLI for judging, so no ANTHROPIC_API_KEY is needed.\n\n"
        "All other instructions in autoresearch_250426.md apply unchanged: git "
        "workflow, results.tsv format, hard constraints, etc.\n\n"
        "Run indefinitely. Do NOT pause, ask for input, or wait for confirmation."
    )

    cmd = [
        "claude",
        "-p", task,
        "--allowedTools", "Bash,Read,Write",
    ]
    if max_turns:
        cmd += ["--max-turns", str(max_turns)]

    label = f"max-turns={max_turns}" if max_turns else "unlimited"
    print(f"Starting autoresearch via Claude Code CLI ({label})")
    print("Press Ctrl+C to stop.\n", flush=True)

    subprocess.run(cmd, cwd=str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Autoresearch agent loop via Claude Code CLI (no API key needed)"
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=0,
        metavar="N",
        help="Stop after N agent turns (0 = unlimited, default: 0)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_loop(args.max_turns)
