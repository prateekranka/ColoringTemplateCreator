#!/usr/bin/env python3
"""Headless Karpathy-style autoresearch agent loop.

Loads autoresearch_250426.md as the system prompt, gives a Claude Opus agent
bash/read_file/write_file tools, and runs a multi-turn agentic loop indefinitely.

Usage:
    ANTHROPIC_API_KEY=sk-... python experiments/loop.py
    ANTHROPIC_API_KEY=sk-... python experiments/loop.py --max-iterations 5
"""

import argparse
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import anthropic

ROOT             = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR  = ROOT / "experiments"
PROGRAM_MD       = EXPERIMENTS_DIR / "autoresearch_250426.md"
SRC_ALLOWED_ROOT = ROOT / "src" / "coloring_template"

AGENT_MODEL      = "claude-opus-4-7"
BASH_TIMEOUT_SEC = 120
MAX_BASH_OUTPUT  = 20_000

BASH_BLOCKLIST = [
    r"\brm\s+-rf\b",
    r"\bgit\s+push\b",
    r"\bpip\s+install\b",
    r"\bconda\s+install\b",
]
# Git commands may reference any path including experiments/
BASH_GIT_EXCEPTION = re.compile(r"\bgit\b")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def tool_bash(command: str) -> str:
    for pattern in BASH_BLOCKLIST:
        if re.search(pattern, command):
            if not BASH_GIT_EXCEPTION.search(command):
                return f"PermissionError: blocked by policy (matched {pattern!r})"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=BASH_TIMEOUT_SEC,
            cwd=str(ROOT),
        )
        output = result.stdout + result.stderr
        if len(output) > MAX_BASH_OUTPUT:
            output = output[:MAX_BASH_OUTPUT] + f"\n[...truncated at {MAX_BASH_OUTPUT} chars]"
        return output
    except subprocess.TimeoutExpired:
        return f"Error: timed out after {BASH_TIMEOUT_SEC}s"
    except Exception as exc:
        return f"Error: {exc}"


def tool_read_file(path: str) -> str:
    p = Path(path) if Path(path).is_absolute() else ROOT / path
    try:
        return p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except Exception as exc:
        return f"Error: {exc}"


def tool_write_file(path: str, content: str) -> str:
    p = Path(path) if Path(path).is_absolute() else ROOT / path
    try:
        resolved = p.resolve()
    except Exception:
        resolved = p

    allowed = SRC_ALLOWED_ROOT.resolve()
    if not str(resolved).startswith(str(allowed)):
        return (
            f"PermissionError: write_file only allowed under src/coloring_template/. "
            f"Attempted: {path}"
        )
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} chars to {path}"
    except Exception as exc:
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Tool definitions for the API
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "bash",
        "description": (
            "Execute a bash command in the project root. 120s timeout. "
            "stdout and stderr are returned combined. "
            "Blocked: rm -rf, git push, pip install, conda install."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the full text of any file in the repository.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Absolute or repo-relative path."}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Overwrite a file. ONLY files under src/coloring_template/ are permitted. "
            "Use to edit strategy or pipeline source code."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Must be under src/coloring_template/."},
                "content": {"type": "string", "description": "Full new file content."},
            },
            "required": ["path", "content"],
        },
    },
]


def dispatch_tool(name: str, inputs: dict) -> str:
    if name == "bash":
        return tool_bash(inputs["command"])
    elif name == "read_file":
        return tool_read_file(inputs["path"])
    elif name == "write_file":
        return tool_write_file(inputs["path"], inputs["content"])
    return f"Error: unknown tool {name!r}"


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------

def extract_score(text: str) -> float | None:
    match = re.search(r'^score:\s*([0-9]+(?:\.[0-9]+)?)$', text, re.MULTILINE)
    return float(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# API call with exponential backoff
# ---------------------------------------------------------------------------

def api_call(client: anthropic.Anthropic, **kwargs) -> anthropic.types.Message:
    delay = 5.0
    for attempt in range(6):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError:
            if attempt == 5:
                raise
            print(f"  [backoff] rate limit, waiting {delay:.0f}s...", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 120)
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500 and attempt < 5:
                print(f"  [backoff] server error {exc.status_code}, waiting {delay:.0f}s...", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 120)
            else:
                raise
    raise RuntimeError("Exceeded maximum retry attempts")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_loop(max_iterations: int) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    system_text = PROGRAM_MD.read_text(encoding="utf-8")
    client = anthropic.Anthropic(api_key=api_key)

    system = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                "Begin the autoresearch loop exactly as described in autoresearch_250426.md. "
                "Start by running the baseline: `python experiments/run_trial.py`. "
                "Record the baseline score in results.tsv, then begin proposing improvements."
            ),
        }
    ]

    iteration = 0

    while True:
        iteration += 1
        if max_iterations and iteration > max_iterations:
            print(f"\n[loop] Reached max_iterations={max_iterations}. Stopping.")
            break

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"\n{'='*60}", flush=True)
        print(f"[loop] Iteration {iteration}  {ts}", flush=True)
        print(f"{'='*60}", flush=True)

        try:
            response = api_call(
                client,
                model=AGENT_MODEL,
                max_tokens=8192,
                system=system,
                messages=messages,
                tools=TOOLS,
            )
        except Exception as exc:
            print(f"[loop] FATAL API error: {exc}", flush=True)
            traceback.print_exc()
            break

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text") and block.text:
                    print(f"[agent] {block.text}", flush=True)
            print("[loop] Agent reached end_turn. Stopping.", flush=True)
            break

        if response.stop_reason != "tool_use":
            print(f"[loop] Unexpected stop_reason: {response.stop_reason!r}. Stopping.", flush=True)
            break

        tool_results: list[dict] = []

        for block in response.content:
            if block.type == "text" and block.text:
                print(f"[agent] {block.text}", flush=True)
            elif block.type == "tool_use":
                print(f"[tool] {block.name}({list(block.input.keys())})", flush=True)
                if block.name == "bash":
                    cmd_preview = block.input.get("command", "")[:200]
                    print(f"  $ {cmd_preview}", flush=True)

                result_text = dispatch_tool(block.name, block.input)

                if block.name == "bash":
                    score = extract_score(result_text)
                    if score is not None:
                        print(f"[loop] Score detected: {score:.2f}", flush=True)

                preview = result_text[:500] + ("..." if len(result_text) > 500 else "")
                print(f"  -> {preview}", flush=True)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Headless autoresearch agent loop")
    p.add_argument("--max-iterations", type=int, default=0,
                   help="Stop after N iterations (0 = run forever)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_loop(max_iterations=args.max_iterations)
