#!/usr/bin/env bash
# Autoresearch overnight runner — Claude Code CLI edition.
#
# Run unattended (e.g. before bed) to let the agent hill-climb the pipeline
# while you sleep. Uses loop_claude_code.py so no ANTHROPIC_API_KEY is needed.
#
# ── One-time setup ───────────────────────────────────────────────────────────
# Make executable:
#   chmod +x experiments/nightly_loop.sh
#
# Run manually:
#   experiments/nightly_loop.sh
#   NIGHTLY_MAX_TURNS=50 experiments/nightly_loop.sh
#
# Schedule with cron (runs at 23:00 every night):
#   crontab -e
#   # Add this line (adjust path):
#   0 23 * * * /path/to/ColoringTemplateCreator/experiments/nightly_loop.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/experiments/logs"
LOG="$LOG_DIR/nightly_$(date +%Y%m%d_%H%M%S).log"
MAX_TURNS="${NIGHTLY_MAX_TURNS:-30}"

mkdir -p "$LOG_DIR"

echo "============================================" | tee "$LOG"
echo "Nightly autoresearch — $(date)"              | tee -a "$LOG"
echo "Max turns : $MAX_TURNS"                      | tee -a "$LOG"
echo "Log       : $LOG"                            | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

cd "$ROOT"
python experiments/loop_claude_code.py --max-turns "$MAX_TURNS" 2>&1 | tee -a "$LOG"

echo "============================================" | tee -a "$LOG"
echo "Finished — $(date)"                          | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
