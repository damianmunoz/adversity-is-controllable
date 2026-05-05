#!/usr/bin/env bash
# Internal — invoked by launchd (com.damianmunoz.research.daily-ab.plist)
# every morning at 09:00 local. Refreshes features and runs the paired
# 1D vs 2D A/B over every detected session. Output:
#   data/derived/daily_ab/results.json   (one row per session × seed × mode)
#   data/logs/daily_ab.log               (stdout/stderr)

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

source .venv/bin/activate

mkdir -p data/logs data/derived/daily_ab

{
    echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') daily_ab tick ====="
    echo "----- featuregen pre-pass -----"
    PYTHONPATH=. python scripts/3_featuregen.py
    echo "----- A/B replay -----"
    PYTHONPATH=. python scripts/_scheduled_daily_ab.py
    echo "===== done in ${SECONDS}s ====="
} >> data/logs/daily_ab.log 2>&1
