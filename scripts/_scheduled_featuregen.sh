#!/usr/bin/env bash
# Internal — invoked by launchd (com.damianmunoz.research.featuregen.plist)
# every 8 hours (00, 08, 16 local time). Runs scripts/3_featuregen.py from
# the repo root with the project venv activated, appending stdout/stderr to
# data/logs/featuregen.log.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

source .venv/bin/activate

mkdir -p data/logs

{
    echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') featuregen tick ====="
    PYTHONPATH=. python scripts/3_featuregen.py
    echo "===== done in ${SECONDS}s ====="
} >> data/logs/featuregen.log 2>&1
