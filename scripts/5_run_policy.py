"""
Script 5 — Dry-run the Hedge policy over Kalman output.

Reads all Kalman state Parquet files from data/derived/kalman/,
builds a PolicyContext for each tick, and runs action selection.

This is a DRY RUN: no losses are fed back and weights are never updated.
The policy starts at uniform weights (1/3 each action) and stays there.
Every output action is a random draw from that uniform distribution.

Why run this at all?
  - Validates the full chain: Kalman output → PolicyContext → HedgePolicy → action log.
  - Confirms the action log Parquet schema is correct before wiring in the simulator.
  - The action log can be inspected to verify the selection frequency is ~1/3 per action
    (law of large numbers check on the RNG).

The real behavior — weights shifting as losses arrive — happens in script 6.

Run: python3 scripts/5_run_policy.py
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List

import pyarrow as pa

from src.policy.hedge import HedgePolicy, PolicyConfig, PolicyContext
from src.utils.config import load_config
from src.utils.io import iter_parquet_dir, write_parquet
from src.utils.logging import get_logger
from src.utils.time_utils import hour_partition

log = get_logger("5_run_policy")

KALMAN_DIR  = Path("data/derived/kalman")
OUTPUT_DIR  = Path("data/derived/policy")

DECISION_SCHEMA = pa.schema([
    ("ts_ms",            pa.int64()),
    ("action",           pa.string()),
    ("prob_wait",        pa.float64()),
    ("prob_passive",     pa.float64()),
    ("prob_aggressive",  pa.float64()),
    ("weight_wait",      pa.float64()),
    ("weight_passive",   pa.float64()),
    ("weight_aggressive",pa.float64()),
])


def main() -> None:
    cfg    = load_config("configs/policy.yaml", PolicyConfig)
    policy = HedgePolicy(cfg, seed=42)

    log.info("Policy config: learning_rate=%.3f (dry run — no weight updates)",
             cfg.learning_rate)

    batches: dict[tuple[str, str], List[dict]] = {}
    total = 0

    for row in iter_parquet_dir(KALMAN_DIR):
        ctx = PolicyContext(
            ts_ms=row["ts_ms"],
            market_pressure=row["market_pressure"],
            regime=row["regime"],
        )
        decision = policy.select(ctx)

        date_str, hour_str = hour_partition(decision.ts_ms)
        key = (date_str, hour_str)
        batches.setdefault(key, []).append(asdict(decision))
        total += 1

    log.info("Selected actions for %d ticks.", total)

    for (date_str, hour_str), records in batches.items():
        path = OUTPUT_DIR / f"date={date_str}" / f"hour={hour_str}" / "decisions.parquet"
        write_parquet(records, DECISION_SCHEMA, path)
        log.info("Wrote %d decisions → %s", len(records), path)

    # Summary: count each action to verify ~uniform distribution
    if total > 0:
        all_records = [r for batch in batches.values() for r in batch]
        for action in ("WAIT", "PASSIVE", "AGGRESSIVE"):
            count = sum(1 for r in all_records if r["action"] == action)
            log.info("  %-12s %5d  (%.1f%%)", action, count, 100 * count / total)

    log.info("Dry run complete. Run script 6 to wire in losses and enable weight updates.")


if __name__ == "__main__":
    main()
