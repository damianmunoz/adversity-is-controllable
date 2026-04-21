"""
Outer loop — chains Kalman → policy → simulator → loss → weight update.

This is the core of the system. Every tick it does five things in sequence:

  1. Feed the current feature vector into the Kalman filter → hidden state estimate
  2. Build a PolicyContext from the Kalman state → what does the policy see?
  3. Ask the policy to select an action → WAIT / PASSIVE / AGGRESSIVE
  4. Simulate the fill using current + next tick's market data → FillResult
  5. Compute loss and feed it back → policy weights shift

After running over all feature data, the policy's weights encode what it learned:
which action minimizes execution cost in each market condition.

Every tick is written to an ExecutionRecord and logged to JSONL. This log is
the primary output for validation, charting, and presentation.

Why run Kalman inline here (not read from disk)?
  The Kalman filter is stateful — its estimate at tick t depends on every tick
  before it. Running it inline guarantees consistency and avoids the risk of
  the on-disk Kalman Parquet being stale or misaligned with features.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from src.execution.loss import compute_loss
from src.execution.simulator import FillResult, simulate_fill
from src.policy.actions import Action
from src.policy.hedge import HedgePolicy, PolicyContext
from src.state.kalman_filter import KalmanFilter
from src.utils.io import append_jsonl
from src.utils.logging import get_logger
from src.utils.time_utils import date_partition

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExecutionConfig:
    """Mirrors configs/execution.yaml."""
    lambda_:    float        # adverse move penalty weight in loss function
    output_dir: str = "data/derived/execution_log"


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

@dataclass
class ExecutionRecord:
    """One row in the execution log — everything that happened at tick t.

    Rich enough to reconstruct the policy's full decision context and score
    any baseline strategy in post-hoc analysis. All fields are plain scalars
    so asdict() writes cleanly to JSONL.

    Fields:
      ts_ms              — timestamp of the decision
      action             — what the policy chose
      filled             — whether an order was executed
      fill_price         — price paid (0.0 if not filled)
      mid_price          — mid price at decision time
      next_mid_price     — mid price one tick later
      slippage           — fill_price - mid_price (0.0 if not filled)
      adverse_move       — how much price moved against us after the decision
      loss               — total loss fed back into the Hedge algorithm
      market_pressure    — Kalman estimate at this tick
      regime             — Kalman estimate at this tick
      weight_wait        — policy weight for WAIT at decision time
      weight_passive     — policy weight for PASSIVE at decision time
      weight_aggressive  — policy weight for AGGRESSIVE at decision time
    """
    ts_ms:             int
    action:            str
    filled:            bool
    fill_price:        float
    mid_price:         float
    next_mid_price:    float
    slippage:          float
    adverse_move:      float
    loss:              float
    market_pressure:   float
    regime:            float
    weight_wait:       float
    weight_passive:    float
    weight_aggressive: float


# ---------------------------------------------------------------------------
# Outer loop
# ---------------------------------------------------------------------------

def run(
    feature_rows: Iterator[dict],
    kalman:       KalmanFilter,
    policy:       HedgePolicy,
    config:       ExecutionConfig,
    obs_features: List[str],
) -> int:
    """Run the full pipeline over a stream of feature rows.

    Consumes feature_rows in pairs (curr, next) so the simulator can access
    both the decision tick and the outcome tick. The first row is consumed
    as context but produces no output — we need a next tick to evaluate it.

    Args:
        feature_rows: iterator of dicts from iter_parquet_dir(FEATURES_DIR)
        kalman:       pre-initialized KalmanFilter (stateful, runs inline)
        policy:       pre-initialized HedgePolicy (stateful, weights update here)
        config:       ExecutionConfig loaded from configs/execution.yaml
        obs_features: list of feature names for Kalman observation vector
                      (from KalmanConfig.obs_features)

    Returns:
        Total number of ExecutionRecords written.
    """
    # Each invocation gets its own run_id so runs don't bleed into each other.
    # Output layout: <output_dir>/run=<UTC-YYYYMMDD-HHMMSS>/date=<YYYY-MM-DD>/log.jsonl
    # A sibling `latest` symlink always points to the newest run for convenience.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base_dir = Path(config.output_dir)
    out_dir = base_dir / f"run={run_id}"
    log.info("Execution log run_id=%s  output=%s", run_id, out_dir)

    total = 0
    prev_row: Optional[dict] = None

    for row in feature_rows:
        if prev_row is None:
            prev_row = row
            continue

        curr = prev_row
        nxt  = row

        # --- Step 1: Kalman filter ---
        z = np.array([curr[f] for f in obs_features], dtype=float)
        kalman_state = kalman.step(curr["ts_ms"], z)

        if kalman_state is None:
            prev_row = row
            continue

        # --- Step 2: Policy context ---
        ctx = PolicyContext(
            ts_ms=curr["ts_ms"],
            market_pressure=kalman_state.market_pressure,
            regime=kalman_state.regime,
        )

        # --- Step 3: Action selection ---
        decision = policy.select(ctx)
        action   = Action(decision.action)
        weights  = policy.weights()

        # --- Step 4: Simulate fill ---
        fill = simulate_fill(
            action=action,
            ts_ms=curr["ts_ms"],
            curr_mid=curr["mid_price"],
            curr_best_bid=curr["best_bid_px"],
            curr_best_ask=curr["best_ask_px"],
            next_mid=nxt["mid_price"],
        )

        # --- Step 5: Compute loss and update policy ---
        loss = compute_loss(fill, config.lambda_)
        policy.update(action, loss)

        # --- Log ---
        slippage     = (fill.fill_price - fill.mid_price) if fill.filled else 0.0
        adverse_move = (
            max(0.0, fill.fill_price - fill.next_mid_price) if fill.filled
            else max(0.0, fill.next_mid_price - fill.mid_price)
        )

        record = ExecutionRecord(
            ts_ms=curr["ts_ms"],
            action=decision.action,
            filled=fill.filled,
            fill_price=fill.fill_price,
            mid_price=fill.mid_price,
            next_mid_price=fill.next_mid_price,
            slippage=slippage,
            adverse_move=adverse_move,
            loss=loss,
            market_pressure=kalman_state.market_pressure,
            regime=kalman_state.regime,
            weight_wait=weights[Action.WAIT],
            weight_passive=weights[Action.PASSIVE],
            weight_aggressive=weights[Action.AGGRESSIVE],
        )

        date_str = date_partition(curr["ts_ms"])
        log_path = out_dir / f"date={date_str}" / "log.jsonl"
        append_jsonl(log_path, asdict(record))
        total += 1

        prev_row = row

    # Update the `latest` symlink only if at least one record was written.
    # Symlink target is relative (run=<id>) so the log dir stays portable.
    if total > 0:
        latest = base_dir / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        os.symlink(f"run={run_id}", latest)
        log.info("Updated 'latest' symlink → run=%s", run_id)

    return total
