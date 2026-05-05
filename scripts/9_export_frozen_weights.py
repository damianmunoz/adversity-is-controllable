"""
scripts/9_export_frozen_weights.py — one-off exporter for the GUI.

Runs the existing 1D bucketed Hedge over a chosen session with a fixed seed,
then dumps the per-bucket final weights to JSON. The GUI loads that JSON and
runs the same policy in inference-only mode (FrozenBucketedPolicy) so the
"live" decisions match what the offline run learned.

Usage:
    PYTHONPATH=. python scripts/9_export_frozen_weights.py --session 7 --seed 0

Output:
    data/derived/frozen_weights/1d_session<N>_seed<S>.json

This script does NOT modify any existing src module. It re-uses
KalmanFilter, BucketedHedgePolicy, simulate_fill, compute_loss, and the
session detection from scripts/6_simulate_and_update.py — and just calls
policy.bucket_summary() at the end to dump the weights.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.execution.loss import compute_loss
from src.execution.simulator import simulate_fill
from src.execution.updater import ExecutionConfig
from src.policy.actions import Action
from src.policy.hedge import PolicyConfig, make_policy
from src.state.kalman_filter import KalmanConfig, KalmanFilter
from src.utils.config import load_config
from src.utils.io import iter_parquet_dir
from src.utils.logging import get_logger

log = get_logger("9_export_frozen_weights")

FEATURES_DIR = Path("data/derived/features")
OUT_DIR      = Path("data/derived/frozen_weights")
SESSION_GAP_THRESHOLD_MS = 5_000


def _detect(rows) -> List[Tuple[int, int]]:
    if not rows:
        return []
    out: List[Tuple[int, int]] = []
    start = 0
    for i in range(1, len(rows)):
        gap = rows[i]["ts_ms"] - rows[i - 1]["ts_ms"]
        if rows[i].get("sequence_gap") or gap > SESSION_GAP_THRESHOLD_MS:
            out.append((start, i)); start = i
    out.append((start, len(rows)))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", type=int, default=None,
                   help="1-indexed session (default: longest)")
    p.add_argument("--seed",    type=int, default=0,
                   help="RNG seed for the policy (default: 0)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    kalman_cfg    = load_config("configs/kalman.yaml",    KalmanConfig)
    policy_cfg    = load_config("configs/policy.yaml",    PolicyConfig)
    execution_cfg = load_config("configs/execution.yaml", ExecutionConfig)

    if policy_cfg.mode != "bucketed":
        log.warning("policy.yaml mode=%s — forcing 'bucketed' for export.", policy_cfg.mode)
        policy_cfg.mode = "bucketed"

    rows = list(iter_parquet_dir(FEATURES_DIR))
    sessions = _detect(rows)
    if args.session is None:
        idx = max(range(len(sessions)), key=lambda i: sessions[i][1] - sessions[i][0])
        sess_idx = idx + 1
    else:
        sess_idx = args.session
    a, b = sessions[sess_idx - 1]
    selected = rows[a:b]
    log.info(
        "Session %d: %d ticks, %.2f h",
        sess_idx, len(selected),
        (selected[-1]["ts_ms"] - selected[0]["ts_ms"]) / 3_600_000.0,
    )

    kalman = KalmanFilter(kalman_cfg)
    policy = make_policy(policy_cfg, seed=args.seed)
    obs    = kalman_cfg.obs_features
    lam    = execution_cfg.lambda_

    total_loss = 0.0
    total_slip = 0.0
    total_adv  = 0.0
    n_filled   = 0
    n_ticks    = 0

    prev = None
    for row in selected:
        if prev is None:
            prev = row; continue
        z = np.array([prev[f] for f in obs], dtype=float)
        ks = kalman.step(prev["ts_ms"], z)
        if ks is None:
            prev = row; continue
        from src.policy.hedge import PolicyContext
        ctx = PolicyContext(prev["ts_ms"], ks.market_pressure, ks.regime)
        decision = policy.select(ctx)
        action   = Action(decision.action)
        fill = simulate_fill(
            action, prev["ts_ms"],
            prev["mid_price"], prev["best_bid_px"], prev["best_ask_px"],
            row["mid_price"],
        )
        loss = compute_loss(fill, lam)
        policy.update(action, loss)

        total_loss += loss
        slip = (fill.fill_price - fill.mid_price) if fill.filled else 0.0
        adv  = (max(0.0, fill.fill_price - fill.next_mid_price) if fill.filled
                else max(0.0, fill.next_mid_price - fill.mid_price))
        total_slip += slip
        total_adv  += adv
        n_filled  += 1 if fill.filled else 0
        n_ticks   += 1
        prev = row

    log.info(
        "Done — ticks=%d  fills=%d  slip=%.4f  adv=%.4f  loss=%.4f",
        n_ticks, n_filled, total_slip, total_adv, total_loss,
    )

    # Dump bucket summary
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"1d_session{sess_idx}_seed{args.seed}.json"

    payload = {
        "source":         "scripts/9_export_frozen_weights.py",
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_idx":    sess_idx,
        "seed":           args.seed,
        "ticks":          n_ticks,
        "lambda_":        lam,
        "learning_rate":  policy_cfg.learning_rate,
        "pressure_edges": list(policy_cfg.pressure_edges),
        "training_metrics": {
            "total_loss":    total_loss,
            "total_slippage": total_slip,
            "total_adverse": total_adv,
            "fill_rate":     n_filled / max(1, n_ticks),
        },
        "buckets": [],
    }
    for b in policy.bucket_summary():
        # Convert Action enum keys to strings; resolve "(-inf, +0.500]" etc.
        rng = b["range"]   # "(<lo>, <hi>]"
        lo_s, hi_s = rng.strip("(]").split(", ")
        def _parse(s: str) -> float:
            if s == "-inf": return -math.inf
            if s == "+inf": return  math.inf
            return float(s)
        lo, hi = _parse(lo_s), _parse(hi_s)
        payload["buckets"].append({
            "bucket":  b["bucket"],
            "lo":      lo if math.isfinite(lo) else (-1e18 if lo < 0 else +1e18),
            "hi":      hi if math.isfinite(hi) else (-1e18 if hi < 0 else +1e18),
            "visits":  b["visits"],
            "weights": {
                "WAIT":       b["weights"][Action.WAIT],
                "PASSIVE":    b["weights"][Action.PASSIVE],
                "AGGRESSIVE": b["weights"][Action.AGGRESSIVE],
            },
        })

    out_path.write_text(json.dumps(payload, indent=2))
    log.info("Wrote frozen weights → %s", out_path)


if __name__ == "__main__":
    main()
