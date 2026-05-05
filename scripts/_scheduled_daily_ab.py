"""
scripts/_scheduled_daily_ab.py — internal, invoked by launchd.

Each invocation:
  1. Loads every feature row from data/derived/features/.
  2. Detects all sessions (same logic as scripts/6_*).
  3. For each session, runs both 1D bucketed and 2D bucketed Hedge with
     IDENTICAL configs (same η, λ, pressure_edges, regime_edges) across
     SEEDS = [0..4], paired by seed.
  4. Writes ONE flat JSON to data/derived/daily_ab/results.json with one
     row per (session × seed × mode). Atomic overwrite each run.

For interactive use, prefer scripts/6_simulate_and_update.py /
6_simulate_and_update_2d.py — those write a per-tick execution log and
print the bucket summary. This script is for unattended overnight runs.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.execution.loss import compute_loss
from src.execution.simulator import simulate_fill
from src.execution.updater import ExecutionConfig
from src.policy.actions import Action
from src.policy.hedge import PolicyConfig, PolicyContext, make_policy
from src.state.kalman_filter import KalmanConfig, KalmanFilter
from src.utils.config import load_config
from src.utils.io import iter_parquet_dir

# Silence the per-tick loggers from src.* — this script runs unattended.
logging.disable(logging.CRITICAL)

FEATURES_DIR   = Path("data/derived/features")
OUTPUT_PATH    = Path("data/derived/daily_ab/results.json")
SESSION_GAP_MS = 5_000
SEEDS          = [0, 1, 2, 3, 4]
MIN_ROWS       = 100  # skip very short sessions


def detect_sessions(rows):
    if not rows:
        return []
    out = []
    start = 0
    for i in range(1, len(rows)):
        tg = rows[i]["ts_ms"] - rows[i - 1]["ts_ms"]
        if rows[i].get("sequence_gap") or tg > SESSION_GAP_MS:
            out.append((start, i))
            start = i
    out.append((start, len(rows)))
    return out


def replay(rows, kalman, policy, lambda_, obs_features):
    n = 0
    n_filled = 0
    sum_slip = 0.0
    sum_adv  = 0.0
    sum_loss = 0.0
    n_act = {Action.WAIT: 0, Action.PASSIVE: 0, Action.AGGRESSIVE: 0}
    prev = None
    for row in rows:
        if prev is None:
            prev = row
            continue
        curr, nxt = prev, row
        z = np.array([curr[f] for f in obs_features], dtype=float)
        ks = kalman.step(curr["ts_ms"], z)
        if ks is None:
            prev = row
            continue
        ctx = PolicyContext(curr["ts_ms"], ks.market_pressure, ks.regime)
        d = policy.select(ctx)
        a = Action(d.action)
        f = simulate_fill(
            action=a, ts_ms=curr["ts_ms"],
            curr_mid=curr["mid_price"],
            curr_best_bid=curr["best_bid_px"],
            curr_best_ask=curr["best_ask_px"],
            next_mid=nxt["mid_price"],
        )
        L = compute_loss(f, lambda_)
        policy.update(a, L)
        slip = (f.fill_price - f.mid_price) if f.filled else 0.0
        adv = (max(0.0, f.fill_price - f.next_mid_price) if f.filled
               else max(0.0, f.next_mid_price - f.mid_price))
        n += 1
        if f.filled: n_filled += 1
        sum_slip += slip
        sum_adv  += adv
        sum_loss += L
        n_act[a] += 1
        prev = row
    if n == 0:
        return None
    return {
        "n": n, "n_filled": n_filled,
        "fill_rate":      n_filled / n,
        "total_slippage": sum_slip,
        "total_adverse":  sum_adv,
        "total_loss":     sum_loss,
        "wait_pct":       100 * n_act[Action.WAIT]       / n,
        "passive_pct":    100 * n_act[Action.PASSIVE]    / n,
        "aggressive_pct": 100 * n_act[Action.AGGRESSIVE] / n,
    }


def _iso(ts_ms): return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


def main():
    print(f"[{datetime.now(timezone.utc).isoformat()}] daily_ab START", flush=True)

    kalman_cfg    = load_config("configs/kalman.yaml",    KalmanConfig)
    policy_cfg    = load_config("configs/policy.yaml",    PolicyConfig)
    execution_cfg = load_config("configs/execution.yaml", ExecutionConfig)
    obs_features  = list(kalman_cfg.obs_features)

    if not policy_cfg.regime_edges:
        raise SystemExit("policy.yaml is missing regime_edges — cannot run 2D mode.")

    all_rows = list(iter_parquet_dir(FEATURES_DIR))
    if not all_rows:
        print("No features under data/derived/features — nothing to do.", flush=True)
        return

    sessions = detect_sessions(all_rows)
    print(f"Detected {len(sessions)} session(s).", flush=True)

    results = []
    for s_idx, (a, b) in enumerate(sessions, start=1):
        sub = all_rows[a:b]
        if len(sub) < MIN_ROWS:
            print(f"  skip session {s_idx}: only {len(sub)} rows", flush=True)
            continue
        start_ts = sub[0]["ts_ms"]
        end_ts   = sub[-1]["ts_ms"]
        dur_h    = (end_ts - start_ts) / 3_600_000
        print(f"  session {s_idx}: {len(sub):,} rows, {dur_h:.2f}h, "
              f"{_iso(start_ts)} → {_iso(end_ts)}", flush=True)

        for seed in SEEDS:
            for mode_label, mode_str in [("1D", "bucketed"), ("2D", "bucketed_2d")]:
                pcfg = copy.deepcopy(policy_cfg)
                pcfg.mode = mode_str
                kalman = KalmanFilter(kalman_cfg)
                policy = make_policy(pcfg, seed=seed)
                m = replay(sub, kalman, policy, execution_cfg.lambda_, obs_features)
                if m is None:
                    continue
                results.append({
                    "session_idx":         s_idx,
                    "session_start_ts_ms": start_ts,
                    "session_start_iso":   _iso(start_ts),
                    "session_end_iso":     _iso(end_ts),
                    "session_n_rows":      len(sub),
                    "session_duration_h":  round(dur_h, 4),
                    "seed":           seed,
                    "mode":           mode_label,
                    "n_ticks":        m["n"],
                    "n_filled":       m["n_filled"],
                    "fill_rate":      m["fill_rate"],
                    "total_slippage": m["total_slippage"],
                    "total_adverse":  m["total_adverse"],
                    "total_loss":     m["total_loss"],
                    "wait_pct":       m["wait_pct"],
                    "passive_pct":    m["passive_pct"],
                    "aggressive_pct": m["aggressive_pct"],
                })

    out = {
        "ran_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "lambda":              execution_cfg.lambda_,
            "eta":                 policy_cfg.learning_rate,
            "pressure_edges":      list(policy_cfg.pressure_edges),
            "regime_edges":        list(policy_cfg.regime_edges),
            "seeds":               SEEDS,
            "kalman_obs_features": obs_features,
        },
        "n_sessions": len(sessions),
        "n_results":  len(results),
        "results":    results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    tmp.replace(OUTPUT_PATH)

    print(f"Wrote {len(results)} rows from {out['n_sessions']} session(s) → {OUTPUT_PATH}",
          flush=True)
    print(f"[{datetime.now(timezone.utc).isoformat()}] daily_ab END", flush=True)


if __name__ == "__main__":
    main()
