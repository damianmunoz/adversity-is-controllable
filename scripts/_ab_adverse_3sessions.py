"""
scripts/_ab_adverse_3sessions.py — internal one-off A/B harness for §24.

For each of the 3 real sessions (Apr-21, Apr-28, Apr-30 main pre-gap)
and each of 10 seeds, runs 4 policies:

  1. 1D bucketed                              (baseline — current winner)
  2. adverse-2D with secondary = vol_delta
  3. adverse-2D with secondary = ofi_window
  4. adverse-2D with secondary = spread_delta

Identical config across all four policies (same η, λ, pressure_edges,
same Kalman filter). Paired by seed so a single seed determines the RNG
draws inside every policy that seed produces. Total runs = 3 × 10 × 4 = 120.

Output: data/derived/ab_adverse_3sessions_10seed.json
        — flat list of one row per (session × seed × mode), easy to
          load into pandas.DataFrame(j["results"]).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.execution.loss import compute_loss
from src.execution.simulator import simulate_fill
from src.execution.updater import ExecutionConfig
from src.features.adverse_signals import SIGNAL_REGISTRY
from src.policy.actions import Action
from src.policy.hedge import PolicyConfig, PolicyContext, make_policy
from src.policy.hedge_adverse import AdverseContext, BucketedHedgeAdversePolicy
from src.state.kalman_filter import KalmanConfig, KalmanFilter
from src.utils.config import load_config
from src.utils.io import iter_parquet_dir

logging.disable(logging.CRITICAL)

FEATURES_DIR   = Path("data/derived/features")
OUTPUT_PATH    = Path("data/derived/ab_adverse_3sessions_10seed.json")
SESSION_GAP_MS = 5_000
SEEDS          = list(range(10))
TARGET_SESSIONS = [1, 2, 3, 7]   # 3 historical sessions + new session 7 (Apr-30 23:00 → May-01 20:05, 21.08h, 75,885 ticks) for §25

# Same defaults as scripts/6_simulate_and_update_adverse.py
DEFAULT_SECONDARY_EDGES = {
    "vol_delta":    [-1.1e-5, +9.0e-6],
    "ofi_window":   [-2.0,    +1.9],
    "spread_delta": [0.0],
}
WINDOW_SEC = 60.0


def detect_sessions(rows):
    if not rows:
        return []
    out, start = [], 0
    for i in range(1, len(rows)):
        tg = rows[i]["ts_ms"] - rows[i - 1]["ts_ms"]
        if rows[i].get("sequence_gap") or tg > SESSION_GAP_MS:
            out.append((start, i)); start = i
    out.append((start, len(rows)))
    return out


def replay_baseline_1d(rows, kalman, policy, lambda_, obs_features):
    """1D bucketed baseline — uses canonical PolicyContext."""
    return _replay_core(rows, kalman, policy, lambda_, obs_features,
                        ctx_builder=lambda ts, ks, _row: PolicyContext(ts, ks.market_pressure, ks.regime))


def replay_adverse(rows, kalman, policy, signal, signal_input_key,
                   lambda_, obs_features):
    """Adverse 2D — uses AdverseContext built from a stateful signal."""
    def builder(ts, ks, row):
        sec = signal.update(ts, row[signal_input_key])
        return AdverseContext(ts, ks.market_pressure, sec)
    return _replay_core(rows, kalman, policy, lambda_, obs_features, ctx_builder=builder)


def _replay_core(rows, kalman, policy, lambda_, obs_features, ctx_builder):
    n = 0; n_filled = 0
    sum_slip = 0.0; sum_adv = 0.0; sum_loss = 0.0
    n_act = {Action.WAIT: 0, Action.PASSIVE: 0, Action.AGGRESSIVE: 0}
    prev = None
    for row in rows:
        if prev is None:
            prev = row; continue
        curr, nxt = prev, row
        z = np.array([curr[f] for f in obs_features], dtype=float)
        ks = kalman.step(curr["ts_ms"], z)
        if ks is None:
            prev = row; continue

        ctx = ctx_builder(curr["ts_ms"], ks, curr)
        d = policy.select(ctx)
        a = Action(d.action)
        f = simulate_fill(action=a, ts_ms=curr["ts_ms"],
                          curr_mid=curr["mid_price"],
                          curr_best_bid=curr["best_bid_px"],
                          curr_best_ask=curr["best_ask_px"],
                          next_mid=nxt["mid_price"])
        L = compute_loss(f, lambda_)
        policy.update(a, L)

        slip = (f.fill_price - f.mid_price) if f.filled else 0.0
        adv  = (max(0.0, f.fill_price - f.next_mid_price) if f.filled
                else max(0.0, f.next_mid_price - f.mid_price))
        n += 1
        if f.filled: n_filled += 1
        sum_slip += slip; sum_adv += adv; sum_loss += L
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
    t0 = time.time()
    print(f"[{datetime.now(timezone.utc).isoformat()}] ab_adverse_3sessions START", flush=True)

    kalman_cfg    = load_config("configs/kalman.yaml",    KalmanConfig)
    policy_cfg    = load_config("configs/policy.yaml",    PolicyConfig)
    execution_cfg = load_config("configs/execution.yaml", ExecutionConfig)
    obs_features  = list(kalman_cfg.obs_features)

    all_rows = list(iter_parquet_dir(FEATURES_DIR))
    sessions = detect_sessions(all_rows)
    print(f"Detected {len(sessions)} session(s).", flush=True)

    results = []
    for s_idx in TARGET_SESSIONS:
        if s_idx > len(sessions):
            print(f"  skip session {s_idx}: not detected", flush=True)
            continue
        a, b = sessions[s_idx - 1]
        sub = all_rows[a:b]
        start_ts, end_ts = sub[0]["ts_ms"], sub[-1]["ts_ms"]
        dur_h = (end_ts - start_ts) / 3_600_000
        print(f"  session {s_idx}: {len(sub):,} rows, {dur_h:.2f}h, "
              f"{_iso(start_ts)} → {_iso(end_ts)}", flush=True)

        for seed in SEEDS:
            # ---- 1D baseline ----
            t_run = time.time()
            pcfg = PolicyConfig(
                learning_rate  = policy_cfg.learning_rate,
                initial_weight = policy_cfg.initial_weight,
                mode           = "bucketed",
                pressure_edges = list(policy_cfg.pressure_edges),
                regime_edges   = [],   # ignored in 1D
            )
            kalman = KalmanFilter(kalman_cfg)
            policy = make_policy(pcfg, seed=seed)
            m = replay_baseline_1d(sub, kalman, policy, execution_cfg.lambda_, obs_features)
            results.append(_record(s_idx, start_ts, end_ts, len(sub), dur_h,
                                   seed, "1D_baseline", m))
            print(f"    seed={seed} 1D_baseline    loss={m['total_loss']:.2f}  "
                  f"slip={m['total_slippage']:.2f}  adv={m['total_adverse']:.2f}  "
                  f"({time.time()-t_run:.1f}s)", flush=True)

            # ---- adverse 2D variants ----
            for sig_name, (sig_cls, sig_input, sig_label) in SIGNAL_REGISTRY.items():
                t_run = time.time()
                kalman = KalmanFilter(kalman_cfg)
                signal = sig_cls(window_sec=WINDOW_SEC)
                policy = BucketedHedgeAdversePolicy(
                    pressure_edges  = list(policy_cfg.pressure_edges),
                    secondary_edges = DEFAULT_SECONDARY_EDGES[sig_name],
                    learning_rate   = policy_cfg.learning_rate,
                    initial_weight  = policy_cfg.initial_weight,
                    secondary_label = sig_label,
                    seed            = seed,
                )
                m = replay_adverse(sub, kalman, policy, signal, sig_input,
                                   execution_cfg.lambda_, obs_features)
                results.append(_record(s_idx, start_ts, end_ts, len(sub), dur_h,
                                       seed, sig_name, m,
                                       sec_edges=DEFAULT_SECONDARY_EDGES[sig_name]))
                print(f"    seed={seed} {sig_name:<12}  loss={m['total_loss']:.2f}  "
                      f"slip={m['total_slippage']:.2f}  adv={m['total_adverse']:.2f}  "
                      f"({time.time()-t_run:.1f}s)", flush=True)

    out = {
        "ran_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "lambda":              execution_cfg.lambda_,
            "eta":                 policy_cfg.learning_rate,
            "pressure_edges":      list(policy_cfg.pressure_edges),
            "secondary_edges":     DEFAULT_SECONDARY_EDGES,
            "window_sec":          WINDOW_SEC,
            "seeds":               SEEDS,
            "kalman_obs_features": obs_features,
            "target_sessions":     TARGET_SESSIONS,
            "modes":               ["1D_baseline", *SIGNAL_REGISTRY.keys()],
        },
        "n_results": len(results),
        "results":   results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    tmp.replace(OUTPUT_PATH)
    print(f"Wrote {len(results)} rows → {OUTPUT_PATH}  (elapsed {time.time()-t0:.1f}s)",
          flush=True)


def _record(sidx, start_ts, end_ts, n_rows, dur_h, seed, mode, m, sec_edges=None):
    rec = {
        "session_idx":         sidx,
        "session_start_iso":   _iso(start_ts),
        "session_end_iso":     _iso(end_ts),
        "session_n_rows":      n_rows,
        "session_duration_h":  round(dur_h, 4),
        "seed":           seed,
        "mode":           mode,
        "n_ticks":        m["n"],
        "n_filled":       m["n_filled"],
        "fill_rate":      m["fill_rate"],
        "total_slippage": m["total_slippage"],
        "total_adverse":  m["total_adverse"],
        "total_loss":     m["total_loss"],
        "wait_pct":       m["wait_pct"],
        "passive_pct":    m["passive_pct"],
        "aggressive_pct": m["aggressive_pct"],
    }
    if sec_edges is not None:
        rec["secondary_edges"] = list(sec_edges)
    return rec


if __name__ == "__main__":
    main()
