"""
Script 6 (adverse variant) — full pipeline with adverse-targeted 2D Hedge.

Sibling of scripts/6_simulate_and_update.py and
scripts/6_simulate_and_update_2d.py. Identical session detection.
Identical Kalman filter. The DIFFERENCE is the policy:

  - 1D bucketed   (script 6):    2nd axis = none (1D pressure-only)
  - 2D bucketed   (script 6_2d): 2nd axis = Kalman regime field
  - adverse 2D    (THIS SCRIPT): 2nd axis = a configurable rolling-window
                                  signal computed from raw features
                                  (vol_delta / ofi_window / spread_delta)

The motivation is in sanity.txt §24: §21 + §23 showed the Kalman regime
predicts FILL probability, not ADVERSE risk. We hypothesize that a
second axis built specifically to predict adverse risk (rising vol,
sustained lopsided OFI, widening spread) might let 2D bucketing actually
beat 1D on combined cost.

Outputs go to a SEPARATE execution log directory so they don't collide
with the canonical 1D / 2D logs:
  data/derived/execution_log_adverse/run=<UTC-stamp>/

Run examples:
  python3 scripts/6_simulate_and_update_adverse.py --signal vol_delta
  python3 scripts/6_simulate_and_update_adverse.py --signal ofi_window --session 3 --seed 0
  python3 scripts/6_simulate_and_update_adverse.py --list-sessions
  python3 scripts/6_simulate_and_update_adverse.py --list-signals
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.execution.loss import compute_loss
from src.execution.simulator import simulate_fill
from src.execution.updater import ExecutionConfig
from src.features.adverse_signals import SIGNAL_REGISTRY
from src.policy.actions import Action
from src.policy.hedge import PolicyConfig
from src.policy.hedge_adverse import AdverseContext, BucketedHedgeAdversePolicy
from src.state.kalman_filter import KalmanConfig, KalmanFilter
from src.utils.config import load_config
from src.utils.io import append_jsonl, iter_parquet_dir
from src.utils.logging import get_logger
from src.utils.time_utils import date_partition

log = get_logger("6_simulate_and_update_adverse")

FEATURES_DIR = Path("data/derived/features")
SESSION_GAP_THRESHOLD_MS = 5_000

# Default secondary-axis bucket edges per signal — calibrated from pooled
# p33/p66 across the 3 real sessions on disk (Apr-21, Apr-28, Apr-30).
# spread_delta uses a single edge at 0 because the signal is concentrated
# at exactly 0 (most 60s windows have no spread change in this market) so
# quantile-based bucketing degenerates; instead we test the binary
# hypothesis "does any spread widening predict adverse moves?"
DEFAULT_SECONDARY_EDGES = {
    "vol_delta":    [-1.1e-5, +9.0e-6],
    "ofi_window":   [-2.0,    +1.9],
    "spread_delta": [0.0],
}
DEFAULT_WINDOW_SEC = 60.0


# ---------------------------------------------------------------------------
# Output schema (flat dataclass — written as JSONL)
# ---------------------------------------------------------------------------

@dataclass
class AdverseExecutionRecord:
    """Like execution_log/ExecutionRecord but with the secondary axis logged."""
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
    regime:            float       # Kalman regime — kept for cross-reference
    secondary:         float       # the actual axis being bucketed on
    signal_name:       str
    weight_wait:       float
    weight_passive:    float
    weight_aggressive: float


# ---------------------------------------------------------------------------
# Session detection (copy of the canonical logic in script 6)
# ---------------------------------------------------------------------------

def detect_sessions(rows: List[dict]) -> List[Tuple[int, int]]:
    if not rows:
        return []
    sessions: List[Tuple[int, int]] = []
    start = 0
    for i in range(1, len(rows)):
        time_gap = rows[i]["ts_ms"] - rows[i - 1]["ts_ms"]
        if rows[i].get("sequence_gap") or time_gap > SESSION_GAP_THRESHOLD_MS:
            sessions.append((start, i)); start = i
    sessions.append((start, len(rows)))
    return sessions


def _fmt_ts(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def log_session_table(sessions: List[Tuple[int, int]], rows: List[dict]) -> None:
    log.info("Detected %d session(s) in %s:", len(sessions), FEATURES_DIR)
    log.info("  %-4s %-10s %-10s %-22s %-22s", "idx", "rows", "hours", "start", "end")
    for i, (a, b) in enumerate(sessions, start=1):
        n = b - a
        duration_h = (rows[b - 1]["ts_ms"] - rows[a]["ts_ms"]) / 3_600_000
        log.info("  %-4d %-10d %-10.2f %-22s %-22s",
                 i, n, duration_h, _fmt_ts(rows[a]["ts_ms"]), _fmt_ts(rows[b - 1]["ts_ms"]))


def pick_session(sessions, requested):
    if requested is None:
        idx = max(range(len(sessions)), key=lambda i: sessions[i][1] - sessions[i][0])
        return idx + 1, sessions[idx]
    if requested < 1 or requested > len(sessions):
        raise SystemExit(f"--session {requested} out of range; detected {len(sessions)} session(s).")
    return requested, sessions[requested - 1]


# ---------------------------------------------------------------------------
# Replay loop (inline — does not reuse src/execution/updater.run because
# the policy needs a different ctx type)
# ---------------------------------------------------------------------------

def replay_adverse(
    rows:            List[dict],
    kalman:          KalmanFilter,
    policy:          BucketedHedgeAdversePolicy,
    signal,                                        # stateful signal computer
    signal_name:     str,
    signal_input_key: str,
    obs_features:    List[str],
    lambda_:         float,
    log_path:        Path,
) -> int:
    """Same five steps as src/execution/updater.run, but:
      - computes the configurable secondary signal each tick
      - builds an AdverseContext (not PolicyContext)
      - logs an AdverseExecutionRecord (with the secondary value)
    """
    total = 0
    prev: dict | None = None
    for row in rows:
        if prev is None:
            prev = row
            continue
        curr, nxt = prev, row

        # --- Kalman ---
        z = np.array([curr[f] for f in obs_features], dtype=float)
        ks = kalman.step(curr["ts_ms"], z)
        if ks is None:
            prev = row
            continue

        # --- Secondary signal (stateful, fed each tick) ---
        sec = signal.update(curr["ts_ms"], curr[signal_input_key])

        # --- Policy ---
        ctx = AdverseContext(
            ts_ms=curr["ts_ms"],
            market_pressure=ks.market_pressure,
            secondary=sec,
        )
        decision = policy.select(ctx)
        action = Action(decision.action)
        weights = policy.weights()

        # --- Simulator + loss + update ---
        fill = simulate_fill(
            action=action,
            ts_ms=curr["ts_ms"],
            curr_mid=curr["mid_price"],
            curr_best_bid=curr["best_bid_px"],
            curr_best_ask=curr["best_ask_px"],
            next_mid=nxt["mid_price"],
        )
        loss = compute_loss(fill, lambda_)
        policy.update(action, loss)

        slippage = (fill.fill_price - fill.mid_price) if fill.filled else 0.0
        adverse  = (max(0.0, fill.fill_price - fill.next_mid_price) if fill.filled
                    else max(0.0, fill.next_mid_price - fill.mid_price))

        rec = AdverseExecutionRecord(
            ts_ms=curr["ts_ms"],
            action=decision.action,
            filled=fill.filled,
            fill_price=fill.fill_price,
            mid_price=fill.mid_price,
            next_mid_price=fill.next_mid_price,
            slippage=slippage,
            adverse_move=adverse,
            loss=loss,
            market_pressure=ks.market_pressure,
            regime=ks.regime,
            secondary=sec,
            signal_name=signal_name,
            weight_wait=weights[Action.WAIT],
            weight_passive=weights[Action.PASSIVE],
            weight_aggressive=weights[Action.AGGRESSIVE],
        )
        date_str = date_partition(curr["ts_ms"])
        full_path = log_path / f"date={date_str}" / "log.jsonl"
        append_jsonl(full_path, asdict(rec))
        total += 1
        prev = row
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--signal", default=None,
                   help=f"Adverse-risk signal to use as second axis. Choices: "
                        f"{sorted(SIGNAL_REGISTRY.keys())}")
    p.add_argument("--list-signals", action="store_true",
                   help="Print available signals and their input feature, then exit.")
    p.add_argument("--session", type=int, default=None,
                   help="1-indexed session to run (default: longest detected).")
    p.add_argument("--list-sessions", action="store_true",
                   help="Print detected sessions and exit.")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for the policy (default: nondeterministic).")
    p.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC,
                   help=f"Rolling-window length in seconds (default {DEFAULT_WINDOW_SEC}).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_signals:
        log.info("Available signals (--signal):")
        for name, (cls, feat, label) in SIGNAL_REGISTRY.items():
            edges = DEFAULT_SECONDARY_EDGES.get(name, "?")
            log.info("  %-14s  reads %-12s  → %s   default edges %s",
                     name, feat, label, edges)
        return

    kalman_cfg    = load_config("configs/kalman.yaml",    KalmanConfig)
    policy_cfg    = load_config("configs/policy.yaml",    PolicyConfig)
    execution_cfg = load_config("configs/execution.yaml", ExecutionConfig)

    rows = list(iter_parquet_dir(FEATURES_DIR))
    if not rows:
        raise SystemExit(f"No feature rows found under {FEATURES_DIR}.")

    sessions = detect_sessions(rows)
    log_session_table(sessions, rows)
    if args.list_sessions:
        return

    if args.signal is None:
        raise SystemExit("--signal required (or pass --list-signals).")
    if args.signal not in SIGNAL_REGISTRY:
        raise SystemExit(
            f"Unknown signal '{args.signal}'. Available: {sorted(SIGNAL_REGISTRY.keys())}")

    sig_cls, sig_input, sig_label = SIGNAL_REGISTRY[args.signal]
    sec_edges = DEFAULT_SECONDARY_EDGES[args.signal]

    session_num, (start, end) = pick_session(sessions, args.session)
    selected = rows[start:end]
    log.info("Running on session %d — %d rows (%s → %s)",
             session_num, len(selected),
             _fmt_ts(selected[0]["ts_ms"]), _fmt_ts(selected[-1]["ts_ms"]))

    kalman = KalmanFilter(kalman_cfg)
    signal = sig_cls(window_sec=args.window_sec)
    policy = BucketedHedgeAdversePolicy(
        pressure_edges  = list(policy_cfg.pressure_edges),
        secondary_edges = sec_edges,
        learning_rate   = policy_cfg.learning_rate,
        initial_weight  = policy_cfg.initial_weight,
        secondary_label = sig_label,
        seed            = args.seed,
    )

    # Dedicated output dir so we don't pollute the canonical execution_log/.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base_dir = Path("data/derived/execution_log_adverse")
    out_dir  = base_dir / f"run={run_id}"
    log.info("Adverse execution log run_id=%s  output=%s", run_id, out_dir)

    log.info("Starting run — mode=adverse_2d signal=%s window=%.0fs λ=%.2f η=%.3f seed=%s",
             args.signal, args.window_sec, execution_cfg.lambda_,
             policy_cfg.learning_rate, args.seed)
    log.info("Pressure  edges: %s  (%d buckets)",
             policy_cfg.pressure_edges, len(policy_cfg.pressure_edges) + 1)
    log.info("Secondary edges: %s  (%d buckets)  [signal: %s = %s]",
             sec_edges, len(sec_edges) + 1, args.signal, sig_label)
    log.info("Adverse 2D grid: %d × %d = %d cells",
             len(policy_cfg.pressure_edges) + 1, len(sec_edges) + 1,
             (len(policy_cfg.pressure_edges) + 1) * (len(sec_edges) + 1))

    total = replay_adverse(
        rows=selected,
        kalman=kalman,
        policy=policy,
        signal=signal,
        signal_name=args.signal,
        signal_input_key=sig_input,
        obs_features=list(kalman_cfg.obs_features),
        lambda_=execution_cfg.lambda_,
        log_path=out_dir,
    )

    if total > 0:
        latest = base_dir / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        os.symlink(f"run={run_id}", latest)
        log.info("Updated 'latest' symlink → run=%s", run_id)

    log.info("Run complete — %d ticks processed.", total)
    final = policy.weights()
    log.info("Final weights (last-bucket view) — WAIT: %.4f  PASSIVE: %.4f  AGGRESSIVE: %.4f",
             final[Action.WAIT], final[Action.PASSIVE], final[Action.AGGRESSIVE])

    log.info("Per-(pressure, %s)-bucket final weights:", args.signal)
    log.info("  %-6s %-22s %-26s %-8s %-10s %-10s %-10s",
             "bucket", "pressure range", f"{args.signal} range",
             "visits", "WAIT", "PASSIVE", "AGGR")
    for b in policy.bucket_summary():
        w = b["weights"]
        log.info("  %-6d %-22s %-26s %-8d %-10.4f %-10.4f %-10.4f",
                 b["bucket"], b["p_range"], b["s_range"], b["visits"],
                 w[Action.WAIT], w[Action.PASSIVE], w[Action.AGGRESSIVE])


if __name__ == "__main__":
    main()
