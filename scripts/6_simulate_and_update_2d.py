"""
Script 6 (2D variant) — full pipeline with 2D state-conditioned Hedge.

Sibling of scripts/6_simulate_and_update.py. Identical structure, identical
session detection, identical execution log layout. The ONLY difference is
the policy: this script forces mode='bucketed_2d' so the policy maintains
one Hedge weight vector per (market_pressure_bucket, regime_bucket) cell of
a 2D grid, instead of one per pressure bucket.

Why a separate script?
  Keeps the validated 1D bucketed run (sanity.txt §15) reproducible from
  the same configs/policy.yaml without changing its `mode` field. Run both
  scripts back to back for an apples-to-apples A/B on the same data.

Run:
  python3 scripts/6_simulate_and_update_2d.py                   # longest session
  python3 scripts/6_simulate_and_update_2d.py --list-sessions   # show and exit
  python3 scripts/6_simulate_and_update_2d.py --session 2       # pick session 2
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from src.execution.updater import ExecutionConfig, run
from src.policy.actions import Action
from src.policy.hedge import PolicyConfig, make_policy
from src.state.kalman_filter import KalmanConfig, KalmanFilter
from src.utils.config import load_config
from src.utils.io import iter_parquet_dir
from src.utils.logging import get_logger

log = get_logger("6_simulate_and_update_2d")

FEATURES_DIR = Path("data/derived/features")
SESSION_GAP_THRESHOLD_MS = 5_000


def detect_sessions(rows: List[dict]) -> List[Tuple[int, int]]:
    if not rows:
        return []
    sessions: List[Tuple[int, int]] = []
    start = 0
    for i in range(1, len(rows)):
        time_gap = rows[i]["ts_ms"] - rows[i - 1]["ts_ms"]
        if rows[i].get("sequence_gap") or time_gap > SESSION_GAP_THRESHOLD_MS:
            sessions.append((start, i))
            start = i
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
        log.info(
            "  %-4d %-10d %-10.2f %-22s %-22s",
            i, n, duration_h,
            _fmt_ts(rows[a]["ts_ms"]),
            _fmt_ts(rows[b - 1]["ts_ms"]),
        )


def pick_session(
    sessions: List[Tuple[int, int]],
    requested: int | None,
) -> Tuple[int, Tuple[int, int]]:
    if requested is None:
        idx = max(range(len(sessions)), key=lambda i: sessions[i][1] - sessions[i][0])
        return idx + 1, sessions[idx]
    if requested < 1 or requested > len(sessions):
        raise SystemExit(
            f"--session {requested} out of range; detected {len(sessions)} session(s). "
            f"Run with --list-sessions to see them."
        )
    return requested, sessions[requested - 1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--session", type=int, default=None,
                   help="1-indexed session to run (default: longest detected session)")
    p.add_argument("--list-sessions", action="store_true",
                   help="Print detected sessions and exit without running the pipeline.")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for the policy (default: nondeterministic)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    kalman_cfg    = load_config("configs/kalman.yaml",    KalmanConfig)
    policy_cfg    = load_config("configs/policy.yaml",    PolicyConfig)
    execution_cfg = load_config("configs/execution.yaml", ExecutionConfig)

    # Force 2D bucketing for this script regardless of the YAML's mode.
    policy_cfg.mode = "bucketed_2d"
    if not policy_cfg.regime_edges:
        raise SystemExit(
            "configs/policy.yaml is missing regime_edges. "
            "Add e.g. `regime_edges: [-0.3, 0.3]` to enable 2D bucketing."
        )

    rows = list(iter_parquet_dir(FEATURES_DIR))
    if not rows:
        raise SystemExit(f"No feature rows found under {FEATURES_DIR}.")

    sessions = detect_sessions(rows)
    log_session_table(sessions, rows)

    if args.list_sessions:
        return

    session_num, (start, end) = pick_session(sessions, args.session)
    selected = rows[start:end]
    log.info(
        "Running on session %d — %d rows (%s → %s)",
        session_num, len(selected),
        _fmt_ts(selected[0]["ts_ms"]), _fmt_ts(selected[-1]["ts_ms"]),
    )

    kalman = KalmanFilter(kalman_cfg)
    policy = make_policy(policy_cfg, seed=args.seed)

    log.info(
        "Starting run — mode=%s λ=%.2f η=%.3f seed=%s",
        policy_cfg.mode, execution_cfg.lambda_, policy_cfg.learning_rate, args.seed,
    )
    log.info("Pressure edges: %s  (%d buckets)",
             policy_cfg.pressure_edges, len(policy_cfg.pressure_edges) + 1)
    log.info("Regime   edges: %s  (%d buckets)",
             policy_cfg.regime_edges, len(policy_cfg.regime_edges) + 1)
    log.info("2D grid: %d × %d = %d cells",
             len(policy_cfg.pressure_edges) + 1,
             len(policy_cfg.regime_edges) + 1,
             (len(policy_cfg.pressure_edges) + 1) * (len(policy_cfg.regime_edges) + 1))

    total = run(
        feature_rows=iter(selected),
        kalman=kalman,
        policy=policy,
        config=execution_cfg,
        obs_features=kalman_cfg.obs_features,
    )

    log.info("Run complete — %d ticks processed.", total)
    final_weights = policy.weights()
    log.info(
        "Final weights (last-bucket view) — WAIT: %.4f  PASSIVE: %.4f  AGGRESSIVE: %.4f",
        final_weights[Action.WAIT],
        final_weights[Action.PASSIVE],
        final_weights[Action.AGGRESSIVE],
    )

    log.info("Per-(pressure, regime)-bucket final weights:")
    log.info("  %-6s %-22s %-22s %-8s %-10s %-10s %-10s",
             "bucket", "pressure range", "regime range",
             "visits", "WAIT", "PASSIVE", "AGGR")
    for b in policy.bucket_summary():
        w = b["weights"]
        log.info("  %-6d %-22s %-22s %-8d %-10.4f %-10.4f %-10.4f",
                 b["bucket"], b["p_range"], b["r_range"], b["visits"],
                 w[Action.WAIT], w[Action.PASSIVE], w[Action.AGGRESSIVE])

    log.info("Execution log → %s", execution_cfg.output_dir)


if __name__ == "__main__":
    main()
