"""
Script 6 — Full pipeline: Kalman → policy → simulate → loss → weight update.

This is the main training run. It reads all feature Parquet files, runs the
complete tick-by-tick loop, and writes an execution log to
data/derived/execution_log/date=YYYY-MM-DD/log.jsonl.

After this script completes, the execution log contains everything needed
for validation and charting:
  - What the Kalman filter estimated at every tick (market_pressure, regime)
  - What action the policy chose and with what weights
  - Whether the order filled and at what price
  - The slippage, adverse move, and total loss
  - How the weights evolved over time

Run: python3 scripts/6_simulate_and_update.py

After running, inspect results with:
  python3 scripts/explore_features.py   (adapt to read execution log)
"""

from __future__ import annotations

from pathlib import Path

from src.execution.updater import ExecutionConfig, run
from src.policy.actions import Action
from src.policy.hedge import PolicyConfig, make_policy
from src.state.kalman_filter import KalmanConfig, KalmanFilter
from src.utils.config import load_config
from src.utils.io import iter_parquet_dir
from src.utils.logging import get_logger

log = get_logger("6_simulate_and_update")

FEATURES_DIR = Path("data/derived/features")


def main() -> None:
    kalman_cfg    = load_config("configs/kalman.yaml",    KalmanConfig)
    policy_cfg    = load_config("configs/policy.yaml",    PolicyConfig)
    execution_cfg = load_config("configs/execution.yaml", ExecutionConfig)

    kalman = KalmanFilter(kalman_cfg)
    policy = make_policy(policy_cfg, seed=None)

    log.info(
        "Starting run — mode=%s λ=%.2f η=%.3f features_dir=%s",
        policy_cfg.mode,
        execution_cfg.lambda_,
        policy_cfg.learning_rate,
        FEATURES_DIR,
    )
    if policy_cfg.mode == "bucketed":
        log.info("Pressure edges: %s  (%d buckets)",
                 policy_cfg.pressure_edges, len(policy_cfg.pressure_edges) + 1)

    feature_rows = iter_parquet_dir(FEATURES_DIR)
    total = run(
        feature_rows=feature_rows,
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

    if hasattr(policy, "bucket_summary"):
        log.info("Per-bucket final weights:")
        log.info("  %-6s %-22s %-8s %-10s %-10s %-10s",
                 "bucket", "pressure range", "visits", "WAIT", "PASSIVE", "AGGR")
        for b in policy.bucket_summary():
            w = b["weights"]
            log.info("  %-6d %-22s %-8d %-10.4f %-10.4f %-10.4f",
                     b["bucket"], b["range"], b["visits"],
                     w[Action.WAIT], w[Action.PASSIVE], w[Action.AGGRESSIVE])

    log.info("Execution log → %s", execution_cfg.output_dir)


if __name__ == "__main__":
    main()
