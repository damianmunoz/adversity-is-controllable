"""
Script 4 — Run Kalman filter over feature Parquet files.

Reads all hourly feature Parquet files from data/derived/features/,
feeds them through the Kalman filter in chronological order, and writes
the resulting KalmanState records to data/derived/kalman/ using the same
Hive partition layout (date=YYYY-MM-DD/hour=HH/kalman.parquet).

Run: python scripts/4_run_kalman.py
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
import pyarrow as pa

from src.state.kalman_filter import KalmanConfig, KalmanFilter, KalmanState
from src.utils.config import load_config
from src.utils.io import iter_parquet_dir, write_parquet
from src.utils.logging import get_logger
from src.utils.time_utils import hour_partition

log = get_logger("4_run_kalman")

FEATURES_DIR = Path("data/derived/features")
OUTPUT_DIR   = Path("data/derived/kalman")

KALMAN_SCHEMA = pa.schema([
    ("ts_ms",            pa.int64()),
    ("market_pressure",  pa.float64()),
    ("regime",           pa.float64()),
    ("p00",              pa.float64()),
    ("p11",              pa.float64()),
    ("p01",              pa.float64()),
])


def main() -> None:
    cfg = load_config("configs/kalman.yaml", KalmanConfig)
    kf  = KalmanFilter(cfg)

    log.info("Kalman config: state_dim=%d, obs_dim=%d, obs_features=%s",
             cfg.state_dim, cfg.obs_dim, cfg.obs_features)

    batches: dict[tuple[str, str], List[dict]] = {}
    total_in  = 0
    total_out = 0

    for row in iter_parquet_dir(FEATURES_DIR):
        total_in += 1

        # Build observation vector in the exact order defined in kalman.yaml
        z = np.array([row[f] for f in cfg.obs_features], dtype=float)

        state = kf.step(row["ts_ms"], z)
        if state is None:
            continue

        total_out += 1
        date_str, hour_str = hour_partition(state.ts_ms)
        key = (date_str, hour_str)
        batches.setdefault(key, []).append(asdict(state))

    log.info("Processed %d feature rows → %d Kalman states (skipped %d NaN ticks).",
             total_in, total_out, total_in - total_out)

    for (date_str, hour_str), records in batches.items():
        path = OUTPUT_DIR / f"date={date_str}" / f"hour={hour_str}" / "kalman.parquet"
        write_parquet(records, KALMAN_SCHEMA, path)
        log.info("Wrote %d states → %s", len(records), path)

    log.info("Kalman run complete.")


if __name__ == "__main__":
    main()
