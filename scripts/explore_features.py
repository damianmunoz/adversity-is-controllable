"""
Quick demo: load all feature parquet files for April 7th and explore them.

The data lives in a Hive-partitioned directory tree:
  data/derived/features/date=YYYY-MM-DD/hour=HH/features.parquet

pandas.read_parquet on a directory path reads all matching files at once
and adds the partition columns (date, hour) as regular columns automatically.
"""

import pandas as pd
from pathlib import Path

# --- Load ---
# Point at the date partition folder; pandas/pyarrow reads all hour subfolders.
# Anchor to project root so this works regardless of CWD.
project_root = Path(__file__).resolve().parents[1]
date_dir = project_root / "data/derived/features/date=2026-04-07"
df = pd.read_parquet(date_dir)

# --- Schema ---
print("=== Shape ===")
print(f"{df.shape[0]:,} rows  x  {df.shape[1]} columns\n")

print("=== Columns & dtypes ===")
print(df.dtypes.to_string())
print()

# --- Sample rows ---
print("=== First 5 rows ===")
print(df.head().to_string())
print()

# --- Quick stats on numeric features ---
print("=== Numeric summary ===")
print(df.describe().to_string())

# ---------------------------------------------------------------------------
# Night session: April 7th, 18:00 UTC onward
# ---------------------------------------------------------------------------
# The Hive partition reader adds a string "hour" column ("00", "01", ...).
# We cast to int for comparison and sort.
df["hour_int"] = df["hour"].astype(int)
night = df[df["hour_int"] >= 18].sort_values("ts_ms").copy()

print("\n\n" + "=" * 60)
print("NIGHT SESSION — April 7th, 18:00 UTC onward")
print("=" * 60)
print(f"{len(night):,} rows  |  hours: {sorted(night['hour_int'].unique())}\n")

# --- BTC Price ---
print("=== BTC Price (mid_price) per hour ===")
price = night.groupby("hour_int")["mid_price"].agg(
    open="first", close="last", low="min", high="max"
)
price["change_$"] = (price["close"] - price["open"]).round(2)
price["change_%"] = ((price["close"] - price["open"]) / price["open"] * 100).round(4)
print(price.to_string())
print()

# --- BTC Slippage ---
# spread_bps = (ask - bid) / mid * 10000
# This is the round-trip crossing cost in basis points — the tightest
# slippage estimate for a market order at each tick.
print("=== BTC Slippage (spread_bps) per hour ===")
slippage = night.groupby("hour_int")["spread_bps"].agg(
    mean="mean", median="median", p95=lambda x: x.quantile(0.95), max="max"
).round(4)
print(slippage.to_string())
print()

# --- Price Change per tick ---
print("=== Price Change per tick (mid_price.diff) per hour ===")
night["price_chg_$"] = night["mid_price"].diff()
night["price_chg_%"] = night["mid_price"].pct_change() * 100
chg = night.groupby("hour_int")[["price_chg_$", "price_chg_%"]].agg(
    ["mean", "std", "min", "max"]
).round(6)
print(chg.to_string())
print()

# --- Predicted Pressure (pre-Kalman proxy) ---
# The Kalman filter will estimate hidden state = [market_pressure, regime].
# Its observations are [depth_imbalance, ofi_l1, vol_5s] (from kalman.yaml).
# Before that model is built, this proxy captures the same directional signal:
#
#   pressure_proxy = 0.6 * depth_imbalance + 0.4 * ofi_l1_norm
#
# depth_imbalance is already in (-1, +1).
# ofi_l1 is z-score normalized here so both terms have equal scale.
# Weights match the rough R diagonal in kalman.yaml (imbalance trusted more).
# Clipped to (-1, +1) so it reads like the Kalman's market_pressure output will.
#
# +1 = strong buy pressure   |   -1 = strong sell pressure
print("=== Predicted Pressure — pre-Kalman proxy per hour ===")
ofi_std = night["ofi_l1"].std()
ofi_norm = night["ofi_l1"] / ofi_std if ofi_std > 0 else 0.0
night["pressure_proxy"] = (
    0.6 * night["depth_imbalance"] + 0.4 * ofi_norm
).clip(-1, 1)

pressure = night.groupby("hour_int")["pressure_proxy"].agg(
    mean="mean", std="std", min="min", max="max"
).round(4)
print(pressure.to_string())
print()
print("Interpretation: +1 = strong buy pressure  |  -1 = strong sell pressure")
print("(Pre-model proxy — Kalman filter will replace this with state estimation)")
