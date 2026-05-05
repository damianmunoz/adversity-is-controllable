"""
TickRecord — the data unit that flows from the engine to the GUI.

One TickRecord per processed tick. Carries everything needed to:
  - render a row in the packet table
  - populate the detail pane on click
  - draw a marker on the chart

All fields are flat scalars so the record is trivially serializable.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Dict


@dataclass
class TickRecord:
    """One row of live/replay output."""

    # ---- identity / time ---------------------------------------------------
    tick_idx: int
    ts_ms:    int                # decision-tick timestamp
    next_ts_ms: int              # outcome-tick timestamp (used for spread)

    # ---- order book at decision time --------------------------------------
    best_bid:   float
    best_ask:   float
    mid:        float
    next_mid:   float
    spread_abs: float
    microprice: float

    # ---- raw features used by Kalman --------------------------------------
    depth_imbalance: float
    ofi_l1:          float
    vol_30s:         float

    # ---- Kalman posterior --------------------------------------------------
    pressure: float
    regime:   float
    p00:      float
    p11:      float
    p01:      float

    # ---- policy decision ---------------------------------------------------
    bucket:     int
    bucket_lo:  float            # -inf encoded as -1e18
    bucket_hi:  float            # +inf encoded as +1e18
    weight_wait:       float
    weight_passive:    float
    weight_aggressive: float
    action:     str              # "WAIT" | "PASSIVE" | "AGGRESSIVE"

    # ---- execution outcome -------------------------------------------------
    filled:        bool
    fill_price:    float
    slippage:      float
    adverse_move:  float
    loss:          float

    # ---- running totals ----------------------------------------------------
    cum_slippage: float
    cum_adverse:  float
    cum_loss:     float
    cum_filled:   int
    cum_ticks:    int

    # ---- counterfactual baseline (always-AGGR) ----------------------------
    # We compute the slippage and adverse the AGGR baseline would have paid
    # AT THIS TICK, regardless of what the policy chose. Useful for
    # "how much did the system save vs always-aggressive" charts.
    aggr_slippage: float
    aggr_adverse:  float
    cum_aggr_loss: float

    # ---------------------------------------------------------------------------
    # Convenience
    # ---------------------------------------------------------------------------

    def utc_str(self) -> str:
        return datetime.fromtimestamp(self.ts_ms / 1000.0, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    def as_dict(self) -> Dict:
        return asdict(self)
