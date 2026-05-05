"""
LiveEngine — orchestrates one tick of the pipeline and emits a TickRecord.

Pulls (curr, next) feature pairs from a source iterator, runs:

   Kalman → policy.select → simulator → loss

and accumulates running totals. Returns one TickRecord per processed tick
or None when:
  - the Kalman filter rejected the observation (NaN feature)
  - we have no `next` row yet (first tick — buffered)

Designed to be GUI-agnostic. The GUI wraps this in a Qt controller.
"""

from __future__ import annotations

import math
from typing import Dict, Iterator, Optional

import numpy as np

from src.execution.loss import compute_loss
from src.execution.simulator import simulate_fill
from src.live.frozen_policy import FrozenBucketedPolicy
from src.live.records import TickRecord
from src.policy.actions import Action
from src.policy.hedge import PolicyContext
from src.state.kalman_filter import KalmanFilter, KalmanConfig


_INF = 1e18


def _bucket_range(edges, b: int):
    """Return (lo, hi) for bucket b in a list with len(edges)+1 buckets.

    Uses ±_INF for unbounded sides so the GUI can render them numerically.
    """
    if b == 0:
        lo = -_INF
    else:
        lo = float(edges[b - 1])
    if b == len(edges):
        hi = +_INF
    else:
        hi = float(edges[b])
    return lo, hi


class LiveEngine:
    """Stateful single-tick driver over a frozen 1D bucketed Hedge.

    Args:
        kalman_cfg: KalmanConfig — fresh filter is constructed internally
        policy:     FrozenBucketedPolicy with weights already loaded
        lambda_:    loss-function adverse weight (default 0.10, matches §9)
        obs_features: ordered names of fields read from each feature row to
                      build the Kalman observation vector
    """

    def __init__(
        self,
        kalman_cfg:   KalmanConfig,
        policy:       FrozenBucketedPolicy,
        lambda_:      float = 0.10,
        obs_features = None,
    ) -> None:
        self.kalman   = KalmanFilter(kalman_cfg)
        self.policy   = policy
        self.lambda_  = float(lambda_)
        self.obs_features = list(obs_features or kalman_cfg.obs_features)

        self._tick_idx     = 0
        self._cum_slip     = 0.0
        self._cum_adv      = 0.0
        self._cum_loss     = 0.0
        self._cum_filled   = 0
        self._cum_aggr_loss = 0.0
        self._prev_row: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, row: dict) -> Optional[TickRecord]:
        """Push one feature row.

        Returns a TickRecord built from the PREVIOUS row (whose outcome is now
        observable through this row's mid_price), or None on the first call
        and on Kalman rejections.
        """
        if self._prev_row is None:
            self._prev_row = row
            return None

        curr = self._prev_row
        nxt  = row
        self._prev_row = row

        z = np.array([curr[f] for f in self.obs_features], dtype=float)
        kalman_state = self.kalman.step(curr["ts_ms"], z)
        if kalman_state is None:
            return None

        ctx = PolicyContext(
            ts_ms=curr["ts_ms"],
            market_pressure=kalman_state.market_pressure,
            regime=kalman_state.regime,
        )
        bucket = self.policy.which_bucket(kalman_state.market_pressure)

        decision = self.policy.select(ctx)
        action   = Action(decision.action)
        weights  = self.policy.weights()

        fill = simulate_fill(
            action=action,
            ts_ms=curr["ts_ms"],
            curr_mid=curr["mid_price"],
            curr_best_bid=curr["best_bid_px"],
            curr_best_ask=curr["best_ask_px"],
            next_mid=nxt["mid_price"],
        )
        loss = compute_loss(fill, self.lambda_)
        # Frozen → no policy update.

        slippage = (fill.fill_price - fill.mid_price) if fill.filled else 0.0
        adverse  = (
            max(0.0, fill.fill_price - fill.next_mid_price) if fill.filled
            else max(0.0, fill.next_mid_price - fill.mid_price)
        )

        # Counterfactual: what if we had gone AGGRESSIVE every tick?
        aggr_slip = curr["best_ask_px"] - curr["mid_price"]
        aggr_adv  = max(0.0, curr["best_ask_px"] - nxt["mid_price"])
        aggr_loss = aggr_slip + self.lambda_ * aggr_adv

        # Accumulate
        self._tick_idx     += 1
        self._cum_slip     += slippage
        self._cum_adv      += adverse
        self._cum_loss     += loss
        self._cum_aggr_loss += aggr_loss
        if fill.filled:
            self._cum_filled += 1

        edges = self.policy.pressure_edges
        bucket_lo, bucket_hi = _bucket_range(edges, bucket)

        rec = TickRecord(
            tick_idx=self._tick_idx,
            ts_ms=curr["ts_ms"],
            next_ts_ms=nxt["ts_ms"],

            best_bid=curr["best_bid_px"],
            best_ask=curr["best_ask_px"],
            mid=curr["mid_price"],
            next_mid=nxt["mid_price"],
            spread_abs=curr["spread_abs"],
            microprice=curr["microprice"],

            depth_imbalance=curr["depth_imbalance"],
            ofi_l1=curr["ofi_l1"],
            vol_30s=curr["vol_30s"],

            pressure=kalman_state.market_pressure,
            regime=kalman_state.regime,
            p00=kalman_state.p00,
            p11=kalman_state.p11,
            p01=kalman_state.p01,

            bucket=bucket,
            bucket_lo=bucket_lo,
            bucket_hi=bucket_hi,
            weight_wait=weights[Action.WAIT],
            weight_passive=weights[Action.PASSIVE],
            weight_aggressive=weights[Action.AGGRESSIVE],
            action=decision.action,

            filled=fill.filled,
            fill_price=fill.fill_price,
            slippage=slippage,
            adverse_move=adverse,
            loss=loss,

            cum_slippage=self._cum_slip,
            cum_adverse=self._cum_adv,
            cum_loss=self._cum_loss,
            cum_filled=self._cum_filled,
            cum_ticks=self._tick_idx,

            aggr_slippage=aggr_slip,
            aggr_adverse=aggr_adv,
            cum_aggr_loss=self._cum_aggr_loss,
        )
        return rec

    def run(self, rows: Iterator[dict]) -> Iterator[TickRecord]:
        """Generator wrapper for offline use (no GUI). Yields TickRecord."""
        for row in rows:
            rec = self.step(row)
            if rec is not None:
                yield rec
