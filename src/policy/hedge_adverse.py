"""
BucketedHedgeAdversePolicy — sibling of BucketedHedge2DPolicy.

Same shape (one Hedge weight vector per cell of a 2D grid), but the second
axis is a *configurable adverse-risk signal* (vol_delta / ofi_window /
spread_delta — see src/features/adverse_signals.py) rather than the
Kalman filter's `regime` field.

WHY a separate class — the existing BucketedHedge2DPolicy is wired to read
`ctx.regime` from PolicyContext. That field semantically means "the
Kalman filter's regime estimate." Re-using it for an unrelated rolling-
window signal would be misleading and would couple two unrelated
experiments through the same field. So:
  - PolicyContext / PolicyConfig are unchanged
  - BucketedHedge2DPolicy is unchanged
  - This module introduces a new context type (AdverseContext) and a
    parallel policy class with explicit constructor arguments

The policy is otherwise mathematically identical to BucketedHedge2DPolicy
(EXP3 / Hedge per cell, multiplicative weights update on observed loss).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.policy.actions import Action
from src.policy.hedge import PolicyDecision  # reuse output schema verbatim
from src.utils.logging import get_logger

log = get_logger(__name__)

_ACTIONS: List[Action] = [Action.WAIT, Action.PASSIVE, Action.AGGRESSIVE]


@dataclass
class AdverseContext:
    """What the adverse-targeted policy sees per tick.

    `secondary` is whatever rolling-window signal the script chose to
    compute — explicitly NOT the Kalman regime. The policy itself is
    agnostic to which signal is being fed in.
    """
    ts_ms: int
    market_pressure: float
    secondary: float


class BucketedHedgeAdversePolicy:
    """One Hedge weight vector per (pressure_bucket, secondary_bucket) cell.

    Storage layout: bucket_index = p_idx * n_secondary + s_idx.

    Interface mirrors BucketedHedge2DPolicy: select(ctx) / update(action, loss)
    / weights() / bucket_summary(). The only schema difference is the
    summary uses 's_range' (secondary range) instead of 'r_range' (regime).
    """

    def __init__(
        self,
        pressure_edges:  List[float],
        secondary_edges: List[float],
        learning_rate:   float,
        initial_weight:  float          = 1.0,
        secondary_label: str            = "secondary",
        seed:            Optional[int]  = None,
    ) -> None:
        if not pressure_edges:
            raise ValueError("BucketedHedgeAdversePolicy requires non-empty pressure_edges")
        if not secondary_edges:
            raise ValueError("BucketedHedgeAdversePolicy requires non-empty secondary_edges")
        if list(pressure_edges)  != sorted(pressure_edges):
            raise ValueError("pressure_edges must be in ascending order")
        if list(secondary_edges) != sorted(secondary_edges):
            raise ValueError("secondary_edges must be in ascending order")

        self._p_edges  = list(pressure_edges)
        self._s_edges  = list(secondary_edges)
        self._n_p      = len(self._p_edges) + 1
        self._n_s      = len(self._s_edges) + 1
        self._eta      = float(learning_rate)
        self._sec_lbl  = secondary_label

        n_total = self._n_p * self._n_s
        uniform = initial_weight / (initial_weight * len(_ACTIONS))
        self._buckets: List[Dict[Action, float]] = [
            {a: uniform for a in _ACTIONS} for _ in range(n_total)
        ]
        self._visits:      List[int] = [0] * n_total
        self._last_bucket: int       = 0
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Bucket lookup
    # ------------------------------------------------------------------

    def _which_bucket(self, market_pressure: float, secondary: float) -> int:
        pi = bisect.bisect_right(self._p_edges, market_pressure)
        si = bisect.bisect_right(self._s_edges, secondary)
        return pi * self._n_s + si

    # ------------------------------------------------------------------
    # Selection / update — same math as BucketedHedge2DPolicy
    # ------------------------------------------------------------------

    def select(self, ctx: AdverseContext) -> PolicyDecision:
        bucket = self._which_bucket(ctx.market_pressure, ctx.secondary)
        self._last_bucket = bucket
        self._visits[bucket] += 1

        w = self._buckets[bucket]
        probs = [w[a] for a in _ACTIONS]
        idx = int(self._rng.choice(len(_ACTIONS), p=probs))
        chosen = _ACTIONS[idx]

        return PolicyDecision(
            ts_ms=ctx.ts_ms,
            action=chosen.value,
            prob_wait=probs[0],
            prob_passive=probs[1],
            prob_aggressive=probs[2],
            weight_wait=probs[0],
            weight_passive=probs[1],
            weight_aggressive=probs[2],
        )

    def update(self, action: Action, loss: float) -> None:
        w = self._buckets[self._last_bucket]
        w[action] *= math.exp(-self._eta * loss)
        total = sum(w.values())
        for a in w:
            w[a] /= total

    def weights(self) -> Dict[Action, float]:
        return dict(self._buckets[self._last_bucket])

    def bucket_summary(self) -> List[Dict]:
        out: List[Dict] = []
        for pi in range(self._n_p):
            p_lo = "-inf"   if pi == 0                  else f"{self._p_edges[pi-1]:+.4f}"
            p_hi = "+inf"   if pi == len(self._p_edges) else f"{self._p_edges[pi]:+.4f}"
            for si in range(self._n_s):
                s_lo = "-inf" if si == 0                    else f"{self._s_edges[si-1]:+.4f}"
                s_hi = "+inf" if si == len(self._s_edges)   else f"{self._s_edges[si]:+.4f}"
                b = pi * self._n_s + si
                out.append({
                    "bucket":  b,
                    "p_range": f"({p_lo}, {p_hi}]",
                    "s_range": f"({s_lo}, {s_hi}]",
                    "s_label": self._sec_lbl,
                    "visits":  self._visits[b],
                    "weights": dict(self._buckets[b]),
                })
        return out
