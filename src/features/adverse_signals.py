"""
Adverse-risk rolling-window signals — candidates for the §24 second-axis
experiment (whose hypothesis is that the regime axis used by 2D bucketing
predicts FILL probability, not ADVERSE-MOVE risk; the right second axis
should predict adverse risk specifically).

Three stateful signal computers, each one tick of state, all sharing the
same interface:

    sig = SignalClass(window_sec=60.0)
    value = sig.update(ts_ms, scalar_input)

`update()` returns the current value of the signal AFTER absorbing the
new sample. Before enough history has accumulated (window not yet full),
`update()` returns 0.0 — a neutral signal that bucketing will treat as
"no information yet."

These signals are intentionally simple:
  - RollingVolDelta:    vol_30s_now − vol_30s_(now − window)
                        positive ⇒ vol is rising
  - RollingOFINetFlow:  sum of ofi_l1 over the trailing window
                        positive ⇒ sustained net buy pressure
                        negative ⇒ sustained net sell pressure
  - RollingSpreadDelta: spread_abs_now − spread_abs_(now − window)
                        positive ⇒ spread is widening (a "things-are-
                        getting-weird" signal — wider spread often
                        precedes adverse price moves)

Each signal is computed inline by the new replay loop in
`scripts/6_simulate_and_update_adverse.py` and the A/B runner; they do
NOT touch the canonical FeaturePipeline schema. That keeps the existing
features parquet format stable and the existing 1D / 2D scripts
reproducible.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Tuple


class _RollingDelta:
    """Common base — keeps a deque of (ts_ms, value) and on update returns
    `current_value − oldest_value_inside_window`. While the window has not
    yet covered `window_sec` worth of history, returns 0.0.
    """

    def __init__(self, window_sec: float = 60.0) -> None:
        if window_sec <= 0:
            raise ValueError("window_sec must be positive")
        self._window_ms: int = int(window_sec * 1000)
        self._buf: Deque[Tuple[int, float]] = deque()
        self._first_ts_ms: int | None = None

    def update(self, ts_ms: int, value: float) -> float:
        if self._first_ts_ms is None:
            self._first_ts_ms = ts_ms
        # Trim entries older than (ts_ms − window).
        cutoff = ts_ms - self._window_ms
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()
        self._buf.append((ts_ms, value))
        # Until the buffer spans the full window from the first observation
        # we have seen, return neutral 0.0 — we don't yet have a fair
        # baseline to compare against.
        if ts_ms - self._first_ts_ms < self._window_ms:
            return 0.0
        oldest_value = self._buf[0][1]
        return value - oldest_value


class RollingVolDelta(_RollingDelta):
    """vol_30s_now − vol_30s_(now − window).

    Feed this with `vol_30s` per tick. Positive ⇒ market volatility is
    *rising*. The hypothesis for §24 is that rising volatility precedes
    adverse moves more reliably than current high volatility does.
    """


class RollingSpreadDelta(_RollingDelta):
    """spread_abs_now − spread_abs_(now − window).

    Feed this with `spread_abs` per tick. Positive ⇒ spread is widening,
    often a sign that liquidity is thinning before an adverse move.
    """


class RollingOFINetFlow:
    """Sum of ofi_l1 over the trailing `window_sec` seconds.

    Feed this with `ofi_l1` per tick. Positive ⇒ sustained net buy
    pressure over the window; negative ⇒ sustained net sell pressure.
    Different from the Kalman `market_pressure` (which is ALSO derived
    from ofi_l1 + depth_imbalance) because it's a longer-horizon
    accumulation, not an instantaneous filtered estimate.
    """

    def __init__(self, window_sec: float = 60.0) -> None:
        if window_sec <= 0:
            raise ValueError("window_sec must be positive")
        self._window_ms: int = int(window_sec * 1000)
        self._buf: Deque[Tuple[int, float]] = deque()
        self._sum: float = 0.0
        self._first_ts_ms: int | None = None

    def update(self, ts_ms: int, value: float) -> float:
        if self._first_ts_ms is None:
            self._first_ts_ms = ts_ms
        cutoff = ts_ms - self._window_ms
        while self._buf and self._buf[0][0] < cutoff:
            _, old = self._buf.popleft()
            self._sum -= old
        self._buf.append((ts_ms, value))
        self._sum += value
        if ts_ms - self._first_ts_ms < self._window_ms:
            return 0.0
        return self._sum


SIGNAL_REGISTRY = {
    # name → (class, input_feature_name, human_label)
    "vol_delta":    (RollingVolDelta,    "vol_30s",    "vol_30s rising"),
    "ofi_window":   (RollingOFINetFlow,  "ofi_l1",     "60s net OFI"),
    "spread_delta": (RollingSpreadDelta, "spread_abs", "spread widening"),
}
