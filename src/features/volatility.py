"""
Rolling realized volatility over multiple time windows.

Maintains a ring buffer of mid-price samples and computes the standard
deviation of log returns over three windows (1s, 5s, 30s by default).

Why log returns?
  Log returns are additive across time and approximately symmetric, making
  them better-behaved statistically than raw price differences.

  log_return_t = ln(mid_t / mid_{t-1})

Why multiple windows?
  - 1s vol  → short-term noise / microstructure friction
  - 5s vol  → medium-term regime signal
  - 30s vol → slower regime / trend context

These three numbers together let the Kalman filter distinguish:
  - Calm:    low vol across all windows
  - Spike:   high 1s, low 30s (transient noise)
  - Trending: rising vol across windows (sustained regime)

Implementation note:
  We use a simple list as a ring buffer keyed by (ts_ms, mid_price).
  Only samples within each window's lookback are used for the calculation.
  This means early in a session the windows may have too few samples —
  vol is returned as 0.0 in that case (caller can treat as NaN if needed).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class VolatilityFeatures:
    vol_1s: float    # realized vol over last 1 second
    vol_5s: float    # realized vol over last 5 seconds
    vol_30s: float   # realized vol over last 30 seconds


class RollingVolatility:
    """Maintains a time-indexed ring buffer of mid prices and emits vol."""

    # Minimum samples needed before vol is considered reliable.
    MIN_SAMPLES = 3

    def __init__(
        self,
        windows_ms: Tuple[int, int, int] = (1_000, 5_000, 30_000),
    ) -> None:
        self.windows_ms = windows_ms
        # Buffer stores (ts_ms, log_return) pairs.
        # We keep at most max_window worth of samples.
        self._max_window_ms = max(windows_ms)
        self._buffer: List[Tuple[int, float]] = []
        self._last_mid: float = 0.0

    def update(self, ts_ms: int, mid_price: float) -> VolatilityFeatures:
        """Push a new mid-price sample and return current vol estimates.

        Args:
            ts_ms:     timestamp of this sample in milliseconds
            mid_price: current mid-price

        Returns:
            VolatilityFeatures with vol_1s, vol_5s, vol_30s.
            Each is 0.0 until MIN_SAMPLES are available in that window.
        """
        if self._last_mid > 0:
            log_ret = math.log(mid_price / self._last_mid)
            self._buffer.append((ts_ms, log_ret))

        self._last_mid = mid_price

        # Prune samples older than the largest window
        cutoff = ts_ms - self._max_window_ms
        self._buffer = [(t, r) for t, r in self._buffer if t >= cutoff]

        vols = []
        for window_ms in self.windows_ms:
            window_cutoff = ts_ms - window_ms
            returns = [r for t, r in self._buffer if t >= window_cutoff]
            if len(returns) >= self.MIN_SAMPLES:
                vols.append(statistics.stdev(returns))
            else:
                vols.append(0.0)

        return VolatilityFeatures(vol_1s=vols[0], vol_5s=vols[1], vol_30s=vols[2])

    def reset(self) -> None:
        self._buffer.clear()
        self._last_mid = 0.0
