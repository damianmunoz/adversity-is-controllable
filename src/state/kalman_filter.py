"""
Kalman filter — estimates hidden market state from observable features.

This is Step 4 in the pipeline. It sits between the feature layer (Step 3)
and the policy layer (Step 5).

What it models:
  Hidden state x = [market_pressure, regime]  (2D, unobservable)
  Observations  z = [depth_imbalance, ofi_l1, vol_5s]  (3D, from FeatureVector)

The filter does two things every tick:
  1. Predict — project the previous state forward using the transition matrix F
     (market pressure decays; regime is stickier)
  2. Update  — correct the prediction using the new observation z via Bayes rule

The output is a KalmanState: the posterior mean x and the diagonal + cross term
of the posterior covariance P (how confident the filter is in each estimate).

This is a standard linear Kalman filter. No approximations. Closed-form.
All parameters come from configs/kalman.yaml.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class KalmanConfig:
    """Mirrors configs/kalman.yaml exactly.

    All matrix fields arrive from YAML as Python list-of-lists.
    KalmanFilter.__init__ converts them to numpy arrays.
    """
    state_dim: int
    obs_dim: int
    obs_features: List[str]
    F: list
    H: list
    Q: list
    R: list
    x0: list
    P0: list


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

@dataclass
class KalmanState:
    """One row of Kalman output per feature tick.

    Flat dataclass — no nested arrays — so it writes directly to Parquet.

    market_pressure: estimated buying/selling pressure (can be negative)
    regime:          estimated market turbulence level (higher = more volatile)
    p00:             posterior variance of market_pressure (filter's confidence)
    p11:             posterior variance of regime
    p01:             cross-covariance (how correlated the two estimates are)
    """
    ts_ms: int
    market_pressure: float
    regime: float
    p00: float
    p11: float
    p01: float


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

class KalmanFilter:
    """Stateful linear Kalman filter.

    Must be instantiated once and fed FeatureVectors in strict chronological
    order. Internal state (_x, _P) carries over from tick to tick — do not
    reset mid-run unless you intend to restart estimation from scratch.

    Usage:
        cfg = load_config("configs/kalman.yaml", KalmanConfig)
        kf  = KalmanFilter(cfg)

        for fv in feature_vectors:
            state = kf.step(fv.ts_ms, z)  # z built from fv by the caller
            if state is not None:
                # use state.market_pressure, state.regime
    """

    def __init__(self, config: KalmanConfig) -> None:
        self.config = config

        # Convert list-of-lists from YAML → numpy arrays once at init
        self.F = np.array(config.F, dtype=float)   # (state_dim, state_dim)
        self.H = np.array(config.H, dtype=float)   # (obs_dim,   state_dim)
        self.Q = np.array(config.Q, dtype=float)   # (state_dim, state_dim)
        self.R = np.array(config.R, dtype=float)   # (obs_dim,   obs_dim)

        # Internal state: posterior mean and covariance
        self._x: np.ndarray = np.array(config.x0, dtype=float)   # (state_dim,)
        self._P: np.ndarray = np.array(config.P0, dtype=float)   # (state_dim, state_dim)

        # Identity matrix reused every update step
        self._I = np.eye(config.state_dim)

    # ------------------------------------------------------------------
    # Core equations
    # ------------------------------------------------------------------

    def _predict(self) -> tuple[np.ndarray, np.ndarray]:
        """Predict step: project state forward one tick.

        x_pred = F @ x          — apply state transition
        P_pred = F @ P @ F.T + Q — propagate uncertainty + add process noise
        """
        x_pred = self.F @ self._x
        P_pred = self.F @ self._P @ self.F.T + self.Q
        return x_pred, P_pred

    def _update(
        self,
        z: np.ndarray,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
    ) -> None:
        """Update step: correct prediction using new observation z.

        y = z - H @ x_pred          — innovation (how wrong the prediction was)
        S = H @ P_pred @ H.T + R    — innovation covariance
        K = P_pred @ H.T @ inv(S)   — Kalman gain (how much to trust z vs prediction)
        x = x_pred + K @ y          — posterior mean
        P = (I - K @ H) @ P_pred    — posterior covariance (Joseph form omitted for speed)
        """
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self._x = x_pred + K @ y
        self._P = (self._I - K @ self.H) @ P_pred

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, ts_ms: int, z: np.ndarray) -> Optional[KalmanState]:
        """Run one predict-update cycle and return the posterior state.

        z must be a 1D numpy array of length obs_dim, ordered to match
        obs_features in kalman.yaml: [depth_imbalance, ofi_l1, vol_5s].

        Returns None if z contains NaN (feature not yet available, e.g.
        vol_5s on the first few ticks). The internal state is NOT updated
        on a skipped tick so the filter resumes cleanly on the next valid z.
        """
        if np.any(np.isnan(z)):
            log.debug("Skipping ts=%d — NaN in observation vector.", ts_ms)
            return None

        x_pred, P_pred = self._predict()
        self._update(z, x_pred, P_pred)

        return KalmanState(
            ts_ms=ts_ms,
            market_pressure=float(self._x[0]),
            regime=float(self._x[1]),
            p00=float(self._P[0, 0]),
            p11=float(self._P[1, 1]),
            p01=float(self._P[0, 1]),
        )
