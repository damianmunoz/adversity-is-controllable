"""
Hedge policy — action selection via exponential weights (EXP3 / Hedge).

This is Step 5 in the pipeline. It sits between the Kalman filter (Step 4)
and the execution simulator (Step 6).

The problem being solved:
  At each market tick we must choose one of three actions (WAIT, PASSIVE,
  AGGRESSIVE) without knowing which will be best. After acting, we observe
  a scalar loss (Step 6). We want to minimize cumulative regret — i.e. perform
  nearly as well as the best fixed action in hindsight.

The Hedge algorithm solves this with a simple rule:
  - Maintain a weight w(a) for each action, starting equal.
  - Select action proportionally to weights (higher weight = selected more often).
  - After observing loss L for the action taken, shrink its weight:
      w(a) ← w(a) * exp(-η * L)
  - Renormalize so weights sum to 1.

Intuition: actions that repeatedly incur high loss get down-weighted
exponentially. Actions with low loss keep their share. The learning rate η
controls how fast this happens — too high and the policy overreacts to noise,
too low and it adapts too slowly.

This is the bandit variant (EXP3): we only observe the loss for the action
we actually took, not for all actions. That is the realistic assumption for
execution — you don't know what would have happened if you had waited.

The policy is intentionally stateless about market features. The Kalman filter
already compresses market state into two numbers; the policy learns from
realized losses what those states imply for execution cost — no handcrafted rules.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.policy.actions import Action
from src.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PolicyConfig:
    """Mirrors configs/policy.yaml.

    `mode` selects the policy variant:
      "marginal"    — classical Hedge with a single weight vector (default).
      "bucketed"    — BucketedHedgePolicy: one Hedge per market_pressure
                      bucket. Requires `pressure_edges`.
      "bucketed_2d" — BucketedHedge2DPolicy: one Hedge per
                      (market_pressure, regime) bucket. Requires both
                      `pressure_edges` and `regime_edges`.
    """
    learning_rate:  float        # η — step size for the multiplicative update
    initial_weight: float        = 1.0
    mode:           str          = "marginal"
    pressure_edges: List[float]  = field(default_factory=list)
    regime_edges:   List[float]  = field(default_factory=list)


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------

@dataclass
class PolicyContext:
    """What the policy sees before making a decision each tick.

    Bundles the Kalman state with a timestamp. Used by the outer loop
    (Step 6) to pass context to the policy without coupling it to the
    full FeatureVector schema.

    market_pressure and regime come directly from KalmanState.
    """
    ts_ms: int
    market_pressure: float
    regime: float


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

@dataclass
class PolicyDecision:
    """The policy's decision at a single tick, with metadata.

    Flat dataclass — written to the action log Parquet in script 5.

    action:            the selected action as a string ("WAIT" etc.)
    prob_wait:         probability assigned to WAIT at selection time
    prob_passive:      probability assigned to PASSIVE
    prob_aggressive:   probability assigned to AGGRESSIVE
    weight_wait etc.:  raw (normalized) weights — same as probs here, kept
                       separately for clarity when comparing across ticks
    """
    ts_ms: int
    action: str
    prob_wait: float
    prob_passive: float
    prob_aggressive: float
    weight_wait: float
    weight_passive: float
    weight_aggressive: float


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

_ACTIONS: List[Action] = [Action.WAIT, Action.PASSIVE, Action.AGGRESSIVE]


class HedgePolicy:
    """Stateful exponential-weights policy.

    Must be instantiated once and driven by the outer loop in script 6.
    select() and update() must alternate: select once per tick, update
    once per tick after the simulator returns a loss.

    Usage:
        policy = HedgePolicy(config, seed=42)

        # selection
        ctx     = PolicyContext(ts_ms, kalman.market_pressure, kalman.regime)
        decision = policy.select(ctx)

        # ... simulate fill, compute loss ...

        # weight update
        policy.update(Action(decision.action), loss)
    """

    def __init__(self, config: PolicyConfig, seed: Optional[int] = None) -> None:
        self.config = config
        self._rng   = np.random.default_rng(seed)

        # Weights are stored normalized so they always form a probability distribution.
        w = config.initial_weight
        total = w * len(_ACTIONS)
        self._weights: Dict[Action, float] = {a: w / total for a in _ACTIONS}

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self, ctx: PolicyContext) -> PolicyDecision:
        """Sample an action from the current weight distribution.

        The context (market_pressure, regime) is not used directly in the
        selection — the weights encode everything the policy has learned.
        The context is passed here so future variants can condition on it
        (e.g. contextual bandit) without changing the interface.
        """
        probs = [self._weights[a] for a in _ACTIONS]
        idx   = int(self._rng.choice(len(_ACTIONS), p=probs))
        chosen = _ACTIONS[idx]

        log.debug(
            "ts=%d select=%s probs=[%.3f, %.3f, %.3f]",
            ctx.ts_ms, chosen.value, probs[0], probs[1], probs[2],
        )

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

    # ------------------------------------------------------------------
    # Weight update
    # ------------------------------------------------------------------

    def update(self, action: Action, loss: float) -> None:
        """Apply multiplicative weight update for the action that was taken.

        Only the taken action's weight changes (bandit feedback).
        After the update, weights are renormalized so they sum to 1,
        which prevents numerical underflow over many ticks.

        A loss of 0 leaves the weight unchanged (exp(0) = 1).
        A large positive loss shrinks the weight.
        Negative loss (a gain) grows the weight — appropriate when the
        simulator returns signed PnL-based loss.
        """
        self._weights[action] *= math.exp(-self.config.learning_rate * loss)

        # Renormalize
        total = sum(self._weights.values())
        for a in self._weights:
            self._weights[a] /= total

        log.debug(
            "update action=%s loss=%.4f → weights=[%.3f, %.3f, %.3f]",
            action.value, loss,
            self._weights[Action.WAIT],
            self._weights[Action.PASSIVE],
            self._weights[Action.AGGRESSIVE],
        )

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def weights(self) -> Dict[Action, float]:
        """Return a copy of the current weight distribution."""
        return dict(self._weights)


# ---------------------------------------------------------------------------
# Bucketed (state-conditioned) Hedge
# ---------------------------------------------------------------------------

class BucketedHedgePolicy:
    """State-conditioned Hedge: one Hedge weight vector per market_pressure bucket.

    Rationale:
      Classical Hedge (above) learns a single marginal distribution over
      actions. It never looks at market_pressure or regime when selecting.
      The Kalman filter's output is therefore decorative in the baseline.

      BucketedHedgePolicy discretizes market_pressure into N buckets using
      config.pressure_edges (N-1 cutpoints) and maintains one Hedge weight
      vector per bucket. At selection time the current bucket is looked up
      from ctx.market_pressure; action is sampled from THAT bucket's weights.
      On update, only that bucket's weights change. Each bucket therefore
      runs its own regret-minimization and the classical Hedge guarantees
      hold WITHIN each bucket.

      Trade-off: data per bucket = total_ticks / N. With 38k ticks and 5
      buckets that's ~7.6k ticks each — still enough for a few hundred
      exponential updates to converge.

    Interface is identical to HedgePolicy: select(ctx) / update(action, loss) /
    weights(). The `weights()` method returns the MOST RECENTLY USED bucket's
    weights so the outer loop's "final weights" log line is well-defined;
    call `bucket_summary()` for the full picture across buckets.
    """

    def __init__(self, config: PolicyConfig, seed: Optional[int] = None) -> None:
        if config.mode != "bucketed":
            raise ValueError(
                f"BucketedHedgePolicy requires mode='bucketed', got '{config.mode}'"
            )
        if not config.pressure_edges:
            raise ValueError(
                "BucketedHedgePolicy requires non-empty pressure_edges "
                "(N-1 cutpoints → N buckets)"
            )
        # Edges must be strictly ascending so bisect returns a meaningful index
        if list(config.pressure_edges) != sorted(config.pressure_edges):
            raise ValueError("pressure_edges must be in ascending order")

        self.config = config
        self._rng   = np.random.default_rng(seed)
        self._edges = list(config.pressure_edges)
        n_buckets   = len(self._edges) + 1

        w_init = config.initial_weight
        uniform_total = w_init * len(_ACTIONS)
        self._buckets: List[Dict[Action, float]] = [
            {a: w_init / uniform_total for a in _ACTIONS}
            for _ in range(n_buckets)
        ]
        self._visits: List[int] = [0] * n_buckets
        self._last_bucket: int  = 0  # valid after the first select()

    # ------------------------------------------------------------------
    # Bucket lookup
    # ------------------------------------------------------------------

    def _which_bucket(self, market_pressure: float) -> int:
        """Return bucket index for a given pressure value.

        bisect_right gives N possible return values for N-1 edges:
          pressure <= edges[0]                        → 0
          edges[i-1] < pressure <= edges[i]           → i
          pressure >  edges[-1]                       → len(edges)
        """
        return bisect.bisect_right(self._edges, market_pressure)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self, ctx: PolicyContext) -> PolicyDecision:
        bucket = self._which_bucket(ctx.market_pressure)
        self._last_bucket = bucket
        self._visits[bucket] += 1

        w = self._buckets[bucket]
        probs = [w[a] for a in _ACTIONS]
        idx = int(self._rng.choice(len(_ACTIONS), p=probs))
        chosen = _ACTIONS[idx]

        log.debug(
            "ts=%d bucket=%d pressure=%.3f select=%s probs=[%.3f,%.3f,%.3f]",
            ctx.ts_ms, bucket, ctx.market_pressure, chosen.value,
            probs[0], probs[1], probs[2],
        )

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

    # ------------------------------------------------------------------
    # Weight update (only the bucket used for selection updates)
    # ------------------------------------------------------------------

    def update(self, action: Action, loss: float) -> None:
        w = self._buckets[self._last_bucket]
        w[action] *= math.exp(-self.config.learning_rate * loss)
        total = sum(w.values())
        for a in w:
            w[a] /= total

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def weights(self) -> Dict[Action, float]:
        """Weights of the most recently selected bucket (for parity with HedgePolicy)."""
        return dict(self._buckets[self._last_bucket])

    def bucket_summary(self) -> List[Dict]:
        """Per-bucket visit count and final weights. Use this for post-run analysis."""
        summaries: List[Dict] = []
        for i, (w, visits) in enumerate(zip(self._buckets, self._visits)):
            # Describe the pressure range covered by this bucket
            lo = "-inf"         if i == 0                 else f"{self._edges[i-1]:+.3f}"
            hi = "+inf"         if i == len(self._edges)  else f"{self._edges[i]:+.3f}"
            summaries.append({
                "bucket":  i,
                "range":   f"({lo}, {hi}]",
                "visits":  visits,
                "weights": dict(w),
            })
        return summaries


# ---------------------------------------------------------------------------
# 2D bucketed (state-conditioned on (market_pressure, regime))
# ---------------------------------------------------------------------------

class BucketedHedge2DPolicy:
    """State-conditioned Hedge with TWO state dimensions.

    The 1D BucketedHedgePolicy buckets only on market_pressure. With the
    regime dimension now alive (sanity.txt §16), we can finally condition on
    BOTH state estimates at once: maintain one Hedge weight vector per
    (pressure_bucket, regime_bucket) cell of a 2D grid.

    Why this might help:
      Pressure tells you DIRECTION (people pushing up vs down). Regime tells
      you VOLATILITY (calm vs turbulent). The optimal action is plausibly
      different in "heavy buy + calm" (linger as PASSIVE, no rush) vs
      "heavy buy + turbulent" (go AGGRESSIVE before the price runs away).
      A 1D pressure-only policy averages those two situations together.

    Why this might NOT help:
      Splitting buckets thins the data per bucket. With 6 pressure buckets
      and 3 regime buckets we get 18 cells; on 38k ticks that is ~2k per
      cell on average, but in practice a few cells will be sparse (especially
      the corners — extreme pressure AND extreme regime simultaneously).
      Sparse buckets give noisy weights. If regime carries no useful
      conditional information, 2D will lose to 1D simply because each cell
      has fewer updates to learn from.

    Storage layout:
      Buckets are flattened: bucket_index = p_idx * n_regime + r_idx.

    Interface matches HedgePolicy / BucketedHedgePolicy: select(), update(),
    weights() (returns the most recently selected bucket's weights). Adds
    bucket_summary() for full post-run inspection of all 2D cells.
    """

    def __init__(self, config: PolicyConfig, seed: Optional[int] = None) -> None:
        if config.mode != "bucketed_2d":
            raise ValueError(
                f"BucketedHedge2DPolicy requires mode='bucketed_2d', got '{config.mode}'"
            )
        if not config.pressure_edges:
            raise ValueError("BucketedHedge2DPolicy requires non-empty pressure_edges")
        if not config.regime_edges:
            raise ValueError("BucketedHedge2DPolicy requires non-empty regime_edges")
        if list(config.pressure_edges) != sorted(config.pressure_edges):
            raise ValueError("pressure_edges must be in ascending order")
        if list(config.regime_edges) != sorted(config.regime_edges):
            raise ValueError("regime_edges must be in ascending order")

        self.config   = config
        self._rng     = np.random.default_rng(seed)
        self._p_edges = list(config.pressure_edges)
        self._r_edges = list(config.regime_edges)
        self._n_p     = len(self._p_edges) + 1
        self._n_r     = len(self._r_edges) + 1
        n_total       = self._n_p * self._n_r

        w_init = config.initial_weight
        uniform_total = w_init * len(_ACTIONS)
        self._buckets: List[Dict[Action, float]] = [
            {a: w_init / uniform_total for a in _ACTIONS}
            for _ in range(n_total)
        ]
        self._visits: List[int] = [0] * n_total
        self._last_bucket: int  = 0  # valid after the first select()

    def _which_bucket(self, market_pressure: float, regime: float) -> int:
        pi = bisect.bisect_right(self._p_edges, market_pressure)
        ri = bisect.bisect_right(self._r_edges, regime)
        return pi * self._n_r + ri

    def select(self, ctx: PolicyContext) -> PolicyDecision:
        bucket = self._which_bucket(ctx.market_pressure, ctx.regime)
        self._last_bucket = bucket
        self._visits[bucket] += 1

        w = self._buckets[bucket]
        probs = [w[a] for a in _ACTIONS]
        idx = int(self._rng.choice(len(_ACTIONS), p=probs))
        chosen = _ACTIONS[idx]

        log.debug(
            "ts=%d bucket=%d pressure=%.3f regime=%.3f select=%s probs=[%.3f,%.3f,%.3f]",
            ctx.ts_ms, bucket, ctx.market_pressure, ctx.regime, chosen.value,
            probs[0], probs[1], probs[2],
        )

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
        w[action] *= math.exp(-self.config.learning_rate * loss)
        total = sum(w.values())
        for a in w:
            w[a] /= total

    def weights(self) -> Dict[Action, float]:
        return dict(self._buckets[self._last_bucket])

    def bucket_summary(self) -> List[Dict]:
        """Per-(pressure, regime)-bucket visit counts and final weights."""
        summaries: List[Dict] = []
        for pi in range(self._n_p):
            p_lo = "-inf"        if pi == 0                  else f"{self._p_edges[pi-1]:+.3f}"
            p_hi = "+inf"        if pi == len(self._p_edges) else f"{self._p_edges[pi]:+.3f}"
            for ri in range(self._n_r):
                r_lo = "-inf"    if ri == 0                  else f"{self._r_edges[ri-1]:+.3f}"
                r_hi = "+inf"    if ri == len(self._r_edges) else f"{self._r_edges[ri]:+.3f}"
                b = pi * self._n_r + ri
                summaries.append({
                    "bucket":  b,
                    "p_range": f"({p_lo}, {p_hi}]",
                    "r_range": f"({r_lo}, {r_hi}]",
                    "visits":  self._visits[b],
                    "weights": dict(self._buckets[b]),
                })
        return summaries


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_policy(config: PolicyConfig, seed: Optional[int] = None):
    """Instantiate the policy variant selected by config.mode.

    Returns an object with select()/update()/weights() compatible with all
    variants. Bucketed variants additionally expose bucket_summary().
    """
    mode = (config.mode or "marginal").lower()
    if mode == "marginal":
        return HedgePolicy(config, seed=seed)
    if mode == "bucketed":
        return BucketedHedgePolicy(config, seed=seed)
    if mode == "bucketed_2d":
        return BucketedHedge2DPolicy(config, seed=seed)
    raise ValueError(
        f"Unknown policy mode '{config.mode}'. "
        f"Expected 'marginal', 'bucketed', or 'bucketed_2d'."
    )
