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

import math
from dataclasses import dataclass
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
    """Mirrors configs/policy.yaml."""
    learning_rate: float        # η — step size for the multiplicative update
    initial_weight: float = 1.0 # all actions start with equal weight


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
