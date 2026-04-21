"""
Execution simulator — models what happens when we act against the order book.

We are always buying 1 unit of BTC. The simulator takes the action the policy
chose and the market state at the current and next tick, and returns a FillResult
describing what actually happened.

Three action outcomes:

  WAIT:       No order sent. No fill. The market moves on without us.
              If price went up, we missed a cheaper entry — opportunity cost.

  PASSIVE:    We post a limit buy order at the current best bid price.
              Fill condition: price moves DOWN next tick (a seller came to our
              price). If price moves UP, sellers are raising their asks and
              our stale bid at the old best bid goes unfilled.
              Fill price = current best bid (below mid — we saved the spread half).

  AGGRESSIVE: We send a market buy order immediately. Takes the cheapest
              available seller (best ask). Always fills. We pay above mid —
              the spread half is our guaranteed slippage.

Why this fill model for PASSIVE?
  In 1-second BTC/USDT ticks, if mid price moved up, aggressive buyers cleared
  the asks above us and sellers are now quoting higher. Our passive bid at the
  old price is below the new market — nobody sells to us. If mid moved down,
  sellers became more aggressive and likely hit our resting bid.
  This is a conservative but defensible simplification for a research simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.policy.actions import Action


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

@dataclass
class FillResult:
    """Outcome of one simulated execution attempt.

    fill_price is 0.0 when filled=False (no trade happened).
    mid_price is the mid at the moment of the decision (tick t).
    next_mid_price is the mid one tick later (tick t+1), used by loss.py.
    """
    ts_ms:           int
    action:          str
    filled:          bool
    fill_price:      float
    mid_price:       float
    next_mid_price:  float


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def simulate_fill(
    action:         Action,
    ts_ms:          int,
    curr_mid:       float,
    curr_best_bid:  float,
    curr_best_ask:  float,
    next_mid:       float,
) -> FillResult:
    """Simulate one fill attempt given the action and current/next market state.

    Args:
        action:        the action the policy selected
        ts_ms:         timestamp of the decision tick
        curr_mid:      mid price at decision time  ((best_bid + best_ask) / 2)
        curr_best_bid: best bid price at decision time  (our passive limit price)
        curr_best_ask: best ask price at decision time  (our aggressive fill price)
        next_mid:      mid price one tick later  (used to assess fill and adverse move)

    Returns:
        FillResult with outcome details for the loss function.
    """
    if action == Action.WAIT:
        return FillResult(
            ts_ms=ts_ms,
            action=action.value,
            filled=False,
            fill_price=0.0,
            mid_price=curr_mid,
            next_mid_price=next_mid,
        )

    elif action == Action.PASSIVE:
        # Post limit buy at best bid. Fills only if price moved down next tick.
        filled = next_mid <= curr_mid
        return FillResult(
            ts_ms=ts_ms,
            action=action.value,
            filled=filled,
            fill_price=curr_best_bid if filled else 0.0,
            mid_price=curr_mid,
            next_mid_price=next_mid,
        )

    else:  # AGGRESSIVE
        return FillResult(
            ts_ms=ts_ms,
            action=action.value,
            filled=True,
            fill_price=curr_best_ask,
            mid_price=curr_mid,
            next_mid_price=next_mid,
        )
