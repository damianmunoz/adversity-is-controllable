"""
Loss function — scores how well or badly an execution went.

The loss is a scalar fed directly into HedgePolicy.update(). Higher loss =
the action was costly = its weight gets shrunk. Lower (or negative) loss =
the action was good = its weight stays or grows.

Formula:
    L = slippage + λ * adverse_move

Where:

  slippage     — how far from mid we executed.
                 Positive: we paid above mid (aggressive, we crossed the spread).
                 Negative: we bought below mid (passive fill, we saved the spread).
                 Zero: no fill (wait or passive miss).

  adverse_move — how much the market moved against our position after the decision.
                 If FILLED:   max(0, fill_price - next_mid)
                   We bought at fill_price. If price fell below that, we overpaid.
                 If UNFILLED: max(0, next_mid - mid_price)
                   We didn't buy. If price rose, we missed a cheaper entry.

  λ (lambda_)  — how much to penalize timing mistakes vs execution price mistakes.
                 λ = 0.0: only care about slippage, ignore timing.
                 λ = 1.0: timing costs and slippage costs are equal.
                 λ = 0.5: default — both matter, slippage slightly more.

Why slippage can be negative (for passive fills):
  A passive fill at best bid buys below the mid price — that is a gain, not a cost.
  The loss function correctly returns a negative value, which causes
  exp(-η * L) > 1, so the passive action's weight INCREASES. The Hedge algorithm
  naturally learns to prefer actions that produce negative loss.
"""

from __future__ import annotations

from src.execution.simulator import FillResult


def compute_loss(fill: FillResult, lambda_: float) -> float:
    """Compute scalar loss for one simulated fill.

    Args:
        fill:    the FillResult from simulator.simulate_fill()
        lambda_: adverse move penalty weight (from configs/execution.yaml)

    Returns:
        Scalar loss. Negative = profitable action. Positive = costly action.
    """
    if fill.filled:
        # How far above mid we paid (negative = bought below mid)
        slippage = fill.fill_price - fill.mid_price

        # How much price fell after our fill (we bought high relative to next tick)
        adverse_move = max(0.0, fill.fill_price - fill.next_mid_price)
    else:
        # No fill — pure opportunity cost
        slippage = 0.0

        # How much price rose while we sat on our hands
        adverse_move = max(0.0, fill.next_mid_price - fill.mid_price)

    return slippage + lambda_ * adverse_move
