"""
Depth imbalance — normalized directional pressure signal.

Depth imbalance aggregates the quantity on both sides of the book across
the top N levels and returns a single number in the range (-1, +1):

    imbalance = (sum_bid_qty - sum_ask_qty) / (sum_bid_qty + sum_ask_qty)

Interpretation:
  +1.0 → all liquidity is on the bid side (strong buy pressure)
  -1.0 → all liquidity is on the ask side (strong sell pressure)
   0.0 → perfectly balanced

Why normalize?
  Raw bid/ask quantities are in BTC and vary by orders of magnitude across
  market regimes. The normalized ratio strips out absolute scale and gives
  the Kalman filter a consistent-range observation regardless of whether
  the book is thin or deep.

Optional volume weighting:
  Levels far from the mid are less likely to trade and so contribute less
  to near-term price pressure. Setting volume_weighted=True applies a
  linear decay: level 1 gets weight N, level 2 gets N-1, ..., level N gets 1.
  This down-weights deep levels without discarding them entirely.
"""

from __future__ import annotations

from typing import List, Tuple


def compute_imbalance(
    bids: List[Tuple[float, float]],
    asks: List[Tuple[float, float]],
    n_levels: int = 10,
    volume_weighted: bool = True,
) -> float:
    """Compute depth imbalance over top n_levels.

    Args:
        bids: top-N bid levels as (price, qty) tuples, best first
        asks: top-N ask levels as (price, qty) tuples, best first
        n_levels: how many levels to include
        volume_weighted: if True, apply linear decay weights to deeper levels

    Returns:
        Float in (-1, +1). Returns 0.0 if book is empty on either side.
    """
    bid_levels = bids[:n_levels]
    ask_levels = asks[:n_levels]

    if not bid_levels or not ask_levels:
        return 0.0

    n = max(len(bid_levels), len(ask_levels))

    def weighted_qty(levels: List[Tuple[float, float]]) -> float:
        total = 0.0
        for i, (_, qty) in enumerate(levels):
            weight = (n - i) if volume_weighted else 1.0
            total += qty * weight
        return total

    bid_total = weighted_qty(bid_levels)
    ask_total = weighted_qty(ask_levels)
    denom = bid_total + ask_total

    if denom == 0.0:
        return 0.0

    return (bid_total - ask_total) / denom
