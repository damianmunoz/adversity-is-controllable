"""
Order Flow Imbalance (OFI) — per price level.

OFI measures the net pressure at each price level by comparing how the
quantity at that level changed between two consecutive book states:

    OFI_i = delta_bid_qty_i - delta_ask_qty_i

Where delta = current_qty - previous_qty for the price level at rank i.

Interpretation:
  - OFI > 0 → more buying pressure (bids grew or asks shrank at that level)
  - OFI < 0 → more selling pressure (asks grew or bids shrank)
  - OFI ≈ 0 → balanced, no directional pressure

This follows the multi-level OFI definition from Bouchaud et al.
"How markets slowly digest changes in supply and demand."

We compute for the top N levels (default 5). Level 1 is the best bid/ask.
"""

from __future__ import annotations

from typing import List, Tuple


def _qty_at_price(levels: List[Tuple[float, float]], rank: int) -> float:
    """Return qty at a given rank (0-indexed). 0.0 if level doesn't exist."""
    if rank < len(levels):
        return levels[rank][1]
    return 0.0


def compute_ofi(
    prev_bids: List[Tuple[float, float]],
    prev_asks: List[Tuple[float, float]],
    curr_bids: List[Tuple[float, float]],
    curr_asks: List[Tuple[float, float]],
    n_levels: int = 5,
) -> List[float]:
    """Compute OFI for the top n_levels price levels.

    Args:
        prev_bids / prev_asks: top-N levels from the previous BookSnapshot
        curr_bids / curr_asks: top-N levels from the current BookSnapshot
        n_levels: how many levels to compute (produces a list of length n_levels)

    Returns:
        List of OFI values [ofi_l1, ofi_l2, ..., ofi_ln].
        Index 0 = best bid/ask level (L1), index 1 = L2, etc.

    Note on price alignment:
        This implementation computes OFI by rank (position in the sorted list),
        not by exact price match. This is a simplification that works well when
        the top-of-book is stable. A stricter implementation would match by price
        across snapshots — that becomes important when the book shifts significantly
        between ticks. For 100ms updates on BTC/USDT this rank-based approach is
        sufficient.
    """
    ofi_values = []
    for i in range(n_levels):
        delta_bid = _qty_at_price(curr_bids, i) - _qty_at_price(prev_bids, i)
        delta_ask = _qty_at_price(curr_asks, i) - _qty_at_price(prev_asks, i)
        ofi_values.append(delta_bid - delta_ask)
    return ofi_values
