"""
Spread and price features derived from the top of book.

Computes:
  - best bid / best ask
  - quoted spread (absolute and in basis points)
  - simple mid price
  - microprice (qty-weighted mid — a better price signal for state estimation)

Microprice intuition:
  If the best bid has 0.1 BTC and the best ask has 10 BTC, the market is
  much more likely to trade down (exhaust the thin bid) than up. Microprice
  reflects this by weighting the mid toward the side with LESS quantity.

  microprice = (bid_qty * ask_px + ask_qty * bid_px) / (bid_qty + ask_qty)

  This is the same quantity calculated in OrderBook.microprice() but here
  it operates directly on a BookSnapshot (no live book needed), which is
  what the feature pipeline works with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class SpreadFeatures:
    best_bid_px: float
    best_ask_px: float
    spread_abs: float       # ask - bid, in quote currency (USDT)
    spread_bps: float       # spread as basis points of mid: (spread / mid) * 10000
    mid_price: float        # simple average of best bid and best ask
    microprice: float       # quantity-weighted mid


def compute_spread(
    bids: List[Tuple[float, float]],
    asks: List[Tuple[float, float]],
) -> Optional[SpreadFeatures]:
    """Compute spread features from top-of-book bid/ask levels.

    Returns None if either side is empty (shouldn't happen on a live feed
    but can occur during replay of gaps).
    """
    if not bids or not asks:
        return None

    bid_px, bid_qty = bids[0]
    ask_px, ask_qty = asks[0]

    if bid_qty + ask_qty == 0:
        return None

    spread_abs = ask_px - bid_px
    mid = (bid_px + ask_px) / 2.0
    spread_bps = (spread_abs / mid) * 10_000 if mid > 0 else 0.0
    microprice = (bid_qty * ask_px + ask_qty * bid_px) / (bid_qty + ask_qty)

    return SpreadFeatures(
        best_bid_px=bid_px,
        best_ask_px=ask_px,
        spread_abs=spread_abs,
        spread_bps=spread_bps,
        mid_price=mid,
        microprice=microprice,
    )
