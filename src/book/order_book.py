"""
In-memory limit order book (LOB).

Maintains two price-sorted sides (bids descending, asks ascending) as
SortedDict objects. Handles:
  - Full snapshot initialization from a Binance REST response
  - Incremental depth update application with sequence gap detection
  - Read access: best bid/ask, mid-price, microprice, top-N levels

Why SortedDict?
  Binance depth updates arrive at arbitrary price levels, not just the top.
  We need O(log n) insert/delete and O(1) min/max (best bid/ask).
  Python's built-in dict gives neither in sorted order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from sortedcontainers import SortedDict


@dataclass
class PriceLevel:
    price: float
    qty: float


@dataclass
class BookSnapshot:
    """Compact snapshot of the top-N book levels at a point in time.

    This is the output contract of OrderBook and the input contract
    of the feature pipeline. Keeping it to top-N levels (not all levels)
    caps memory and Parquet size.
    """
    ts_ms: int
    symbol: str
    last_update_id: int
    bids: List[Tuple[float, float]]   # (price, qty) descending
    asks: List[Tuple[float, float]]   # (price, qty) ascending
    sequence_gap: bool = False        # True if a depth update ID gap was detected


class OrderBook:
    """Price-sorted limit order book.

    Bids are stored in a SortedDict with negated keys so the best bid
    (highest price) is always at index 0 — SortedDict is ascending by
    default, so negation gives us descending order for free.

    Asks are stored with positive keys so the best ask (lowest price)
    is at index 0 naturally.
    """

    def __init__(self, symbol: str, top_n: int = 20) -> None:
        self.symbol = symbol
        self.top_n = top_n
        self.last_update_id: int = 0
        # bids: key = -price so SortedDict[0] is best bid
        self._bids: SortedDict = SortedDict()
        # asks: key = +price so SortedDict[0] is best ask
        self._asks: SortedDict = SortedDict()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def apply_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Seed the book from a Binance REST depth snapshot.

        Clears existing state entirely before applying, so calling this
        again mid-session resets the book cleanly.
        """
        self._bids.clear()
        self._asks.clear()
        self.last_update_id = int(snapshot["last_update_id"])

        for px, qty in snapshot["bids"]:
            px, qty = float(px), float(qty)
            if qty > 0:
                self._bids[-px] = qty

        for px, qty in snapshot["asks"]:
            px, qty = float(px), float(qty)
            if qty > 0:
                self._asks[px] = qty

    # ------------------------------------------------------------------
    # Incremental updates
    # ------------------------------------------------------------------

    def apply_depth_update(self, event: Dict[str, Any]) -> bool:
        """Apply one incremental depth update event.

        Returns True if the update was applied cleanly.
        Returns False if a sequence gap was detected (caller should log/flag).

        Binance sequence rule:
          - Drop events where final_update_id <= last_update_id (already seen)
          - Apply events where first_update_id <= last_update_id + 1
          - Flag a gap if first_update_id > last_update_id + 1
        """
        first_uid = int(event["first_update_id"])
        final_uid = int(event["final_update_id"])

        # Already processed
        if final_uid <= self.last_update_id:
            return True

        gap = first_uid > self.last_update_id + 1

        for px, qty in event["bids"]:
            px, qty = float(px), float(qty)
            if qty == 0.0:
                self._bids.pop(-px, None)
            else:
                self._bids[-px] = qty

        for px, qty in event["asks"]:
            px, qty = float(px), float(qty)
            if qty == 0.0:
                self._asks.pop(px, None)
            else:
                self._asks[px] = qty

        self.last_update_id = final_uid
        return not gap

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    def best_bid(self) -> Optional[PriceLevel]:
        if not self._bids:
            return None
        neg_px, qty = self._bids.peekitem(0)
        return PriceLevel(price=-neg_px, qty=qty)

    def best_ask(self) -> Optional[PriceLevel]:
        if not self._asks:
            return None
        px, qty = self._asks.peekitem(0)
        return PriceLevel(price=px, qty=qty)

    def mid_price(self) -> Optional[float]:
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return None
        return (bid.price + ask.price) / 2.0

    def microprice(self) -> Optional[float]:
        """Quantity-weighted mid price — a better price estimate than simple mid.

        microprice = (bid_qty * ask_px + ask_qty * bid_px) / (bid_qty + ask_qty)

        Weighs the mid toward the side with less quantity, because that side
        is more likely to be exhausted first.
        """
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return None
        total = bid.qty + ask.qty
        if total == 0:
            return None
        return (bid.qty * ask.price + ask.qty * bid.price) / total

    def spread(self) -> Optional[float]:
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return None
        return ask.price - bid.price

    def levels(self, side: str, n: int) -> List[PriceLevel]:
        """Return top-n price levels for 'bids' or 'asks'."""
        if side == "bids":
            items = self._bids.items()
            return [PriceLevel(price=-k, qty=v) for k, v in list(items)[:n]]
        elif side == "asks":
            items = self._asks.items()
            return [PriceLevel(price=k, qty=v) for k, v in list(items)[:n]]
        else:
            raise ValueError(f"side must be 'bids' or 'asks', got '{side}'")

    # ------------------------------------------------------------------
    # Snapshot export
    # ------------------------------------------------------------------

    def to_snapshot(self, ts_ms: int, sequence_gap: bool = False) -> BookSnapshot:
        """Export current top-N state as a BookSnapshot for downstream use."""
        bids = [(lvl.price, lvl.qty) for lvl in self.levels("bids", self.top_n)]
        asks = [(lvl.price, lvl.qty) for lvl in self.levels("asks", self.top_n)]
        return BookSnapshot(
            ts_ms=ts_ms,
            symbol=self.symbol,
            last_update_id=self.last_update_id,
            bids=bids,
            asks=asks,
            sequence_gap=sequence_gap,
        )

    def __repr__(self) -> str:
        bid = self.best_bid()
        ask = self.best_ask()
        return (
            f"OrderBook({self.symbol} "
            f"bid={bid.price if bid else 'empty'} "
            f"ask={ask.price if ask else 'empty'} "
            f"uid={self.last_update_id})"
        )
