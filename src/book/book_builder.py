"""
Order book reconstruction engine.

Takes a REST snapshot (to initialize) and a stream of incremental depth
update events (to advance state), and emits BookSnapshot objects at a
configurable cadence.

Why snapshot-first?
  Binance's depth stream is a differential feed — it only tells you what
  changed, not the full state. You need a snapshot to know the starting
  state, then apply deltas in order.

Sequence integrity and self-healing:
  Binance assigns monotonically increasing update IDs to every depth event.
  If an event arrives with first_update_id > last_update_id + 1, there is
  a gap — events were dropped (usually a WS reconnect). Once a gap occurs
  the local book is stale and cannot be trusted until re-synced against a
  REST snapshot whose last_update_id covers or post-dates the gap.

  The builder scans the full snapshot directory up front. On any gap it
  searches for the next snapshot whose last_update_id bridges or exceeds
  the gap and re-seeds from it. If no such snapshot exists, the book is
  marked corrupted and every subsequent emission carries sequence_gap=True
  until a fresh snapshot eventually becomes available. This prevents the
  silent-corruption failure mode where only the first post-gap emission
  was flagged and everything downstream looked clean but had a stale book.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

from src.book.order_book import BookSnapshot, OrderBook
from src.utils.io import iter_jsonl, iter_jsonl_dir
from src.utils.logging import get_logger
from src.utils.time_utils import utc_now_ms

log = get_logger(__name__)


@dataclass
class BookConfig:
    top_n_levels: int = 20
    snapshot_every_n_events: int = 10   # ~1 second at 100ms depth stream


def load_snapshot_file(path: Path) -> dict:
    """Load a single REST snapshot JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def list_snapshots(snapshot_dir: Path, symbol: str) -> List[Path]:
    """Return all snapshots for a symbol, sorted chronologically by filename.

    Filename format from rest_snapshot.py is `snapshot_{utc_ms}.json`, so
    lexicographic sort equals chronological sort.
    """
    pattern = f"symbol={symbol.upper()}/date=*/*.json"
    return sorted(snapshot_dir.glob(pattern))


def find_latest_snapshot(snapshot_dir: Path, symbol: str) -> Optional[Path]:
    """Return the most-recently-written snapshot file for a given symbol."""
    candidates = list_snapshots(snapshot_dir, symbol)
    return candidates[-1] if candidates else None


class BookBuilder:
    """Replays snapshots + incremental depth events into a live OrderBook.

    Usage:
        builder = BookBuilder(config, symbol="BTCUSDT")
        for snap in builder.run(snapshot_dir, depth_updates_dir):
            # snap is a BookSnapshot — pass to FeaturePipeline
    """

    def __init__(self, config: BookConfig, symbol: str = "BTCUSDT") -> None:
        self.config = config
        self.symbol = symbol
        self.book = OrderBook(symbol=symbol, top_n=config.top_n_levels)
        self.corrupted = False

    def _seed_from(self, snapshot_path: Path) -> None:
        raw = load_snapshot_file(snapshot_path)
        self.book.apply_snapshot(raw)
        log.info(
            "Seeded from %s  uid=%d  bids=%d  asks=%d",
            snapshot_path.name,
            self.book.last_update_id,
            len(self.book._bids),
            len(self.book._asks),
        )

    def _try_reseed_after_gap(
        self,
        remaining_snapshots: List[Path],
        first_uid: int,
    ) -> int:
        """Find the earliest remaining snapshot whose last_update_id >= first_uid - 1
        (i.e., that bridges or post-dates the gap) and re-seed from it.

        Returns the number of snapshots consumed (0 = none usable).
        """
        for i, path in enumerate(remaining_snapshots):
            raw = load_snapshot_file(path)
            new_uid = int(raw["last_update_id"])
            if new_uid + 1 >= first_uid:
                self.book.apply_snapshot(raw)
                self.corrupted = False
                log.info(
                    "Gap recovered: reseeded from %s  uid=%d",
                    path.name,
                    new_uid,
                )
                return i + 1
        return 0

    def run(
        self,
        snapshot_dir: Path,
        depth_updates_dir: Path,
    ) -> Iterator[BookSnapshot]:
        """Seed from earliest snapshot, replay depth events, yield BookSnapshots.

        Args:
            snapshot_dir: Root of snapshot hive (`symbol=.../date=.../*.json`).
            depth_updates_dir: Root of depth_updates hive.

        Yields:
            BookSnapshot every `snapshot_every_n_events` depth updates. On
            any gap, immediately emits a snapshot with sequence_gap=True,
            then attempts to reseed from the next available REST snapshot.
            If no reseed is possible, every subsequent emission carries
            sequence_gap=True until a fresh snapshot appears.
        """
        snapshots = list_snapshots(snapshot_dir, self.symbol)
        if not snapshots:
            raise FileNotFoundError(
                f"No snapshots under {snapshot_dir} for {self.symbol}"
            )

        # Seed from the earliest snapshot; the rest are held in reserve for gap recovery.
        self._seed_from(snapshots[0])
        snap_idx = 0

        event_count = 0
        for event in iter_jsonl_dir(depth_updates_dir):
            recv_ts = event.get("recv_time_ms", utc_now_ms())
            first_uid = int(event["first_update_id"])
            final_uid = int(event["final_update_id"])

            # Skip events already covered by the current snapshot
            if final_uid <= self.book.last_update_id:
                continue

            gap = first_uid > self.book.last_update_id + 1
            if gap:
                log.warning(
                    "Sequence gap at uid=%d  first=%d  expected=%d",
                    self.book.last_update_id,
                    first_uid,
                    self.book.last_update_id + 1,
                )
                consumed = self._try_reseed_after_gap(
                    snapshots[snap_idx + 1:], first_uid
                )
                if consumed > 0:
                    snap_idx += consumed
                    # Skip this event if the new snapshot already covers it
                    if final_uid <= self.book.last_update_id:
                        continue
                    # Else fall through and apply normally
                else:
                    # No fresh snapshot available — book is now corrupted
                    self.corrupted = True
                    self.book.apply_depth_update(event)
                    event_count += 1
                    # Emit immediately so downstream sees the gap marker
                    yield self.book.to_snapshot(ts_ms=recv_ts, sequence_gap=True)
                    continue

            self.book.apply_depth_update(event)
            event_count += 1

            if event_count % self.config.snapshot_every_n_events == 0:
                yield self.book.to_snapshot(
                    ts_ms=recv_ts, sequence_gap=self.corrupted
                )
