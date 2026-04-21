"""
Script 2 — Build local order book from snapshot + depth updates.

Finds the latest REST snapshot, replays all depth update JSONL files,
and prints a summary of snapshots produced.

Run: python scripts/2_build_book.py
"""

from __future__ import annotations

from pathlib import Path

from src.book.book_builder import BookBuilder, BookConfig, find_latest_snapshot
from src.utils.config import load_config
from src.utils.logging import get_logger

log = get_logger("2_build_book")

RAW_DIR = Path("data/raw")
SNAPSHOT_DIR = RAW_DIR / "depth_snapshots"
DEPTH_UPDATES_DIR = RAW_DIR / "depth_updates"
SYMBOL = "BTCUSDT"


def main() -> None:
    config = load_config("configs/book.yaml", BookConfig)

    snapshot_path = find_latest_snapshot(SNAPSHOT_DIR, SYMBOL)
    if snapshot_path is None:
        log.error(
            "No snapshot found under %s — run rest_snapshot.py first.", SNAPSHOT_DIR
        )
        return

    log.info("Using snapshot: %s", snapshot_path)
    builder = BookBuilder(config=config, symbol=SYMBOL)

    count = 0
    gaps = 0
    for snap in builder.run(snapshot_path, DEPTH_UPDATES_DIR):
        count += 1
        if snap.sequence_gap:
            gaps += 1
        if count % 100 == 0:
            log.info(
                "Snapshots emitted: %d  gaps: %d  last uid: %d  mid: %.2f",
                count,
                gaps,
                snap.last_update_id,
                (snap.bids[0][0] + snap.asks[0][0]) / 2 if snap.bids and snap.asks else 0,
            )

    log.info("Done. Total snapshots: %d  gaps detected: %d", count, gaps)


if __name__ == "__main__":
    main()
