"""
Script 3 — Generate features from reconstructed order book.

Replays the book (snapshot + depth updates) through the feature pipeline
and writes hourly Parquet files to data/derived/features/.

This script combines steps 2 and 3: it builds the book internally and
immediately pipes each BookSnapshot into the feature pipeline without
writing book state to disk first. This keeps memory usage flat.

Run: python scripts/3_featuregen.py
"""

from __future__ import annotations

from pathlib import Path

from src.book.book_builder import BookBuilder, BookConfig, list_snapshots
from src.features.feature_pipeline import FeatureConfig, FeaturePipeline
from src.utils.config import load_config
from src.utils.logging import get_logger

log = get_logger("3_featuregen")

RAW_DIR = Path("data/raw")
SNAPSHOT_DIR = RAW_DIR / "depth_snapshots"
DEPTH_UPDATES_DIR = RAW_DIR / "depth_updates"
SYMBOL = "BTCUSDT"


def main() -> None:
    book_config = load_config("configs/book.yaml", BookConfig)
    feature_config = load_config("configs/features.yaml", FeatureConfig)

    snapshots = list_snapshots(SNAPSHOT_DIR, SYMBOL)
    if not snapshots:
        log.error("No snapshot found — run rest_snapshot.py first.")
        return

    log.info("Snapshots available: %d (earliest=%s)", len(snapshots), snapshots[0].name)

    builder = BookBuilder(config=book_config, symbol=SYMBOL)
    pipeline = FeaturePipeline(config=feature_config)

    book_snapshots = builder.run(SNAPSHOT_DIR, DEPTH_UPDATES_DIR)
    total = pipeline.run_and_save(book_snapshots)

    log.info("Feature generation complete. Total vectors written: %d", total)


if __name__ == "__main__":
    main()
