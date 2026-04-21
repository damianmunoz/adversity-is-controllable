"""
Feature pipeline — orchestrates all feature modules into a single FeatureVector.

This is the canonical data contract between the market data half of the
pipeline (steps 1–3) and the inference half (steps 4–7, Kalman + policy).

Input:  stream of BookSnapshot objects (from book_builder.py)
Output: stream of FeatureVector objects, one per snapshot

What it does per tick:
  1. Compute spread features (best bid/ask, mid, microprice, spread_bps)
  2. Compute OFI across 5 levels (requires previous snapshot for delta)
  3. Compute depth imbalance across 10 levels
  4. Update rolling volatility for 1s / 5s / 30s windows

FeatureVector is a flat dataclass (no nested objects) so it can be written
directly to Parquet without any transformation.

Persistence:
  Features are written to data/derived/features/ as hourly Parquet files.
  The Parquet schema is defined once here so every write is consistent.
  PyArrow enforces the schema at write time — a type mismatch raises immediately
  rather than silently corrupting the file.
"""

from __future__ import annotations

import pyarrow as pa
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, List, Optional

from src.book.order_book import BookSnapshot
from src.features.imbalance import compute_imbalance
from src.features.ofi import compute_ofi
from src.features.spread import compute_spread
from src.features.volatility import RollingVolatility
from src.utils.io import write_parquet
from src.utils.logging import get_logger
from src.utils.time_utils import hour_partition

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

@dataclass
class FeatureVector:
    """One row of features per book snapshot tick (~every 1 second).

    All floats. ts_ms and symbol are the join keys back to raw data.
    ofi_l1 through ofi_l5 are the 5-level Order Flow Imbalance values.
    """
    ts_ms: int
    symbol: str
    # Spread
    best_bid_px: float
    best_ask_px: float
    spread_abs: float
    spread_bps: float
    mid_price: float
    microprice: float
    # Order flow
    ofi_l1: float
    ofi_l2: float
    ofi_l3: float
    ofi_l4: float
    ofi_l5: float
    # Book shape
    depth_imbalance: float
    # Volatility
    vol_1s: float
    vol_5s: float
    vol_30s: float
    # Metadata
    sequence_gap: bool


# PyArrow schema — used to enforce types at Parquet write time.
FEATURE_SCHEMA = pa.schema([
    ("ts_ms",           pa.int64()),
    ("symbol",          pa.string()),
    ("best_bid_px",     pa.float64()),
    ("best_ask_px",     pa.float64()),
    ("spread_abs",      pa.float64()),
    ("spread_bps",      pa.float64()),
    ("mid_price",       pa.float64()),
    ("microprice",      pa.float64()),
    ("ofi_l1",          pa.float64()),
    ("ofi_l2",          pa.float64()),
    ("ofi_l3",          pa.float64()),
    ("ofi_l4",          pa.float64()),
    ("ofi_l5",          pa.float64()),
    ("depth_imbalance", pa.float64()),
    ("vol_1s",          pa.float64()),
    ("vol_5s",          pa.float64()),
    ("vol_30s",         pa.float64()),
    ("sequence_gap",    pa.bool_()),
])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    ofi_levels: int = 5
    imbalance_levels: int = 10
    volume_weighted_imbalance: bool = True
    output_dir: str = "data/derived/features"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """Stateful feature extractor.

    Stateful because:
    - OFI requires the *previous* snapshot to compute deltas
    - RollingVolatility maintains a time-indexed buffer across ticks

    Usage:
        pipeline = FeaturePipeline(config)
        for fv in pipeline.run(book_snapshot_iterator):
            # fv is a FeatureVector
    """

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self._prev_snapshot: Optional[BookSnapshot] = None
        self._vol = RollingVolatility()

    def process(self, snap: BookSnapshot) -> Optional[FeatureVector]:
        """Process one BookSnapshot into a FeatureVector.

        Returns None on the first tick (no previous snapshot for OFI delta)
        or when spread computation fails (empty book side).
        """
        spread = compute_spread(snap.bids, snap.asks)
        if spread is None:
            log.warning("Empty book side at ts=%d, skipping.", snap.ts_ms)
            self._prev_snapshot = snap
            return None

        # OFI requires a previous snapshot — skip first tick
        if self._prev_snapshot is None:
            self._prev_snapshot = snap
            self._vol.update(snap.ts_ms, spread.mid_price)
            return None

        ofi = compute_ofi(
            prev_bids=self._prev_snapshot.bids,
            prev_asks=self._prev_snapshot.asks,
            curr_bids=snap.bids,
            curr_asks=snap.asks,
            n_levels=self.config.ofi_levels,
        )

        imbalance = compute_imbalance(
            bids=snap.bids,
            asks=snap.asks,
            n_levels=self.config.imbalance_levels,
            volume_weighted=self.config.volume_weighted_imbalance,
        )

        vol = self._vol.update(snap.ts_ms, spread.mid_price)

        self._prev_snapshot = snap

        return FeatureVector(
            ts_ms=snap.ts_ms,
            symbol=snap.symbol,
            best_bid_px=spread.best_bid_px,
            best_ask_px=spread.best_ask_px,
            spread_abs=spread.spread_abs,
            spread_bps=spread.spread_bps,
            mid_price=spread.mid_price,
            microprice=spread.microprice,
            ofi_l1=ofi[0],
            ofi_l2=ofi[1],
            ofi_l3=ofi[2],
            ofi_l4=ofi[3],
            ofi_l5=ofi[4],
            depth_imbalance=imbalance,
            vol_1s=vol.vol_1s,
            vol_5s=vol.vol_5s,
            vol_30s=vol.vol_30s,
            sequence_gap=snap.sequence_gap,
        )

    def run(self, snapshots: Iterator[BookSnapshot]) -> Iterator[FeatureVector]:
        """Process a stream of BookSnapshots, yielding FeatureVectors."""
        for snap in snapshots:
            fv = self.process(snap)
            if fv is not None:
                yield fv

    def run_and_save(
        self,
        snapshots: Iterator[BookSnapshot],
        output_dir: Optional[Path] = None,
    ) -> int:
        """Process snapshots and write hourly Parquet files.

        Batches FeatureVectors by (date, hour) and flushes each batch to
        a separate Parquet file, matching the Hive partition layout of raw data.

        Returns total number of FeatureVectors written.
        """
        out_dir = Path(output_dir or self.config.output_dir)
        batches: dict[tuple[str, str], List[dict]] = {}
        total = 0

        for fv in self.run(snapshots):
            date_str, hour_str = hour_partition(fv.ts_ms)
            key = (date_str, hour_str)
            batches.setdefault(key, []).append(asdict(fv))
            total += 1

        for (date_str, hour_str), records in batches.items():
            path = out_dir / f"date={date_str}" / f"hour={hour_str}" / "features.parquet"
            write_parquet(records, FEATURE_SCHEMA, path)
            log.info("Wrote %d features → %s", len(records), path)

        return total
