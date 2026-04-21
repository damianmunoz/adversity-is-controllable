"""
Shared timestamp helpers.

Centralises utc_now_ms and hour_partition so they are not duplicated
across ws_ingest, rest_snapshot, and future modules.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone


def utc_now_ms() -> int:
    """Current UTC time in milliseconds."""
    return int(time.time() * 1000)


def hour_partition(ts_ms: int) -> tuple[str, str]:
    """Return (date_str, hour_str) for Hive-style directory partitioning."""
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H")


def date_partition(ts_ms: int) -> str:
    """Return date string YYYY-MM-DD for a UTC timestamp in ms."""
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


def align_to_grid(ts_ms: int, grid_ms: int) -> int:
    """Floor ts_ms to the nearest grid boundary.

    Example: align_to_grid(1_000_123, 100) → 1_000_100
    Used to snap feature windows to fixed intervals.
    """
    return (ts_ms // grid_ms) * grid_ms
