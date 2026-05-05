"""
ReplaySource — yields feature rows from a saved session, one per call.

Reads the existing data/derived/features/*.parquet files via
src.utils.io.iter_parquet_dir, splits into sessions using the same 5-second
gap rule that scripts/6_simulate_and_update.py uses, and exposes a
session-picker plus a row iterator.

The GUI's replay controller treats each call to next_row() as one tick of
``simulated time'' and times them with a Qt timer to control playback speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from src.utils.io import iter_parquet_dir


SESSION_GAP_THRESHOLD_MS = 5_000
DEFAULT_FEATURES_DIR = Path("data/derived/features")


@dataclass
class SessionInfo:
    idx:      int          # 1-indexed
    start:    int          # row index (inclusive)
    end:      int          # row index (exclusive)
    ticks:    int
    duration_h: float
    start_ts_ms: int
    end_ts_ms:   int

    @property
    def label(self) -> str:
        return (
            f"S{self.idx} — {self.ticks:>6,} ticks, {self.duration_h:5.2f} h, "
            f"{datetime.fromtimestamp(self.start_ts_ms/1000, tz=timezone.utc):%Y-%m-%d %H:%M} UTC"
        )


def _detect(rows: List[dict]) -> List[Tuple[int, int]]:
    if not rows:
        return []
    out: List[Tuple[int, int]] = []
    start = 0
    for i in range(1, len(rows)):
        time_gap = rows[i]["ts_ms"] - rows[i - 1]["ts_ms"]
        if rows[i].get("sequence_gap") or time_gap > SESSION_GAP_THRESHOLD_MS:
            out.append((start, i))
            start = i
    out.append((start, len(rows)))
    return out


class ReplaySource:
    """Scans the features directory and exposes sessions for replay."""

    def __init__(self, features_dir: Path = DEFAULT_FEATURES_DIR) -> None:
        self._dir = Path(features_dir)
        self._rows: List[dict] = list(iter_parquet_dir(self._dir))
        self._sessions: List[SessionInfo] = []
        for i, (a, b) in enumerate(_detect(self._rows), start=1):
            ticks = b - a
            dur_h = (self._rows[b - 1]["ts_ms"] - self._rows[a]["ts_ms"]) / 3_600_000.0
            self._sessions.append(SessionInfo(
                idx=i, start=a, end=b, ticks=ticks, duration_h=dur_h,
                start_ts_ms=self._rows[a]["ts_ms"],
                end_ts_ms=self._rows[b - 1]["ts_ms"],
            ))

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @property
    def sessions(self) -> List[SessionInfo]:
        return list(self._sessions)

    def session(self, idx: int) -> SessionInfo:
        for s in self._sessions:
            if s.idx == idx:
                return s
        raise IndexError(f"No session #{idx}; have {len(self._sessions)}.")

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def iter_session(self, idx: int) -> Iterator[dict]:
        s = self.session(idx)
        for i in range(s.start, s.end):
            yield self._rows[i]
