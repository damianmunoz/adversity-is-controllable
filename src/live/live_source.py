"""
LiveSource — in-process Binance WebSocket consumer that yields feature dicts.

This is a streaming sibling of scripts/1_ingest_ws.py + scripts/3_featuregen.py
that does NOT write to disk. It exists so the GUI can show real-time decisions
without requiring the user to also run the offline ingest+featuregen pipeline.

Synchronization protocol (CRITICAL — sanity.txt §1):
  Binance's depth stream is differential. To build a correct local book you
  MUST follow the documented sync protocol:

    1. Open the WS connection FIRST and start buffering events.
    2. Concurrently fetch the REST snapshot (it may take a couple of seconds).
    3. When the snapshot arrives, locate the FIRST buffered event whose
       [first_update_id, final_update_id] range straddles
       snap.lastUpdateId + 1 — that is the bridge.
    4. Apply the snapshot, then apply the bridge event and onwards.
    5. While processing live events, each new event must continue the chain:
       its first_update_id must equal the previous event's final_update_id+1.
       Any gap means the local book is suspect and we must reseed.

  Two failure modes a naive implementation hits and how this module handles them:

    * "Snapshot is ahead of buffer." Snapshot's lastUpdateId is greater than
      every buffered event's final_update_id. This is normal — REST took
      longer to return than the WS events we've buffered cover. Solution:
      KEEP READING WS until the bridge arrives. Do NOT throw away the buffer.

    * "Snapshot is behind buffer." Snapshot's lastUpdateId is less than the
      first buffered event's first_update_id - 1. The snapshot was so stale
      that we already skipped past it. Solution: refetch snapshot only.

  The previous version of this module fetched the snapshot BEFORE opening
  the WS, which is the inverse of the protocol and leads to a corrupted
  book — same root cause as sanity.txt §1's catastrophic bug.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import asdict
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Tuple

import websockets

from src.book.order_book import OrderBook
from src.features.feature_pipeline import FeatureConfig, FeaturePipeline
from src.ingest.rest_snapshot import fetch_depth_snapshot
from src.utils.logging import get_logger

log = get_logger(__name__)


BINANCE_WS_URL = "wss://stream.binance.com:9443/stream?streams=btcusdt@depth@100ms"
EMIT_EVERY_N_EVENTS = 10           # ~1 Hz at the @100ms cadence
SEED_TIMEOUT_SEC    = 20.0
WS_RECV_TIMEOUT_SEC = 0.3
RECONNECT_DELAY_SEC = 1.5


def _normalize_depth(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Same shape as src/ingest/ws_ingest.py:normalize_depth_event."""
    return {
        "event_type":      payload["e"],
        "event_time_ms":   int(payload["E"]),
        "symbol":          payload["s"],
        "first_update_id": int(payload["U"]),
        "final_update_id": int(payload["u"]),
        "bids": [[float(px), float(qty)] for px, qty in payload["b"]],
        "asks": [[float(px), float(qty)] for px, qty in payload["a"]],
    }


# ---------------------------------------------------------------------------
# LiveSource
# ---------------------------------------------------------------------------

class LiveSource:

    def __init__(self, symbol: str = "BTCUSDT") -> None:
        self.symbol = symbol
        self._out: "Queue[Dict[str, Any]]" = Queue(maxsize=10_000)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Re-created on every successful seeding so a reconnect can never
        # carry over a corrupt book or stale OFI/vol state.
        self._book = OrderBook(symbol=symbol, top_n=20)
        self._pipeline = FeaturePipeline(FeatureConfig())
        self._event_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="live-source", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def next_row(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        try:
            return self._out.get(timeout=timeout) if timeout > 0 else self._out.get_nowait()
        except Empty:
            return None

    def qsize(self) -> int:
        return self._out.qsize()

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            asyncio.run(self._run_async())
        except Exception as exc:                                   # noqa: BLE001
            log.exception("LiveSource worker crashed: %s", exc)

    async def _run_async(self) -> None:
        """Top-level reconnect loop: each iteration reseeds and processes."""
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    BINANCE_WS_URL, max_size=2**22, ping_interval=20, ping_timeout=20,
                ) as ws:
                    seeded = await self._seed(ws)
                    if not seeded:
                        log.warning(
                            "LiveSource: seeding failed — retry in %.1fs",
                            RECONNECT_DELAY_SEC,
                        )
                        await asyncio.sleep(RECONNECT_DELAY_SEC)
                        continue
                    await self._process_live(ws)
            except (websockets.ConnectionClosed, OSError, asyncio.TimeoutError) as exc:
                log.warning("LiveSource WS dropped: %s — reconnecting.", exc)
                await asyncio.sleep(RECONNECT_DELAY_SEC)
                continue

    # ------------------------------------------------------------------
    # Seeding (Binance protocol — see module docstring)
    # ------------------------------------------------------------------

    async def _seed(self, ws) -> bool:
        """Buffer WS events while a snapshot fetch runs concurrently, find the
        bridge between them, then apply snapshot + bridge + tail."""
        loop = asyncio.get_event_loop()
        deadline = loop.time() + SEED_TIMEOUT_SEC

        # Kick off snapshot fetch in the background — runs while we buffer WS.
        snapshot_task = asyncio.create_task(
            asyncio.to_thread(fetch_depth_snapshot, self.symbol, 5000)
        )

        buffer: List[Dict[str, Any]] = []
        snapshot: Optional[Dict[str, Any]] = None
        snap_uid: Optional[int] = None

        ws_events_read = 0

        while not self._stop.is_set() and loop.time() < deadline:
            # 1. Read one WS event with a short timeout so we never block
            #    indefinitely while waiting for the snapshot to land.
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=WS_RECV_TIMEOUT_SEC)
                ev = self._parse(raw)
                if ev is not None:
                    buffer.append(ev)
                    ws_events_read += 1
            except asyncio.TimeoutError:
                pass

            # 2. Pick up the snapshot if it has arrived.
            if snapshot is None and snapshot_task.done():
                try:
                    snapshot = snapshot_task.result()
                    snap_uid = int(snapshot["last_update_id"])
                    log.info(
                        "LiveSource: snapshot arrived uid=%d (buffered %d WS events)",
                        snap_uid, len(buffer),
                    )
                except Exception as exc:                          # noqa: BLE001
                    log.warning("LiveSource: snapshot fetch failed: %s", exc)
                    snapshot_task = asyncio.create_task(
                        asyncio.to_thread(fetch_depth_snapshot, self.symbol, 5000)
                    )
                    continue

            # 3. Once we have a snapshot, try to bridge.
            if snapshot is not None:
                idx, status = _classify_buffer(buffer, snap_uid)

                if status == "found":
                    return self._apply_bridge(buffer, idx, snapshot)

                if status in ("behind", "gap"):
                    log.info(
                        "LiveSource: snap_uid=%d %s buffer (first_U=%d, last_u=%d) — "
                        "refetching snapshot.",
                        snap_uid, status,
                        buffer[0]["first_update_id"], buffer[-1]["final_update_id"],
                    )
                    snapshot = None
                    snap_uid = None
                    snapshot_task = asyncio.create_task(
                        asyncio.to_thread(fetch_depth_snapshot, self.symbol, 5000)
                    )
                    # Drop everything except the most recent events to keep memory bounded.
                    buffer = buffer[-200:]
                    continue
                # status == "ahead": keep reading WS — bridge hasn't arrived yet.

        log.warning(
            "LiveSource: seeding timed out after %.0fs (read %d WS events, "
            "snap_uid=%s, last buffered u=%s)",
            SEED_TIMEOUT_SEC, ws_events_read,
            snap_uid if snap_uid is not None else "n/a",
            buffer[-1]["final_update_id"] if buffer else "n/a",
        )
        return False

    def _apply_bridge(
        self,
        buffer: List[Dict[str, Any]],
        first_idx: int,
        snapshot: Dict[str, Any],
    ) -> bool:
        # Reset book + pipeline + counter from scratch — never carry corrupt state.
        self._book = OrderBook(symbol=self.symbol, top_n=20)
        self._pipeline = FeaturePipeline(FeatureConfig())
        self._event_count = 0
        self._book.apply_snapshot(snapshot)

        snap_uid = int(snapshot["last_update_id"])
        applied = 0
        for ev in buffer[first_idx:]:
            if ev["final_update_id"] <= self._book.last_update_id:
                continue
            self._book.apply_depth_update(ev)
            self._maybe_emit(ev)
            applied += 1

        bridge = buffer[first_idx]
        log.info(
            "LiveSource SEEDED — snap_uid=%d  bridge=[U=%d, u=%d]  "
            "applied=%d  pre-snap dropped=%d",
            snap_uid, bridge["first_update_id"], bridge["final_update_id"],
            applied, first_idx,
        )
        return True

    # ------------------------------------------------------------------
    # Live processing (after a successful seed)
    # ------------------------------------------------------------------

    async def _process_live(self, ws) -> None:
        async for raw in ws:
            if self._stop.is_set():
                return
            ev = self._parse(raw)
            if ev is None:
                continue

            if ev["final_update_id"] <= self._book.last_update_id:
                continue

            expected = self._book.last_update_id + 1
            if ev["first_update_id"] > expected:
                log.warning(
                    "LiveSource: sequence gap (expected=%d, got U=%d). Reseeding.",
                    expected, ev["first_update_id"],
                )
                # Force the outer loop to reconnect and reseed by closing the WS.
                await ws.close()
                return

            self._book.apply_depth_update(ev)
            self._maybe_emit(ev)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse(self, raw: str) -> Optional[Dict[str, Any]]:
        try:
            payload = json.loads(raw)
        except Exception:                                         # noqa: BLE001
            return None
        data = payload.get("data") if isinstance(payload, dict) else None
        if not data or data.get("e") != "depthUpdate":
            return None
        return _normalize_depth(data)

    def _maybe_emit(self, ev: Dict[str, Any]) -> None:
        """Emit a feature row roughly every 1 second (every Nth applied event)."""
        self._event_count += 1
        if self._event_count % EMIT_EVERY_N_EVENTS != 0:
            return

        snap = self._book.to_snapshot(ts_ms=ev["event_time_ms"], sequence_gap=False)
        fv = self._pipeline.process(snap)
        if fv is None:
            return

        # Defensive guard. Should not fire after the protocol fix, but
        # cheap insurance against any future regression.
        if not (fv.spread_abs > 0
                and fv.best_ask_px > fv.best_bid_px
                and fv.best_bid_px <= fv.mid_price <= fv.best_ask_px):
            log.warning(
                "LiveSource: dropping corrupt feature  bid=%.2f ask=%.2f mid=%.2f spread=%.4f",
                fv.best_bid_px, fv.best_ask_px, fv.mid_price, fv.spread_abs,
            )
            return

        if self._out.full():
            try:
                self._out.get_nowait()
            except Empty:
                pass
        self._out.put_nowait(asdict(fv))


# ---------------------------------------------------------------------------
# Pure helper — buffer / snapshot classification
# ---------------------------------------------------------------------------

def _classify_buffer(
    buffer: List[Dict[str, Any]],
    snap_uid: int,
) -> Tuple[Optional[int], str]:
    """Where does snap_uid sit relative to the buffered WS events?

    Returns (first_post_snap_idx_or_None, status):
      'found'  — bridge is at the returned index
      'ahead'  — all buffered events end at or before snap_uid, keep reading WS
      'behind' — first buffered event starts past snap_uid+1, snapshot is stale
      'gap'    — there are post-snapshot events but none bridges (protocol gap)
    """
    if not buffer:
        return None, "ahead"

    # All events ended at or before the snapshot's uid → WS hasn't reached it yet.
    if buffer[-1]["final_update_id"] <= snap_uid:
        return None, "ahead"

    # First event starts strictly past snap_uid+1 → we already skipped the bridge,
    # snapshot is too old.
    if buffer[0]["first_update_id"] > snap_uid + 1:
        return None, "behind"

    # Walk forward looking for the first post-snapshot event.
    for i, ev in enumerate(buffer):
        if ev["final_update_id"] <= snap_uid:
            continue
        # First event past the snapshot. It must also start at or before snap_uid+1
        # for the chain to be valid.
        if ev["first_update_id"] <= snap_uid + 1:
            return i, "found"
        return i, "gap"

    # Should be unreachable given the early returns above, but typed for safety.
    return None, "ahead"
