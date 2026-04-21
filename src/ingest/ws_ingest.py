"""
WebSocket market data ingestion for Binance Spot.

Purpose:
- Subscribe to BTCUSDT trade and depth streams
- Normalize incoming events
- Write newline-delimited JSON (jsonl) files, partitioned by hour

Why jsonl first?
- Easier to debug than Parquet
- Lets us verify connectivity and schema before optimizing storage
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import websockets

from src.ingest.rest_snapshot import fetch_depth_snapshot, save_snapshot


BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream"
DEFAULT_STREAMS = ["btcusdt@trade", "btcusdt@depth@100ms"]


@dataclass
class WSIngestConfig:
    raw_data_dir: Path
    streams: list[str]
    reconnect_delay_sec: float = 5.0
    snapshot_interval_sec: float = 900.0  # 15 min; gives book_builder fresh snapshots to self-heal after WS gaps
    symbol: str = "BTCUSDT"


def utc_now_ms() -> int:
    return int(time.time() * 1000)


def hour_partition(ts_ms: int) -> tuple[str, str]:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_trade_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event_type": payload["e"],
        "event_time_ms": int(payload["E"]),
        "symbol": payload["s"],
        "trade_id": int(payload["t"]),
        "price": float(payload["p"]),
        "qty": float(payload["q"]),
        "trade_time_ms": int(payload["T"]),
        "is_buyer_maker": bool(payload["m"]),
        "recv_time_ms": utc_now_ms(),
        "source": "binance_ws",
    }


def normalize_depth_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event_type": payload["e"],
        "event_time_ms": int(payload["E"]),
        "symbol": payload["s"],
        "first_update_id": int(payload["U"]),
        "final_update_id": int(payload["u"]),
        "bids": [[float(px), float(qty)] for px, qty in payload["b"]],
        "asks": [[float(px), float(qty)] for px, qty in payload["a"]],
        "recv_time_ms": utc_now_ms(),
        "source": "binance_ws",
    }


def output_path(raw_data_dir: Path, stream_kind: str, ts_ms: int) -> Path:
    date_str, hour_str = hour_partition(ts_ms)
    out_dir = raw_data_dir / stream_kind / f"date={date_str}" / f"hour={hour_str}"
    ensure_dir(out_dir)
    return out_dir / "events.jsonl"


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


def parse_stream_message(message: str) -> tuple[str, Dict[str, Any]]:
    parsed = json.loads(message)
    stream_name = parsed["stream"]
    payload = parsed["data"]

    if payload["e"] == "trade":
        return "trades", normalize_trade_event(payload)
    if payload["e"] == "depthUpdate":
        return "depth_updates", normalize_depth_event(payload)

    raise ValueError(f"Unsupported event type: {payload.get('e')}")


async def consume_forever(config: WSIngestConfig) -> None:
    stream_path = "/".join(config.streams)
    url = f"{BINANCE_WS_BASE}?streams={stream_path}"

    while True:
        try:
            print(f"[ws_ingest] Connecting to: {url}")
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                print("[ws_ingest] Connected.")
                async for message in ws:
                    try:
                        stream_kind, record = parse_stream_message(message)
                        path = output_path(config.raw_data_dir, stream_kind, record["recv_time_ms"])
                        append_jsonl(path, record)
                    except Exception as e:
                        print(f"[ws_ingest] Failed to process message: {e}")
        except Exception as e:
            print(f"[ws_ingest] Connection error: {e}")
            print(f"[ws_ingest] Reconnecting in {config.reconnect_delay_sec} seconds...")
            await asyncio.sleep(config.reconnect_delay_sec)


async def snapshot_periodically(config: WSIngestConfig) -> None:
    """Periodically fetch REST depth snapshots to disk.

    Runs concurrently with the WS consumer. Each snapshot's last_update_id
    falls inside the live event stream, so book_builder can re-seed cleanly
    from any of these snapshots if a WS reconnect causes a sequence gap.
    The REST call is sync, so it's dispatched to an executor to avoid
    blocking the WS event loop.
    """
    loop = asyncio.get_running_loop()
    while True:
        try:
            snapshot = await loop.run_in_executor(
                None, fetch_depth_snapshot, config.symbol, 5000
            )
            path = await loop.run_in_executor(None, save_snapshot, snapshot)
            print(f"[ws_ingest] Periodic snapshot saved: {path}")
        except Exception as e:
            print(f"[ws_ingest] Periodic snapshot failed: {e}")
        await asyncio.sleep(config.snapshot_interval_sec)


async def main() -> None:
    config = WSIngestConfig(
        raw_data_dir=Path("data/raw"),
        streams=DEFAULT_STREAMS,
    )
    ensure_dir(config.raw_data_dir)
    await asyncio.gather(
        consume_forever(config),
        snapshot_periodically(config),
    )


if __name__ == "__main__":
    asyncio.run(main())