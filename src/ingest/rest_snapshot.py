"""
REST snapshot downloader for Binance Spot order book.

Purpose:
- Pull a depth snapshot for BTCUSDT
- Save raw snapshot to disk
- Used later to initialize / reconcile a local order book
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import requests


BINANCE_REST_BASE = "https://api.binance.com"
DEPTH_ENDPOINT = "/api/v3/depth"


def utc_now_ms() -> int:
    return int(time.time() * 1000)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def date_partition(ts_ms: int) -> str:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


def fetch_depth_snapshot(symbol: str = "BTCUSDT", limit: int = 5000) -> Dict[str, Any]:
    url = f"{BINANCE_REST_BASE}{DEPTH_ENDPOINT}"
    params = {"symbol": symbol.upper(), "limit": limit}

    start_ms = utc_now_ms()
    response = requests.get(url, params=params, timeout=10)
    latency_ms = utc_now_ms() - start_ms
    response.raise_for_status()
    payload = response.json()

    return {
        "snapshot_time_ms": utc_now_ms(),
        "symbol": symbol.upper(),
        "last_update_id": int(payload["lastUpdateId"]),
        "bids": [[float(px), float(qty)] for px, qty in payload["bids"]],
        "asks": [[float(px), float(qty)] for px, qty in payload["asks"]],
        "http_latency_ms": latency_ms,
        "source": "binance_rest",
    }


def save_snapshot(snapshot: Dict[str, Any], base_dir: Path = Path("data/raw/depth_snapshots")) -> Path:
    date_str = date_partition(snapshot["snapshot_time_ms"])
    out_dir = base_dir / f"symbol={snapshot['symbol']}" / f"date={date_str}"
    ensure_dir(out_dir)

    filename = f"snapshot_{snapshot['snapshot_time_ms']}.json"
    out_path = out_dir / filename
    out_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    snapshot = fetch_depth_snapshot("BTCUSDT", 5000)
    out_path = save_snapshot(snapshot)
    print(f"[rest_snapshot] Saved snapshot to {out_path}")


if __name__ == "__main__":
    main()