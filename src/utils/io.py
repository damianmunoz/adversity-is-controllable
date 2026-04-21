"""
Centralised I/O helpers.

Two responsibilities:
1. Read raw JSONL files lazily — one record at a time, never the whole file.
2. Write derived data as Parquet via PyArrow with a fixed schema.

All modules use these instead of opening files directly, so file layout
decisions are made in one place.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List

import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# JSONL reading
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path) -> Generator[Dict[str, Any], None, None]:
    """Yield one parsed record per line from a .jsonl file.

    Skips blank lines. Raises json.JSONDecodeError on malformed lines
    so the caller knows something is wrong rather than silently dropping data.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_jsonl_dir(directory: Path, glob: str = "**/*.jsonl") -> Generator[Dict[str, Any], None, None]:
    """Walk a Hive-partitioned directory and yield all records in path order.

    Files are sorted so date=.../hour=... partitions are replayed in
    chronological order without loading everything into memory.
    """
    for path in sorted(directory.glob(glob)):
        yield from iter_jsonl(path)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append a single record as a JSON line. Creates the file and parent dirs if absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# Parquet writing
# ---------------------------------------------------------------------------

def write_parquet(records: List[Dict[str, Any]], schema: pa.Schema, path: Path) -> None:
    """Write a list of dicts to a Parquet file using the given Arrow schema.

    - Creates parent directories automatically.
    - Overwrites if the file already exists.
    - Schema enforcement means a missing or wrong-typed column raises immediately.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records, schema=schema)
    pq.write_table(table, path, compression="snappy")


def read_parquet(path: Path) -> List[Dict[str, Any]]:
    """Read a Parquet file back as a list of dicts."""
    table = pq.read_table(path)
    return table.to_pylist()


def iter_parquet_dir(directory: Path, glob: str = "**/*.parquet") -> Iterator[Dict[str, Any]]:
    """Walk a partitioned Parquet directory and yield records in path order."""
    for path in sorted(directory.glob(glob)):
        yield from read_parquet(path)
