"""
YAML config loader.

Loads a YAML file and returns a typed dataclass, validating that all
required fields are present. Each module defines its own config dataclass
and passes it here to get a fully-populated object.

Usage:
    @dataclass
    class BookConfig:
        top_n_levels: int
        snapshot_every_n_events: int

    cfg = load_config("configs/book.yaml", BookConfig)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Type, TypeVar

import yaml

T = TypeVar("T")


def load_config(path: str | Path, cls: Type[T]) -> T:
    """Load a YAML file and populate a dataclass of type cls.

    - Extra YAML keys (not in the dataclass) are silently ignored.
    - Missing keys that have no default in the dataclass raise ValueError.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping at {path}, got {type(raw)}")

    fields = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    kwargs: dict = {}

    for name, field in fields.items():
        if name in raw:
            kwargs[name] = raw[name]
        elif field.default is not dataclasses.MISSING:
            kwargs[name] = field.default
        elif field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            kwargs[name] = field.default_factory()
        else:
            raise ValueError(f"Required config key '{name}' missing in {path}")

    return cls(**kwargs)
