"""
FrozenBucketedPolicy — inference-only wrapper around the 1D bucketed Hedge.

The existing BucketedHedgePolicy lives in src/policy/hedge.py and is fully
generic (selection + multiplicative-weights update). For the GUI we want
INFERENCE ONLY — load weights that were learned offline (e.g. on session
S7 with seed 0) and use them deterministically without further updating.

This module does NOT modify src/policy/hedge.py. It composes the existing
policy class: it instantiates BucketedHedgePolicy, overwrites the per-bucket
weight dicts with values loaded from JSON, and exposes a select() that
returns the same PolicyDecision shape, but with the policy's internal RNG
controlled by the seed passed in. The update() method is a no-op.

Bucket lookup logic is reused exactly from BucketedHedgePolicy._which_bucket
(via composition, not duplication).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.policy.actions import Action
from src.policy.hedge import BucketedHedgePolicy, PolicyConfig, PolicyContext, PolicyDecision


# ---------------------------------------------------------------------------
# JSON schema (produced by scripts/9_export_frozen_weights.py):
#
# {
#   "source": "scripts/9_export_frozen_weights.py",
#   "session_idx": 7,
#   "seed":        0,
#   "ticks":       75884,
#   "pressure_edges": [-0.5, -0.2, 0.0, 0.2, 0.5],
#   "buckets": [
#     {"bucket": 0, "lo": -inf, "hi": -0.5,
#      "visits": 43, "weights": {"WAIT": 0.37, "PASSIVE": 0.33, "AGGR": 0.29}},
#     ...
#   ]
# }
# ---------------------------------------------------------------------------


@dataclass
class FrozenWeights:
    """In-memory representation of a frozen weights JSON."""
    pressure_edges: List[float]
    weights:        List[Dict[Action, float]]   # one dict per bucket, length = N+1
    visits:         List[int]
    source_session: int
    source_seed:    int


def load_frozen_weights(path: Path) -> FrozenWeights:
    raw = json.loads(Path(path).read_text())
    edges = list(raw["pressure_edges"])

    # Action keys in JSON are strings; convert to Action enum.
    weights = []
    visits  = []
    for b in raw["buckets"]:
        w = b["weights"]
        weights.append({
            Action.WAIT:       float(w["WAIT"]),
            Action.PASSIVE:    float(w["PASSIVE"]),
            Action.AGGRESSIVE: float(w["AGGRESSIVE"]),
        })
        visits.append(int(b.get("visits", 0)))

    if len(weights) != len(edges) + 1:
        raise ValueError(
            f"Frozen weights file has {len(weights)} buckets but "
            f"{len(edges)} edges (expected {len(edges)+1} buckets)."
        )

    return FrozenWeights(
        pressure_edges=edges,
        weights=weights,
        visits=visits,
        source_session=int(raw.get("session_idx", -1)),
        source_seed=int(raw.get("seed", -1)),
    )


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class FrozenBucketedPolicy:
    """Inference-only 1D bucketed Hedge with weights loaded from JSON.

    Public interface mirrors BucketedHedgePolicy:
      - select(ctx) -> PolicyDecision
      - update(action, loss) is a no-op (kept for interface compatibility)
      - weights() returns the most recently selected bucket's weights
      - bucket_summary() returns full per-bucket info
      - which_bucket(pressure) returns the bucket index used for selection
    """

    def __init__(self, frozen: FrozenWeights, seed: Optional[int] = None) -> None:
        # Compose with BucketedHedgePolicy so we reuse _which_bucket and
        # the same weight-dict shape, but overwrite the weights to the
        # frozen ones immediately.
        cfg = PolicyConfig(
            learning_rate=0.0,            # never used (we don't update)
            initial_weight=1.0,
            mode="bucketed",
            pressure_edges=list(frozen.pressure_edges),
        )
        self._inner = BucketedHedgePolicy(cfg, seed=seed)

        # Overwrite the per-bucket weight dicts with frozen values.
        # _buckets is a List[Dict[Action, float]] inside the inner policy.
        for i, w in enumerate(frozen.weights):
            self._inner._buckets[i] = dict(w)

        # Keep frozen metadata for the GUI to show in a status bar.
        self._frozen_meta = {
            "session": frozen.source_session,
            "seed":    frozen.source_seed,
            "edges":   list(frozen.pressure_edges),
        }

    # ------------------------------------------------------------------
    # Selection / update interface
    # ------------------------------------------------------------------

    def select(self, ctx: PolicyContext) -> PolicyDecision:
        return self._inner.select(ctx)

    def update(self, action: Action, loss: float) -> None:
        """No-op — weights are frozen. Kept to satisfy the policy interface."""
        return

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def weights(self) -> Dict[Action, float]:
        return self._inner.weights()

    def bucket_summary(self):
        return self._inner.bucket_summary()

    def which_bucket(self, pressure: float) -> int:
        return self._inner._which_bucket(pressure)

    @property
    def pressure_edges(self) -> List[float]:
        return list(self._inner._edges)

    @property
    def meta(self) -> dict:
        return dict(self._frozen_meta)
