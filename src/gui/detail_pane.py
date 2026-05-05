"""
DetailPane — Wireshark-style collapsible tree of all fields of a TickRecord.

Click a row in the packet table → this panel shows everything about that
tick: market state, features, Kalman posterior, policy decision (with the
pre-decision per-action probabilities), execution outcome, and running
totals.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui  import QFont
from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem

from src.live.records import TickRecord


_INF = 1e18


def _fmt_inf(x: float) -> str:
    if x >=  +_INF / 2: return "+inf"
    if x <=  -_INF / 2: return "-inf"
    return f"{x:+.4f}"


def _utc(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def _record_to_groups(rec: TickRecord) -> List[Tuple[str, List[Tuple[str, str]]]]:
    fill_str = "filled" if rec.filled else "no-fill"
    return [
        ("Tick", [
            ("Index",        f"{rec.tick_idx}"),
            ("Decision time",_utc(rec.ts_ms)),
            ("Outcome time", _utc(rec.next_ts_ms)),
            ("ts_ms",        f"{rec.ts_ms}"),
        ]),
        ("Order book", [
            ("Best bid",     f"{rec.best_bid:,.4f}"),
            ("Best ask",     f"{rec.best_ask:,.4f}"),
            ("Mid",          f"{rec.mid:,.4f}"),
            ("Microprice",   f"{rec.microprice:,.4f}"),
            ("Spread",       f"{rec.spread_abs:.4f}"),
            ("Next mid",     f"{rec.next_mid:,.4f}"),
            ("Δ mid (next−curr)", f"{rec.next_mid - rec.mid:+.4f}"),
        ]),
        ("Features", [
            ("depth_imbalance", f"{rec.depth_imbalance:+.6f}"),
            ("ofi_l1",          f"{rec.ofi_l1:+.6f}"),
            ("vol_30s",         f"{rec.vol_30s:.6e}"),
        ]),
        ("Kalman state (posterior)", [
            ("market_pressure", f"{rec.pressure:+.6f}"),
            ("regime",          f"{rec.regime:+.6f}"),
            ("Var(pressure)",   f"{rec.p00:.6f}"),
            ("Var(regime)",     f"{rec.p11:.6f}"),
            ("Cov(pressure, regime)", f"{rec.p01:+.6f}"),
        ]),
        ("Policy decision", [
            ("Bucket index",       f"{rec.bucket}"),
            ("Bucket range",       f"({_fmt_inf(rec.bucket_lo)}, {_fmt_inf(rec.bucket_hi)}]"),
            ("Pre-decision weights", ""),
            ("    P(WAIT)",        f"{rec.weight_wait:.4f}"),
            ("    P(PASSIVE)",     f"{rec.weight_passive:.4f}"),
            ("    P(AGGRESSIVE)",  f"{rec.weight_aggressive:.4f}"),
            ("Action chosen",      rec.action),
        ]),
        ("Execution", [
            ("Filled",      fill_str),
            ("Fill price",  f"{rec.fill_price:,.4f}" if rec.filled else "—"),
            ("Slippage",    f"{rec.slippage:+.4f}"),
            ("Adverse move",f"{rec.adverse_move:.4f}"),
            ("Loss (slip + λ·adv)", f"{rec.loss:+.4f}"),
        ]),
        ("Counterfactual: always-AGGRESSIVE this tick", [
            ("AGGR slippage", f"{rec.aggr_slippage:+.4f}"),
            ("AGGR adverse",  f"{rec.aggr_adverse:.4f}"),
            ("AGGR loss",     f"{(rec.aggr_slippage + 0.10*rec.aggr_adverse):+.4f}"),
        ]),
        ("Cumulative (since session start)", [
            ("Ticks processed",     f"{rec.cum_ticks}"),
            ("Fills",               f"{rec.cum_filled}"),
            ("Fill rate",           f"{(rec.cum_filled / max(1, rec.cum_ticks)):.2%}"),
            ("Total slippage",      f"{rec.cum_slippage:+.4f}"),
            ("Total adverse",       f"{rec.cum_adverse:.4f}"),
            ("Total loss",          f"{rec.cum_loss:+.4f}"),
            ("Total AGGR-baseline loss", f"{rec.cum_aggr_loss:+.4f}"),
            ("Saved vs always-AGGR", f"{rec.cum_aggr_loss - rec.cum_loss:+.4f}"),
        ]),
    ]


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class DetailPane(QTreeWidget):

    def __init__(self) -> None:
        super().__init__()
        self.setColumnCount(2)
        self.setHeaderLabels(["Field", "Value"])
        self.setRootIsDecorated(True)
        self.setUniformRowHeights(True)
        self.setAlternatingRowColors(True)

        f = QFont("Menlo, Consolas, monospace")
        f.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(f)

        self.setColumnWidth(0, 260)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def show_record(self, rec: TickRecord) -> None:
        self.clear()
        for group_name, items in _record_to_groups(rec):
            top = QTreeWidgetItem([group_name, ""])
            top_font = top.font(0); top_font.setBold(True); top.setFont(0, top_font)
            self.addTopLevelItem(top)
            for label, value in items:
                child = QTreeWidgetItem([label, value])
                if value == "":
                    cf = child.font(0); cf.setItalic(True); child.setFont(0, cf)
                top.addChild(child)
            top.setExpanded(True)
