"""
ChartPane — three panels that together prove the model is working.

  Top    — Mid price + decision markers (what's happening right now)
           Color matches the packet table:
             green  = AGGRESSIVE fill
             red    = PASSIVE fill
             pink   = PASSIVE no-fill
             yellow = WAIT

  Middle — Money SAVED vs always-AGGRESSIVE baseline (the headline)
           Single bold green line of (cum_aggr_loss − cum_loss). Filled
           under the curve. Horizontal reference at zero. Big readout in
           the upper-left shows the current $ saved.
           If the line goes up we're winning. That's the entire pitch.

  Bottom — Per-bucket policy weights (interpretability)
           One stacked bar per market_pressure bucket. Stack: WAIT (yellow)
           on bottom, PASSIVE (red) in middle, AGGRESSIVE (green) on top.
           The bar of the CURRENT bucket is bright; the others are dimmed.
           Watching the highlight move as the market moves makes the
           policy's logic visible to a non-expert audience.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui  import QFont
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from src.live.records import TickRecord


# ---------------------------------------------------------------------------
# Color palette (matches the packet table)
# ---------------------------------------------------------------------------

C_AGG       = "#2ca02c"
C_PASS_FILL = "#d62728"
C_PASS_NO   = "#f4a8a8"     # pale red for unfilled passive
C_WAIT      = "#f7c52d"     # punchy yellow

C_MID       = "#1f77b4"
C_SAVINGS   = "#2ca02c"     # green — money saved
C_SAVINGS_NEG = "#d62728"   # red — currently losing vs AGGR

# Dimmed versions for inactive buckets
C_AGG_DIM   = (44,  160, 44,  90)
C_PASS_DIM  = (214, 39,  40,  90)
C_WAIT_DIM  = (247, 197, 45,  90)

# Bright versions for the active bucket
C_AGG_HOT   = (44,  160, 44,  235)
C_PASS_HOT  = (214, 39,  40,  235)
C_WAIT_HOT  = (247, 197, 45,  235)


def _marker_for(rec: TickRecord):
    """Return (symbol, brush_color, pen_color, size) for a tick's marker."""
    if rec.action == "AGGRESSIVE":
        return "o", C_AGG,       "#1f5d1f", 9
    if rec.action == "PASSIVE":
        if rec.filled:
            return "o", C_PASS_FILL, "#7a1a1a", 9
        return "o", C_PASS_NO,   C_PASS_FILL, 7
    return "x", C_WAIT, "#7a6310", 9


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class ChartPane(QWidget):

    def __init__(self) -> None:
        super().__init__()

        pg.setConfigOptions(antialias=True, background="w", foreground="k")

        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(2, 2, 2, 2)
        self._lay.setSpacing(2)

        # --- 1. Mid + markers --------------------------------------------
        self._mid_plot = pg.PlotWidget(title="BTCUSDT mid + decisions")
        self._mid_plot.showGrid(x=True, y=True, alpha=0.25)
        self._mid_plot.setLabel("left",   "Mid (USDT)")
        self._mid_plot.setLabel("bottom", "Tick")
        self._mid_curve = self._mid_plot.plot(
            [], [], pen=pg.mkPen(C_MID, width=1.3),
        )
        self._scatter = pg.ScatterPlotItem(pxMode=True)
        self._mid_plot.addItem(self._scatter)
        self._lay.addWidget(self._mid_plot, stretch=3)

        # --- 2. Money saved vs always-AGGRESSIVE -------------------------
        # Filled area plot using fillLevel=0 + brush — same render path as
        # the mid curve (so guaranteed to draw), no FillBetweenItem hackery.
        self._sav_plot = pg.PlotWidget(title="Money saved vs always-AGGRESSIVE")
        self._sav_plot.showGrid(x=True, y=True, alpha=0.30)
        self._sav_plot.setLabel("left",   "$ saved (cumulative)")
        self._sav_plot.setLabel("bottom", "Tick")
        self._sav_plot.addLine(
            y=0, pen=pg.mkPen("#888888", width=1, style=Qt.PenStyle.DashLine),
        )
        self._sav_curve = self._sav_plot.plot(
            [], [],
            pen=pg.mkPen(C_SAVINGS, width=2.5),
            fillLevel=0,
            brush=pg.mkBrush(44, 160, 44, 80),
        )
        self._sav_plot.setXLink(self._mid_plot)
        self._lay.addWidget(self._sav_plot, stretch=2)

        # --- 3. Per-bucket weights stacked bar ---------------------------
        self._bw_plot = pg.PlotWidget(
            title="Per-bucket policy weights — current market_pressure bucket is highlighted"
        )
        self._bw_plot.showGrid(x=False, y=True, alpha=0.25)
        self._bw_plot.setLabel("left",   "Weight")
        self._bw_plot.setLabel("bottom", "Pressure bucket")
        self._bw_plot.setYRange(0, 1)
        self._bw_plot.getPlotItem().hideAxis("bottom")    # we put labels manually
        self._bw_dim_items: List[pg.BarGraphItem] = []
        self._bw_hot_items: List[pg.BarGraphItem] = []
        self._bw_labels:    List[pg.TextItem]      = []
        self._bw_loaded = False
        self._lay.addWidget(self._bw_plot, stretch=2)

        # ---- internal state ----
        self._t:       List[int]   = []
        self._mid:     List[float] = []
        self._sav:     List[float] = []

        # Parallel arrays for the scatter — one entry per tick
        self._spx: List[float] = []
        self._spy: List[float] = []
        self._sym: List[str]   = []
        self._brh: List[str]   = []
        self._pen: List[str]   = []
        self._sze: List[int]   = []

        self._weights_per_bucket: Optional[List[Dict[str, float]]] = None
        self._current_bucket: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_bucket_weights(
        self,
        weights_per_bucket: List[Dict[str, float]],
        pressure_edges:     List[float],
    ) -> None:
        """Install the static frozen per-bucket weights.

        weights_per_bucket: list of N dicts, each with keys WAIT/PASSIVE/AGGRESSIVE.
        pressure_edges:     list of N-1 cutpoints used only for axis labels.
        """
        self._weights_per_bucket = list(weights_per_bucket)
        self._draw_bucket_bars(pressure_edges)
        self._bw_loaded = True

    def append_record(self, rec: TickRecord) -> None:
        self._t.append(rec.tick_idx)
        self._mid.append(rec.mid)
        self._sav.append(rec.cum_aggr_loss - rec.cum_loss)

        sym, brush, pen, size = _marker_for(rec)
        self._spx.append(rec.tick_idx)
        self._spy.append(rec.mid)
        self._sym.append(sym)
        self._brh.append(brush)
        self._pen.append(pen)
        self._sze.append(size)

        # Track current bucket (for highlight)
        if rec.bucket != self._current_bucket:
            self._current_bucket = rec.bucket
            self._highlight_current_bucket()

        # Throttle line/scatter redraws to keep the UI snappy under max-speed replay.
        # First few ticks render every tick so the user sees something immediately.
        if len(self._t) <= 8 or len(self._t) % 5 == 0:
            self._refresh()

    def clear(self) -> None:
        self._t.clear()
        self._mid.clear()
        self._sav.clear()
        self._spx.clear(); self._spy.clear()
        self._sym.clear(); self._brh.clear()
        self._pen.clear(); self._sze.clear()
        self._current_bucket = None
        self._refresh()
        if self._bw_loaded:
            self._highlight_current_bucket()

    # ------------------------------------------------------------------
    # Internal — top + middle panels
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        if not self._t:
            self._mid_curve.setData([], [])
            self._scatter.setData([])
            self._sav_curve.setData([], [])
            self._sav_plot.setTitle("Money saved vs always-AGGRESSIVE")
            return

        # Numpy arrays for explicit dtype contracts — pyqtgraph handles
        # python lists too but converting up-front avoids edge cases.
        t_arr = np.asarray(self._t,   dtype=float)
        m_arr = np.asarray(self._mid, dtype=float)
        s_arr = np.asarray(self._sav, dtype=float)

        self._mid_curve.setData(t_arr, m_arr)
        self._sav_curve.setData(t_arr, s_arr)

        # Scatter — parallel-array form (most reliable across pyqtgraph versions)
        brushes = [pg.mkBrush(c) for c in self._brh]
        pens    = [pg.mkPen(c, width=1.0) for c in self._pen]
        self._scatter.setData(
            x=self._spx, y=self._spy,
            symbol=self._sym, size=self._sze,
            brush=brushes, pen=pens,
        )

        # Live readout — embed the running $ amount in the panel title.
        # Using HTML lets us color the number green when winning, red when losing.
        cur  = float(s_arr[-1])
        sign = "+" if cur >= 0 else "−"
        color = "#1a6c1a" if cur >= 0 else "#7a1a1a"
        self._sav_plot.setTitle(
            f"Money saved vs always-AGGRESSIVE — "
            f"<span style='color:{color}; font-weight:bold;'>"
            f"${sign}{abs(cur):,.2f}</span>"
        )

    # ------------------------------------------------------------------
    # Internal — bottom panel: stacked bars
    # ------------------------------------------------------------------

    def _draw_bucket_bars(self, pressure_edges: List[float]) -> None:
        """Build static dimmed stacked bars + empty hot overlays for highlight."""
        # Clear any previous content
        for it in self._bw_dim_items + self._bw_hot_items + self._bw_labels:
            self._bw_plot.removeItem(it)
        self._bw_dim_items.clear()
        self._bw_hot_items.clear()
        self._bw_labels.clear()

        weights = self._weights_per_bucket or []
        n = len(weights)
        if n == 0:
            return

        x       = np.arange(n, dtype=float)
        wait    = np.array([w["WAIT"]       for w in weights], dtype=float)
        passive = np.array([w["PASSIVE"]    for w in weights], dtype=float)
        aggr    = np.array([w["AGGRESSIVE"] for w in weights], dtype=float)

        # Stacked dimmed bars (everything not the current bucket).
        # We add three full-width dim bars; we'll mask the current bucket
        # by drawing hot bars on top.
        bar_w = 0.7
        wait_dim = pg.BarGraphItem(x=x, height=wait,    width=bar_w, y0=0,
                                   brush=pg.mkBrush(*C_WAIT_DIM), pen=pg.mkPen(0,0,0,30))
        pass_dim = pg.BarGraphItem(x=x, height=passive, width=bar_w, y0=wait,
                                   brush=pg.mkBrush(*C_PASS_DIM), pen=pg.mkPen(0,0,0,30))
        aggr_dim = pg.BarGraphItem(x=x, height=aggr,    width=bar_w, y0=wait + passive,
                                   brush=pg.mkBrush(*C_AGG_DIM),  pen=pg.mkPen(0,0,0,30))
        for it in (wait_dim, pass_dim, aggr_dim):
            self._bw_plot.addItem(it)
            self._bw_dim_items.append(it)

        # Hot overlay placeholders (will be populated when a bucket becomes current).
        wait_hot = pg.BarGraphItem(x=[], height=[], width=bar_w, y0=[],
                                   brush=pg.mkBrush(*C_WAIT_HOT), pen=pg.mkPen("#000", width=1.5))
        pass_hot = pg.BarGraphItem(x=[], height=[], width=bar_w, y0=[],
                                   brush=pg.mkBrush(*C_PASS_HOT), pen=pg.mkPen("#000", width=1.5))
        aggr_hot = pg.BarGraphItem(x=[], height=[], width=bar_w, y0=[],
                                   brush=pg.mkBrush(*C_AGG_HOT),  pen=pg.mkPen("#000", width=1.5))
        for it in (wait_hot, pass_hot, aggr_hot):
            self._bw_plot.addItem(it)
            self._bw_hot_items.append(it)

        # Bucket axis labels — show the pressure range of each bucket.
        for i in range(n):
            lo = "-∞" if i == 0           else f"{pressure_edges[i-1]:+.2f}"
            hi = "+∞" if i == n - 1       else f"{pressure_edges[i]:+.2f}"
            t = pg.TextItem(f"{i}\n({lo}, {hi}]", anchor=(0.5, 0), color="#444444")
            tf = QFont("Menlo, Consolas, monospace"); tf.setPointSize(8)
            t.setFont(tf)
            t.setPos(i, -0.02)
            self._bw_plot.addItem(t)
            self._bw_labels.append(t)

        self._bw_plot.setXRange(-0.5, n - 0.5, padding=0)

    def _highlight_current_bucket(self) -> None:
        """Move the hot-color overlay to the current bucket (or hide it)."""
        if not self._bw_loaded or not self._weights_per_bucket:
            return
        wait_hot, pass_hot, aggr_hot = self._bw_hot_items
        if self._current_bucket is None or self._current_bucket >= len(self._weights_per_bucket):
            empty = []
            wait_hot.setOpts(x=empty, height=empty, y0=empty)
            pass_hot.setOpts(x=empty, height=empty, y0=empty)
            aggr_hot.setOpts(x=empty, height=empty, y0=empty)
            return
        b = self._current_bucket
        w = self._weights_per_bucket[b]
        x = [b]
        wait_hot.setOpts(x=x, height=[w["WAIT"]],       y0=[0])
        pass_hot.setOpts(x=x, height=[w["PASSIVE"]],    y0=[w["WAIT"]])
        aggr_hot.setOpts(x=x, height=[w["AGGRESSIVE"]], y0=[w["WAIT"] + w["PASSIVE"]])
