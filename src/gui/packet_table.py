"""
PacketTable — Wireshark-like color-coded list of TickRecords.

One row per processed tick. Background color encodes the action:

  AGGRESSIVE → green
  PASSIVE    → red
  WAIT       → yellow

Rows are append-only. Selecting a row emits row_selected(TickRecord) so the
detail pane and chart pane can react.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from PyQt6.QtCore import (
    QAbstractTableModel, QModelIndex, QSize, Qt, pyqtSignal, QObject,
)
from PyQt6.QtGui  import QBrush, QColor, QFont
from PyQt6.QtWidgets import QHeaderView, QTableView, QAbstractItemView

from src.live.records import TickRecord


# ---------------------------------------------------------------------------
# Color palette (close to Wireshark conventions)
# ---------------------------------------------------------------------------

COLOR_AGG       = QColor("#a8e6a8")   # soft green
COLOR_PASS_FILL = QColor("#f4a8a8")   # soft red (pass + fill)
COLOR_PASS_NOFILL = QColor("#f7d4d4") # paler red (pass + no fill)
COLOR_WAIT      = QColor("#fff4a8")   # soft yellow
COLOR_GAP       = QColor("#d8d8d8")   # gray for sequence gaps (rare)

FONT_MONO = QFont("Menlo, Consolas, monospace")
FONT_MONO.setStyleHint(QFont.StyleHint.Monospace)


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------

COLUMNS = [
    ("#",          "tick_idx",      8),
    ("Time",       None,           20),    # formatted from ts_ms
    ("Mid",        "mid",          11),
    ("Spread",     "spread_abs",    8),
    ("Pressure",   "pressure",      9),
    ("Bucket",     "bucket",        7),
    ("Action",     "action",       11),
    ("Filled",     "filled",        7),
    ("Slip",       "slippage",      9),
    ("Adv",        "adverse_move",  9),
    ("Loss",       "loss",          9),
    ("Cum loss",   "cum_loss",     11),
]


def _color_for(rec: TickRecord) -> QColor:
    if rec.action == "AGGRESSIVE":
        return COLOR_AGG
    if rec.action == "PASSIVE":
        return COLOR_PASS_FILL if rec.filled else COLOR_PASS_NOFILL
    if rec.action == "WAIT":
        return COLOR_WAIT
    return COLOR_GAP


def _format_cell(rec: TickRecord, key: Optional[str]) -> str:
    if key is None:
        # Time
        return datetime.fromtimestamp(rec.ts_ms / 1000.0, tz=timezone.utc).strftime("%H:%M:%S")
    if key == "tick_idx":
        return f"{rec.tick_idx}"
    if key == "mid":
        return f"{rec.mid:,.2f}"
    if key == "spread_abs":
        return f"{rec.spread_abs:,.2f}"
    if key == "pressure":
        return f"{rec.pressure:+.3f}"
    if key == "bucket":
        return f"{rec.bucket}"
    if key == "action":
        if rec.action == "PASSIVE":
            return "PASSIVE✓" if rec.filled else "PASSIVE×"
        return rec.action
    if key == "filled":
        return "Y" if rec.filled else "—"
    if key == "slippage":
        return f"{rec.slippage:+.4f}"
    if key == "adverse_move":
        return f"{rec.adverse_move:.4f}"
    if key == "loss":
        return f"{rec.loss:+.4f}"
    if key == "cum_loss":
        return f"{rec.cum_loss:+.2f}"
    return ""


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PacketModel(QAbstractTableModel):

    def __init__(self) -> None:
        super().__init__()
        self._rows: List[TickRecord] = []

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return COLUMNS[section][0]
        return str(section + 1)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        rec = self._rows[index.row()]
        col = index.column()
        key = COLUMNS[col][1]

        if role == Qt.ItemDataRole.DisplayRole:
            return _format_cell(rec, key)
        if role == Qt.ItemDataRole.BackgroundRole:
            return QBrush(_color_for(rec))
        if role == Qt.ItemDataRole.FontRole:
            return FONT_MONO
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if key in ("mid", "spread_abs", "pressure", "slippage",
                       "adverse_move", "loss", "cum_loss"):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    # ------------------------------------------------------------------
    # Public mutators
    # ------------------------------------------------------------------

    def append(self, rec: TickRecord) -> None:
        self.beginInsertRows(QModelIndex(), len(self._rows), len(self._rows))
        self._rows.append(rec)
        self.endInsertRows()

    def clear(self) -> None:
        self.beginResetModel()
        self._rows.clear()
        self.endResetModel()

    def at(self, row: int) -> Optional[TickRecord]:
        if 0 <= row < len(self._rows):
            return self._rows[row]
        return None


# ---------------------------------------------------------------------------
# View
# ---------------------------------------------------------------------------

class PacketTable(QTableView):

    row_selected = pyqtSignal(object)   # TickRecord

    def __init__(self) -> None:
        super().__init__()
        self._model = PacketModel()
        self.setModel(self._model)

        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setShowGrid(False)
        self.setAlternatingRowColors(False)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(20)
        self.horizontalHeader().setHighlightSections(False)

        # Initial column widths (chars * 8px)
        for i, (_, _, w_chars) in enumerate(COLUMNS):
            self.setColumnWidth(i, max(60, w_chars * 8))
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

        self._auto_scroll = True
        self.selectionModel().currentRowChanged.connect(self._on_row_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append_record(self, rec: TickRecord) -> None:
        self._model.append(rec)
        if self._auto_scroll:
            last = self._model.rowCount() - 1
            self.scrollTo(self._model.index(last, 0))

    def clear(self) -> None:
        self._model.clear()

    def set_auto_scroll(self, on: bool) -> None:
        self._auto_scroll = on

    def selected_record(self) -> Optional[TickRecord]:
        idx = self.selectionModel().currentIndex()
        if not idx.isValid():
            return None
        return self._model.at(idx.row())

    def _on_row_changed(self, current, _previous) -> None:
        if not current.isValid():
            return
        rec = self._model.at(current.row())
        if rec is not None:
            self.row_selected.emit(rec)
