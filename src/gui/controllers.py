"""
Engine controllers — Qt-aware glue between LiveEngine and the GUI.

Two controllers, same Qt signal interface (`tick_ready(TickRecord)`,
`session_changed(str)`, `state_changed(str)`):

  ReplayController — ticks come from a ReplaySource via QTimer at a chosen
                     speed. Pause/resume/step/seek all live here. This is
                     the mode that proves the model in retrospect.

  LiveController   — ticks come from a LiveSource (background thread that
                     consumes Binance WS). A QTimer in the main thread
                     drains LiveSource.next_row() and feeds the engine.
                     Pause stops emission but keeps consuming so resume
                     drains the buffered rows.

Both produce TickRecord objects that the rest of the GUI consumes
identically.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from src.live.engine import LiveEngine
from src.live.live_source import LiveSource
from src.live.records import TickRecord
from src.live.replay_source import ReplaySource, SessionInfo


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

class ReplayController(QObject):
    """Drives LiveEngine from a saved session."""

    tick_ready      = pyqtSignal(object)   # TickRecord
    state_changed   = pyqtSignal(str)      # "playing" / "paused" / "ended"
    session_changed = pyqtSignal(str)      # human label

    SPEEDS = {"1x": 1000, "10x": 100, "100x": 10, "1000x": 1, "max": 0}

    def __init__(self, source: ReplaySource, engine_factory) -> None:
        super().__init__()
        self._source: ReplaySource = source
        self._engine_factory = engine_factory   # () -> LiveEngine

        self._engine: Optional[LiveEngine] = None
        self._iter = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick_once)

        self._state = "paused"
        self._speed_label = "10x"
        self._timer.setInterval(self.SPEEDS[self._speed_label])

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------

    def start_session(self, session_idx: int) -> None:
        info: SessionInfo = self._source.session(session_idx)
        self._engine = self._engine_factory()
        self._iter = self._source.iter_session(session_idx)
        self._state = "paused"
        self.session_changed.emit(info.label)
        self.state_changed.emit(self._state)

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def play(self) -> None:
        if self._engine is None:
            return
        self._state = "playing"
        self._timer.start()
        self.state_changed.emit(self._state)

    def pause(self) -> None:
        self._timer.stop()
        if self._state != "ended":
            self._state = "paused"
        self.state_changed.emit(self._state)

    def toggle(self) -> None:
        if self._state == "playing":
            self.pause()
        else:
            self.play()

    def step(self) -> None:
        """Advance exactly one tick while paused."""
        if self._state == "playing":
            return
        self._tick_once()

    def set_speed(self, label: str) -> None:
        if label not in self.SPEEDS:
            return
        self._speed_label = label
        ms = self.SPEEDS[label]
        if ms == 0:
            ms = 1
        self._timer.setInterval(ms)

    @property
    def speed(self) -> str:
        return self._speed_label

    @property
    def state(self) -> str:
        return self._state

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _tick_once(self) -> None:
        if self._engine is None or self._iter is None:
            return
        try:
            row = next(self._iter)
        except StopIteration:
            self._timer.stop()
            self._state = "ended"
            self.state_changed.emit(self._state)
            return

        rec = self._engine.step(row)
        if rec is not None:
            self.tick_ready.emit(rec)


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------

class LiveController(QObject):
    """Drives LiveEngine from a Binance WebSocket source.

    The LiveSource runs on its own thread and pushes feature dicts onto a
    thread-safe queue. We poll that queue at 50ms in the main thread and
    feed each row into the engine. Pause stops emission; resume drains
    whatever the queue accumulated.
    """

    tick_ready      = pyqtSignal(object)
    state_changed   = pyqtSignal(str)
    session_changed = pyqtSignal(str)

    POLL_INTERVAL_MS = 50

    def __init__(self, source: LiveSource, engine_factory) -> None:
        super().__init__()
        self._source = source
        self._engine_factory = engine_factory

        self._engine: Optional[LiveEngine] = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._drain)
        self._timer.setInterval(self.POLL_INTERVAL_MS)

        self._state = "paused"

    def start_live(self) -> None:
        self._engine = self._engine_factory()
        self._source.start()
        self._state = "paused"
        self.session_changed.emit("LIVE — Binance BTCUSDT @ depth@100ms")
        self.state_changed.emit(self._state)

    def play(self) -> None:
        if self._engine is None:
            return
        self._state = "playing"
        self._timer.start()
        self.state_changed.emit(self._state)

    def pause(self) -> None:
        self._timer.stop()
        self._state = "paused"
        self.state_changed.emit(self._state)

    def toggle(self) -> None:
        if self._state == "playing":
            self.pause()
        else:
            self.play()

    def stop(self) -> None:
        self._timer.stop()
        self._source.stop()
        self._state = "ended"
        self.state_changed.emit(self._state)

    @property
    def state(self) -> str:
        return self._state

    def _drain(self) -> None:
        if self._engine is None:
            return
        # Drain everything pending in one timer tick. Caps loop at 200 rows
        # to keep the UI responsive if a backlog has accumulated.
        for _ in range(200):
            row = self._source.next_row()
            if row is None:
                return
            rec = self._engine.step(row)
            if rec is not None:
                self.tick_ready.emit(rec)
