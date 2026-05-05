"""
MainWindow — Wireshark-style assembly of toolbar + table + detail + chart.

Layout:

  +---------- toolbar -----------------------------------------------+
  | [Mode▾] [Session▾] [▶ Play] [⏸ Pause] [⏭ Step] [Speed▾] [Restart]|
  +---------- splitter (vertical) -----------------------------------+
  | PacketTable  (color-coded rows, scrolls live)                   |
  +---------- splitter (horizontal) ---------------------------------+
  | DetailPane (tree)        | ChartPane (mid + loss curves)        |
  +---------- status bar ---------------------------------------------+
  | mode | session | state | ticks | fills | cumloss | savings vs AGG|
  +------------------------------------------------------------------+
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui  import QAction
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSplitter, QStatusBar, QToolBar, QVBoxLayout, QWidget, QMessageBox,
)

from src.gui.chart_pane    import ChartPane
from src.gui.controllers   import LiveController, ReplayController
from src.gui.detail_pane   import DetailPane
from src.gui.packet_table  import PacketTable
from src.live.engine       import LiveEngine
from src.live.frozen_policy import load_frozen_weights, FrozenBucketedPolicy
from src.live.live_source  import LiveSource
from src.live.records      import TickRecord
from src.live.replay_source import ReplaySource
from src.state.kalman_filter import KalmanConfig
from src.utils.config import load_config


# ---------------------------------------------------------------------------
# Engine factory — produces a fresh engine for each new session
# ---------------------------------------------------------------------------

def _make_engine_factory(weights_path: Path, lambda_: float):
    def _factory():
        kalman_cfg = load_config("configs/kalman.yaml", KalmanConfig)
        frozen = load_frozen_weights(weights_path)
        # seed=0 so the inference draws are reproducible across replays
        policy = FrozenBucketedPolicy(frozen, seed=0)
        return LiveEngine(
            kalman_cfg=kalman_cfg,
            policy=policy,
            lambda_=lambda_,
            obs_features=kalman_cfg.obs_features,
        )
    return _factory


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):

    def __init__(
        self,
        weights_path: Path,
        replay_source: ReplaySource,
        live_source:  Optional[LiveSource] = None,
        lambda_: float = 0.10,
    ) -> None:
        super().__init__()
        self.setWindowTitle("adversity-is-controllable — live execution viewer")
        self.resize(1420, 920)

        # ---- frozen weights (used both by the engine factory and the chart) --
        self._frozen = load_frozen_weights(weights_path)

        # ---- engine factory + controllers --------------------------------
        engine_factory = _make_engine_factory(weights_path, lambda_)

        self._replay = ReplayController(replay_source, engine_factory)
        self._live   = LiveController(live_source, engine_factory) if live_source is not None else None

        self._mode = "replay"
        self._active = self._replay

        # ---- widgets -----------------------------------------------------
        self._table   = PacketTable()
        self._detail  = DetailPane()
        self._chart   = ChartPane()

        # Feed the chart pane the static per-bucket weights once.
        # The Action enum keys are converted to the string keys the chart wants.
        from src.policy.actions import Action as _Action
        chart_weights = [
            {
                "WAIT":       w[_Action.WAIT],
                "PASSIVE":    w[_Action.PASSIVE],
                "AGGRESSIVE": w[_Action.AGGRESSIVE],
            }
            for w in self._frozen.weights
        ]
        self._chart.set_bucket_weights(chart_weights, self._frozen.pressure_edges)

        self._table.row_selected.connect(self._detail.show_record)

        # Bottom split: detail (left) + chart (right)
        bottom = QSplitter(Qt.Orientation.Horizontal)
        bottom.addWidget(self._detail)
        bottom.addWidget(self._chart)
        bottom.setStretchFactor(0, 1)
        bottom.setStretchFactor(1, 3)

        # Vertical: table on top, bottom split below
        main_split = QSplitter(Qt.Orientation.Vertical)
        main_split.addWidget(self._table)
        main_split.addWidget(bottom)
        main_split.setStretchFactor(0, 3)
        main_split.setStretchFactor(1, 4)

        self.setCentralWidget(main_split)

        # ---- toolbar -----------------------------------------------------
        self._build_toolbar(replay_source, live_source is not None)

        # ---- status bar --------------------------------------------------
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status_session_label = QLabel("(no session)")
        self._status_state_label   = QLabel("paused")
        self._status_metrics_label = QLabel("ticks=0  fills=0  loss=0.00  saved=0.00")
        self._status.addPermanentWidget(self._status_session_label, 2)
        self._status.addPermanentWidget(self._status_state_label,   1)
        self._status.addPermanentWidget(self._status_metrics_label, 3)

        # ---- wiring ------------------------------------------------------
        self._wire(self._replay)
        if self._live is not None:
            self._wire(self._live)

        # Auto-load the longest replay session (the one most users will want)
        self._load_default_replay()

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------

    def _build_toolbar(self, replay: ReplaySource, has_live: bool) -> None:
        tb = QToolBar("controls", self)
        tb.setMovable(False)
        self.addToolBar(tb)

        # Mode selector
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Replay", userData="replay")
        if has_live:
            self._mode_combo.addItem("Live (Binance)", userData="live")
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        tb.addWidget(QLabel(" Mode: "))
        tb.addWidget(self._mode_combo)

        tb.addSeparator()

        # Session selector (replay only)
        self._session_combo = QComboBox()
        for s in replay.sessions:
            self._session_combo.addItem(s.label, userData=s.idx)
        self._session_combo.currentIndexChanged.connect(self._on_session_changed)
        tb.addWidget(QLabel(" Session: "))
        self._session_combo.setMinimumWidth(360)
        tb.addWidget(self._session_combo)

        tb.addSeparator()

        self._play_btn = QPushButton("▶ Play")
        self._play_btn.clicked.connect(self._on_play)
        tb.addWidget(self._play_btn)

        self._pause_btn = QPushButton("⏸ Pause")
        self._pause_btn.clicked.connect(self._on_pause)
        tb.addWidget(self._pause_btn)

        self._step_btn = QPushButton("⏭ Step")
        self._step_btn.clicked.connect(self._on_step)
        tb.addWidget(self._step_btn)

        tb.addSeparator()

        self._speed_combo = QComboBox()
        for s in ReplayController.SPEEDS:
            self._speed_combo.addItem(s)
        self._speed_combo.setCurrentText("10x")
        self._speed_combo.currentTextChanged.connect(self._on_speed_changed)
        tb.addWidget(QLabel(" Speed: "))
        tb.addWidget(self._speed_combo)

        tb.addSeparator()

        self._restart_btn = QPushButton("⟲ Restart session")
        self._restart_btn.clicked.connect(self._on_restart)
        tb.addWidget(self._restart_btn)

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def _wire(self, controller) -> None:
        controller.tick_ready.connect(self._on_tick_ready)
        controller.state_changed.connect(self._on_state_changed)
        controller.session_changed.connect(self._on_session_label)

    def _on_tick_ready(self, rec: TickRecord) -> None:
        self._table.append_record(rec)
        self._chart.append_record(rec)
        savings = rec.cum_aggr_loss - rec.cum_loss
        self._status_metrics_label.setText(
            f"ticks={rec.cum_ticks:,}  fills={rec.cum_filled:,}  "
            f"slip={rec.cum_slippage:+.2f}  adv={rec.cum_adverse:.0f}  "
            f"loss={rec.cum_loss:+.2f}  vs-AGGR={savings:+.2f}"
        )

    def _on_state_changed(self, state: str) -> None:
        self._status_state_label.setText(state)

    def _on_session_label(self, label: str) -> None:
        self._status_session_label.setText(label)

    # ------------------------------------------------------------------
    # Toolbar handlers
    # ------------------------------------------------------------------

    def _load_default_replay(self) -> None:
        if self._session_combo.count() == 0:
            QMessageBox.warning(
                self,
                "No sessions found",
                "Could not detect any sessions in data/derived/features/. "
                "Run scripts/3_featuregen.py first.",
            )
            return
        # Pick the longest session
        sessions = self._replay._source.sessions
        longest = max(range(len(sessions)), key=lambda i: sessions[i].ticks)
        self._session_combo.setCurrentIndex(longest)
        self._load_replay_session(sessions[longest].idx)

    def _load_replay_session(self, session_idx: int) -> None:
        self._table.clear()
        self._chart.clear()
        self._replay.start_session(session_idx)

    def _on_mode_changed(self, _idx: int) -> None:
        mode = self._mode_combo.currentData()
        if mode == self._mode:
            return

        # Stop whatever is currently active
        try:
            self._active.pause()
        except Exception:
            pass

        if mode == "live":
            if self._live is None:
                return
            self._mode = "live"
            self._active = self._live
            self._session_combo.setEnabled(False)
            self._speed_combo.setEnabled(False)
            self._step_btn.setEnabled(False)
            self._table.clear(); self._chart.clear()
            self._live.start_live()
        else:
            self._mode = "replay"
            self._active = self._replay
            self._session_combo.setEnabled(True)
            self._speed_combo.setEnabled(True)
            self._step_btn.setEnabled(True)
            self._table.clear(); self._chart.clear()
            self._load_default_replay()

    def _on_session_changed(self, _idx: int) -> None:
        if self._mode != "replay":
            return
        sid = self._session_combo.currentData()
        if sid is None:
            return
        self._load_replay_session(int(sid))

    def _on_play(self) -> None:
        self._active.play()

    def _on_pause(self) -> None:
        self._active.pause()

    def _on_step(self) -> None:
        if isinstance(self._active, ReplayController):
            self._active.step()

    def _on_speed_changed(self, label: str) -> None:
        if isinstance(self._active, ReplayController):
            self._active.set_speed(label)

    def _on_restart(self) -> None:
        if self._mode == "replay":
            sid = self._session_combo.currentData()
            if sid is not None:
                self._load_replay_session(int(sid))
        else:
            # In live mode, "restart" wipes the table and clears running totals.
            self._table.clear(); self._chart.clear()
            if self._live is not None:
                # Re-create the engine so cumulative totals reset
                self._live.start_live()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        if self._live is not None:
            self._live.stop()
        super().closeEvent(event)
