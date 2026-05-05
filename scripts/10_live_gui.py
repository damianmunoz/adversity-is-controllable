"""
scripts/10_live_gui.py — entry point for the Wireshark-style GUI.

Two modes:
  --mode replay (default): reads sessions from data/derived/features/ and
                           replays them through the frozen 1D policy. The
                           "savings vs always-AGGR" curve in the chart pane
                           proves the model retroactively on real data.
  --mode live  :           connects to Binance BTCUSDT @depth@100ms, builds
                           the book in process, and runs the same frozen
                           policy on incoming ticks.

Usage:
    PYTHONPATH=. python scripts/10_live_gui.py
    PYTHONPATH=. python scripts/10_live_gui.py --mode replay
    PYTHONPATH=. python scripts/10_live_gui.py --mode live
    PYTHONPATH=. python scripts/10_live_gui.py --weights data/derived/frozen_weights/1d_session7_seed0.json

If no frozen-weights JSON is found the script will tell you to run
scripts/9_export_frozen_weights.py first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMessageBox

from src.gui.main_window   import MainWindow
from src.live.live_source  import LiveSource
from src.live.replay_source import ReplaySource


DEFAULT_WEIGHTS_DIR = Path("data/derived/frozen_weights")


def _find_default_weights() -> Path | None:
    """Pick the most recently modified frozen-weights JSON, if any."""
    if not DEFAULT_WEIGHTS_DIR.exists():
        return None
    candidates = sorted(
        DEFAULT_WEIGHTS_DIR.glob("1d_session*_seed*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["replay", "live"], default="replay",
                   help="Initial mode (the GUI lets you switch at runtime).")
    p.add_argument("--weights", type=Path, default=None,
                   help="Path to frozen weights JSON. Auto-detected if omitted.")
    p.add_argument("--lambda", dest="lambda_", type=float, default=0.10,
                   help="λ for the loss function (default 0.10 — matches §9).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    weights_path = args.weights or _find_default_weights()

    app = QApplication(sys.argv)

    if weights_path is None or not weights_path.exists():
        QMessageBox.critical(
            None, "Missing frozen weights",
            "No frozen weights JSON found in data/derived/frozen_weights/.\n\n"
            "Run this first to generate one:\n\n"
            "    PYTHONPATH=. python scripts/9_export_frozen_weights.py "
            "--session 7 --seed 0\n\n"
            "Then relaunch this GUI.",
        )
        return 2

    replay = ReplaySource()
    # Always create the LiveSource — it doesn't open a connection until
    # start_live() is called, so the user can switch modes at runtime via
    # the toolbar without restarting the app.
    live: LiveSource | None = LiveSource()

    window = MainWindow(
        weights_path=weights_path,
        replay_source=replay,
        live_source=live,
        lambda_=args.lambda_,
    )
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
