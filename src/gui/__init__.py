"""
src/gui/ — PyQt6 + PyQtGraph Wireshark-style GUI for the live/replay engine.

Layout (mirrors Wireshark):

  +---------------------------------------------------------------+
  | menu / toolbar (mode, session, pause, step, speed)            |
  +---------------------------------------------------------------+
  | packet table  (one row per tick, color-coded by action)       |
  +---------------------------------------------------------------+
  | detail tree (selected tick — every field, grouped)            |
  |                                                               |
  | chart pane (mid + colored markers, cumulative loss line)      |
  +---------------------------------------------------------------+
  | status bar: cum loss, fills, action mix, live/replay state    |
  +---------------------------------------------------------------+

All Qt-aware code lives here. The src/live/ package is pure Python and
reusable in other contexts (e.g. headless evaluation).
"""
