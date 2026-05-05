"""
src/live/ — live and replay engine for the Wireshark-style GUI.

This package wraps the existing offline pipeline (KalmanFilter, BucketedHedge
policy, simulator, loss) into a streaming engine that produces one TickRecord
per processed tick. It does NOT modify any existing src module.

Two sources are provided:
  - ReplaySource: reads a contiguous session from data/derived/features/*.parquet
                  and yields one input dict per tick at the requested speed
  - LiveSource:   subscribes to Binance BTCUSDT depth + trade WebSocket, builds
                  the book live via the existing BookBuilder + FeaturePipeline,
                  yields the same input shape

Both sources feed a LiveEngine that runs Kalman + frozen Hedge weights +
simulator + loss to produce TickRecord objects.

The GUI layer (src/gui/) wraps an engine in a Qt-aware controller and renders.
"""
