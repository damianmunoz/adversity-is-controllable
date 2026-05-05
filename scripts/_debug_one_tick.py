"""
scripts/_debug_one_tick.py — instrument one real tick end-to-end and dump
every intermediate value. Used to populate the worked-example section of
docs/math_explained.md with exact numbers.

Picks a tick from session S7 where pressure has had time to grow large
enough to land in a "decided" bucket (so the action choice is meaningful),
then prints:

  - the parquet row at tick N (and the next row, for fill simulation)
  - the Kalman prior (x_{N-1|N-1}, P_{N-1|N-1}) carried over from the
    full replay tick 0..N-1
  - the standardization of the new observation z
  - the predict step (x_pred, P_pred) with intermediate matrix products
  - the update step (innovation y, innov cov S, gain K, posterior x, P)
  - the bucket lookup
  - the frozen weights for that bucket
  - the sampling step with a fixed RNG seed
  - the simulator output (filled?, fill_price, slippage, adverse)
  - the Hedge update of that bucket's weights

Run with:
    PYTHONPATH=. python scripts/_debug_one_tick.py
"""

from __future__ import annotations

import bisect
import json
import logging
import math
from pathlib import Path

import numpy as np

logging.disable(logging.CRITICAL)

from src.execution.loss import compute_loss
from src.execution.simulator import simulate_fill
from src.policy.actions import Action
from src.state.kalman_filter import KalmanConfig, KalmanFilter
from src.utils.config import load_config
from src.utils.io import iter_parquet_dir


SESSION_GAP_MS = 5_000


def _detect(rows):
    out, start = [], 0
    for i in range(1, len(rows)):
        tg = rows[i]["ts_ms"] - rows[i - 1]["ts_ms"]
        if rows[i].get("sequence_gap") or tg > SESSION_GAP_MS:
            out.append((start, i)); start = i
    out.append((start, len(rows)))
    return out


def _h(s):
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


def _kv(label, val, w=22):
    print(f"  {label:<{w}}{val}")


def main():
    target_session = 7
    target_tick    = 800       # past warmup; pressure has grown
    target_seed    = 0

    kalman_cfg = load_config("configs/kalman.yaml", KalmanConfig)
    obs        = list(kalman_cfg.obs_features)
    pressure_edges = [-0.5, -0.2, 0.0, 0.2, 0.5]
    eta        = 0.10
    lambda_    = 0.10

    rows = list(iter_parquet_dir(Path("data/derived/features")))
    sessions = _detect(rows)
    a, b = sessions[target_session - 1]
    sub = rows[a:b]

    # Run Kalman from tick 0 through tick (target_tick - 1) to set up the
    # prior state that the target tick will operate on.
    kalman = KalmanFilter(kalman_cfg)
    prev = None
    last_state = None
    for i, row in enumerate(sub[:target_tick]):
        if prev is None:
            prev = row; continue
        z = np.array([prev[f] for f in obs], dtype=float)
        last_state = kalman.step(prev["ts_ms"], z)
        prev = row

    # Capture the prior x and P just BEFORE the target tick's update.
    # The KalmanFilter exposes _x and _P internally; the way step() works
    # is it always returns the POSTERIOR state, so after the loop above
    # _x and _P are already the posterior of tick (target_tick - 1).
    x_prior = kalman._x.copy()
    P_prior = kalman._P.copy()

    # Now grab the target tick's parquet row + the one after it.
    curr = sub[target_tick]
    nxt  = sub[target_tick + 1]

    # ----------------------------------------------------------------------
    _h(f"PARQUET ROWS — session S{target_session}, tick {target_tick}")
    print("Current row (the decision tick):")
    for k in ("ts_ms", "best_bid_px", "best_ask_px", "spread_abs",
              "mid_price", "microprice", "depth_imbalance",
              "ofi_l1", "vol_30s"):
        _kv(k, f"{curr[k]:>18}")
    print("\nNext row (used by simulator for fill outcome):")
    for k in ("ts_ms", "mid_price"):
        _kv(k, f"{nxt[k]:>18}")

    # ----------------------------------------------------------------------
    _h("KALMAN — prior state (carried over from tick 0..N-1)")
    _kv("x_prior[0] (pressure)", f"{x_prior[0]:.6f}")
    _kv("x_prior[1] (regime)",   f"{x_prior[1]:.6f}")
    _kv("P_prior[0,0]",          f"{P_prior[0,0]:.6f}")
    _kv("P_prior[0,1]",          f"{P_prior[0,1]:.6f}")
    _kv("P_prior[1,0]",          f"{P_prior[1,0]:.6f}")
    _kv("P_prior[1,1]",          f"{P_prior[1,1]:.6f}")

    # ----------------------------------------------------------------------
    _h("STEP A — standardize the observation")
    z_raw = np.array([curr[f] for f in obs], dtype=float)
    print(f"z_raw  (depth_imb, ofi_l1, vol_30s) = {z_raw}")
    print(f"obs_center                           = {kalman_cfg.obs_center}")
    print(f"obs_scale                            = {kalman_cfg.obs_scale}")
    z_norm = (z_raw - np.array(kalman_cfg.obs_center)) * np.array(kalman_cfg.obs_scale)
    print(f"z_norm = (z_raw - center) * scale    = {z_norm}")

    # ----------------------------------------------------------------------
    _h("STEP B — Kalman predict")
    F = kalman.F
    Q = kalman.Q
    print(f"F =\n{F}")
    print(f"Q =\n{Q}")
    x_pred = F @ x_prior
    P_pred = F @ P_prior @ F.T + Q
    print(f"\nx_pred = F · x_prior          = {x_pred}")
    print(f"P_pred = F · P · F^T + Q     =\n{P_pred}")

    # ----------------------------------------------------------------------
    _h("STEP C — Kalman update (innovation, gain, posterior)")
    H = kalman.H
    R = kalman.R
    print(f"H =\n{H}")
    print(f"R =\n{R}")
    y = z_norm - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_post = x_pred + K @ y
    P_post = (np.eye(2) - K @ H) @ P_pred
    print(f"\ny = z_norm - H · x_pred       = {y}")
    print(f"S = H · P_pred · H^T + R     =\n{S}")
    print(f"K (Kalman gain, 2x3)         =\n{K}")
    print(f"\nx_posterior = x_pred + K·y    = {x_post}")
    print(f"P_posterior                   =\n{P_post}")
    print(f"\n→ pressure estimate           = {x_post[0]:.6f}")
    print(f"→ regime   estimate           = {x_post[1]:.6f}")

    # ----------------------------------------------------------------------
    _h("STEP D — bucket lookup")
    pressure = x_post[0]
    bucket = bisect.bisect_right(pressure_edges, pressure)
    _kv("pressure value",          f"{pressure:+.6f}")
    _kv("pressure_edges",          str(pressure_edges))
    _kv("bucket index (0..5)",     bucket)
    if bucket == 0:
        rng = "(-inf, -0.500]"
    elif bucket == len(pressure_edges):
        rng = "(+0.500, +inf)"
    else:
        rng = f"({pressure_edges[bucket-1]:+.3f}, {pressure_edges[bucket]:+.3f}]"
    _kv("bucket range",            rng)

    # ----------------------------------------------------------------------
    _h("STEP E — load frozen weights for that bucket")
    fz_path = next(Path("data/derived/frozen_weights").glob("1d_session*_seed*.json"))
    with open(fz_path) as f:
        fz = json.load(f)
    bucket_weights = fz["buckets"][bucket]["weights"]
    visits = fz["buckets"][bucket]["visits"]
    print(f"frozen weights file:  {fz_path.name}")
    print(f"bucket {bucket} visit count during training: {visits:,}")
    print(f"\nweights distribution for bucket {bucket}:")
    for action_name, w in bucket_weights.items():
        bar = "█" * int(round(w * 50))
        print(f"  {action_name:<12} {w:.4f}  {bar}")
    total = sum(bucket_weights.values())
    print(f"\nsum of weights = {total:.4f}  (must equal 1)")

    # ----------------------------------------------------------------------
    _h("STEP F — sample an action")
    rng = np.random.default_rng(target_seed)
    actions  = ["WAIT", "PASSIVE", "AGGRESSIVE"]
    probs    = [bucket_weights[a] for a in actions]
    cumsum   = np.cumsum(probs)
    print(f"cumulative probs:  {cumsum}")
    u = float(rng.random())
    idx = int(np.searchsorted(cumsum, u))
    chosen = actions[idx]
    print(f"\nuniform(0,1) draw = {u:.4f}")
    print(f"  searchsorted({u:.4f}, {cumsum.tolist()}) = {idx}")
    print(f"chosen action     = {chosen}")

    # ----------------------------------------------------------------------
    _h("STEP G — simulator + loss")
    fill = simulate_fill(
        action=Action(chosen), ts_ms=curr["ts_ms"],
        curr_mid=curr["mid_price"],
        curr_best_bid=curr["best_bid_px"],
        curr_best_ask=curr["best_ask_px"],
        next_mid=nxt["mid_price"],
    )
    L = compute_loss(fill, lambda_)
    slippage = (fill.fill_price - fill.mid_price) if fill.filled else 0.0
    adverse  = (max(0.0, fill.fill_price - fill.next_mid_price) if fill.filled
                else max(0.0, fill.next_mid_price - fill.mid_price))
    _kv("filled",        fill.filled)
    _kv("fill_price",    f"{fill.fill_price:.4f}")
    _kv("mid",           f"{fill.mid_price:.4f}")
    _kv("next_mid",      f"{fill.next_mid_price:.4f}")
    _kv("slippage",      f"{slippage:+.6f}")
    _kv("adverse_move",  f"{adverse:.6f}")
    _kv("lambda",        f"{lambda_:.2f}")
    _kv("loss = slip + λ·adv", f"{L:+.6f}")

    # ----------------------------------------------------------------------
    _h("STEP H — Hedge update for the chosen action")
    print(f"old weight for {chosen} = {bucket_weights[chosen]:.4f}")
    factor = math.exp(-eta * L)
    print(f"multiplicative factor = exp(-η·L) = exp(-{eta}·{L:+.6f}) = {factor:.6f}")
    new_unnormalized = dict(bucket_weights)
    new_unnormalized[chosen] = bucket_weights[chosen] * factor
    print(f"\nweights after multiplying (UNNORMALIZED):")
    for a in actions:
        print(f"  {a:<12} {new_unnormalized[a]:.6f}")
    total_un = sum(new_unnormalized.values())
    print(f"\nsum (must renormalize): {total_un:.6f}")
    new_normalized = {a: w / total_un for a, w in new_unnormalized.items()}
    print(f"\nweights after RENORMALIZATION (these go into next tick):")
    for a in actions:
        delta = new_normalized[a] - bucket_weights[a]
        sign  = "+" if delta >= 0 else ""
        print(f"  {a:<12} {new_normalized[a]:.4f}   (delta vs old: {sign}{delta:+.6f})")
    print(f"\nsum: {sum(new_normalized.values()):.6f}")


if __name__ == "__main__":
    main()
