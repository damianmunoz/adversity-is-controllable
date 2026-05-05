"""
scripts/_plots_preview.py — generates ALL the figures for the slide-deck
preview. Run once after the new harness; re-runnable any time from the
JSON outputs already on disk.

Outputs (data/derived/figures/preview/):
  p01_marginal_vs_1d_bars.png         — 4-session bar chart
  p02_marginal_vs_1d_paired.png       — paired diff with |t| labels
  p03_marginal_vs_1d_scatter.png      — per-seed scatter, all 4 sessions
  p04_kalman_trace.png                — mid + pressure + regime over time
  p05_features_trace.png              — depth_imb + ofi_l1 + vol_30s over time
  p06_hedge_weights_evolution.png     — bucket weights converging through a session
  p07_per_bucket_final_weights.png    — what the policy learned, per bucket
  p08_action_mix_marginal_vs_1d.png   — WAIT/PASSIVE/AGG share by mode
  p09_savings_vs_aggr.png             — cumulative savings vs always-AGGR over time
  p10_loss_decomposition.png          — slippage + λ·adverse stacked, per mode

Re-run with:
    PYTHONPATH=. python scripts/_plots_preview.py
"""

from __future__ import annotations

import json
import math
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logging.disable(logging.CRITICAL)

from src.execution.loss import compute_loss
from src.execution.simulator import simulate_fill
from src.execution.updater import ExecutionConfig
from src.policy.actions import Action
from src.policy.hedge import PolicyConfig, PolicyContext, make_policy
from src.state.kalman_filter import KalmanConfig, KalmanFilter
from src.utils.config import load_config
from src.utils.io import iter_parquet_dir


FIG_DIR        = Path("data/derived/figures/preview")
HARNESS_NEW    = Path("data/derived/ab_1d_vs_marginal_4sessions.json")
HARNESS_OLD    = Path("data/derived/ab_adverse_3sessions_10seed.json")
FEATURES_DIR   = Path("data/derived/features")
SESSION_GAP_MS = 5_000

plt.rcParams.update({
    "figure.dpi":     120,
    "savefig.dpi":    150,
    "font.size":      10.5,
    "axes.titlesize": 12,
    "axes.labelsize": 10.5,
    "axes.grid":      True,
    "grid.alpha":     0.25,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.edgecolor":    "#444444",
})

C_MARGINAL = "#888888"
C_1D       = "#1f77b4"
C_2D       = "#ff7f0e"
C_VOL      = "#2ca02c"
C_OFI      = "#9467bd"
C_SPREAD   = "#d62728"
C_GAIN     = "#2ca02c"

ACTION_COLORS = {
    "WAIT":       "#f7c52d",
    "PASSIVE":    "#d62728",
    "AGGRESSIVE": "#2ca02c",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_harness(p: Path):
    with open(p) as f:
        return json.load(f)


def _detect_sessions(rows):
    out, start = [], 0
    for i in range(1, len(rows)):
        tg = rows[i]["ts_ms"] - rows[i - 1]["ts_ms"]
        if rows[i].get("sequence_gap") or tg > SESSION_GAP_MS:
            out.append((start, i)); start = i
    out.append((start, len(rows)))
    return out


def _paired_diff_stats(losses_a, losses_b):
    """Return (mean_diff, std_diff, t_stat) where diff = a - b."""
    diffs = np.asarray(losses_a) - np.asarray(losses_b)
    mean = float(diffs.mean())
    std  = float(diffs.std(ddof=1))
    t    = mean / (std / math.sqrt(len(diffs))) if std > 0 else 0.0
    return mean, std, t


def _by_session_seed(harness, mode):
    """Build {(session_idx): [seeds-ordered losses]} for a given mode."""
    out: Dict[int, Dict[int, float]] = {}
    for r in harness["results"]:
        if r["mode"] != mode:
            continue
        out.setdefault(r["session_idx"], {})[r["seed"]] = r["total_loss"]
    return {sid: [v for _, v in sorted(d.items())] for sid, d in out.items()}


# ---------------------------------------------------------------------------
# 1. Marginal vs 1D — bar chart with error bars
# ---------------------------------------------------------------------------

def plot_marginal_vs_1d_bars(harness):
    margl = _by_session_seed(harness, "marginal")
    onel  = _by_session_seed(harness, "1D_bucketed")

    sessions = sorted(margl.keys())
    n = len(sessions)
    x = np.arange(n)
    w = 0.36

    m_means = [np.mean(margl[s]) for s in sessions]
    m_std   = [np.std(margl[s], ddof=1) for s in sessions]
    o_means = [np.mean(onel[s]) for s in sessions]
    o_std   = [np.std(onel[s], ddof=1) for s in sessions]

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    b1 = ax.bar(x - w/2, m_means, w, yerr=m_std, capsize=4,
                color=C_MARGINAL, label="Hedge marginal (sin estado)")
    b2 = ax.bar(x + w/2, o_means, w, yerr=o_std, capsize=4,
                color=C_1D, label="Hedge 1D bucketed (condicionado en presión)")

    # Improvement labels on top of pairs
    for xi, m, o in zip(x, m_means, o_means):
        improvement = (m - o) / m * 100
        ymax = max(m, o)
        ax.text(xi, ymax + ymax*0.04, f"−{improvement:.1f}%",
                ha="center", fontsize=11, fontweight="bold", color="#1a6c1a")

    labels = [f"S{s}\n({_session_brief(harness, s)})" for s in sessions]
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Pérdida total acumulada (slippage + λ·adverse)")
    ax.set_title("Pérdida total: Hedge marginal vs Hedge condicionado por presión\n"
                 "(media ± desviación estándar sobre 10 semillas, λ = 0.1)")
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p01_marginal_vs_1d_bars.png")
    plt.close(fig)


def _session_brief(harness, sid):
    """Pull a short label like '38k ticks / 10.8h' from the harness rows."""
    for r in harness["results"]:
        if r["session_idx"] == sid:
            return f"{r['session_n_rows']/1000:.0f}k ticks / {r['session_duration_h']:.1f}h"
    return f"S{sid}"


# ---------------------------------------------------------------------------
# 2. Paired diff with |t|
# ---------------------------------------------------------------------------

def plot_paired_diff(harness):
    margl = _by_session_seed(harness, "marginal")
    onel  = _by_session_seed(harness, "1D_bucketed")

    sessions = sorted(margl.keys())
    diffs_mean = []
    diffs_std  = []
    t_stats    = []
    for s in sessions:
        m, sd, t = _paired_diff_stats(onel[s], margl[s])
        diffs_mean.append(m); diffs_std.append(sd); t_stats.append(t)

    x = np.arange(len(sessions))
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    bars = ax.bar(x, diffs_mean, yerr=diffs_std, capsize=4,
                  color=C_1D, edgecolor="#0d3b66")
    for xi, m, t in zip(x, diffs_mean, t_stats):
        ax.text(xi, m - 30, f"|t|={abs(t):.1f}",
                ha="center", fontsize=11, fontweight="bold", color="white")

    ax.axhline(0, color="black", linewidth=1)
    labels = [f"S{s}\n({_session_brief(harness, s)})" for s in sessions]
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Diferencia pareada (1D − marginal)")
    ax.set_title("Diferencia pareada por semilla — todas negativas, todas significativas\n"
                 "(barras hacia abajo = 1D vence a marginal)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p02_marginal_vs_1d_paired.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Per-seed scatter (4 panels, one per session)
# ---------------------------------------------------------------------------

def plot_scatter(harness):
    margl = _by_session_seed(harness, "marginal")
    onel  = _by_session_seed(harness, "1D_bucketed")
    sessions = sorted(margl.keys())

    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.0), sharey=False)
    for ax, s in zip(axes, sessions):
        ax.scatter(margl[s], onel[s], color=C_1D, edgecolor="#0d3b66", s=70, zorder=3)
        # y=x reference
        lo = min(min(margl[s]), min(onel[s])) * 0.97
        hi = max(max(margl[s]), max(onel[s])) * 1.03
        ax.plot([lo, hi], [lo, hi], "--", color="#666666", linewidth=1, label="y = x")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Pérdida — marginal")
        ax.set_ylabel("Pérdida — 1D bucketed")
        wins = sum(1 for a, b in zip(onel[s], margl[s]) if a < b)
        ax.set_title(f"S{s} — {wins}/10 semillas\n"
                     f"({_session_brief(harness, s)})")
        ax.legend(loc="upper left")

    fig.suptitle("Comparación pareada por semilla (cada punto = una semilla)\n"
                 "Punto debajo de la línea y=x  →  1D venció a marginal en esa semilla",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p03_marginal_vs_1d_scatter.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4-5. Kalman trace + features trace on one representative session
# ---------------------------------------------------------------------------

def plot_kalman_and_features(target_session=2, n_ticks=2000, start_offset=2000):
    """Run Kalman live on a slice of one session and plot everything."""
    kalman_cfg = load_config("configs/kalman.yaml", KalmanConfig)
    rows = list(iter_parquet_dir(FEATURES_DIR))
    sessions = _detect_sessions(rows)
    a, b = sessions[target_session - 1]
    sub = rows[a + start_offset : a + start_offset + n_ticks]
    obs = list(kalman_cfg.obs_features)

    kalman = KalmanFilter(kalman_cfg)
    t       = []
    mid     = []
    pressure = []
    regime   = []
    depth_im = []
    ofi      = []
    vol30    = []
    for r in sub:
        z = np.array([r[f] for f in obs], dtype=float)
        ks = kalman.step(r["ts_ms"], z)
        if ks is None:
            continue
        t.append(r["ts_ms"]/1000)
        mid.append(r["mid_price"])
        pressure.append(ks.market_pressure)
        regime.append(ks.regime)
        depth_im.append(r["depth_imbalance"])
        ofi.append(r["ofi_l1"])
        vol30.append(r["vol_30s"])
    t = np.array(t); t = (t - t[0]) / 60.0   # minutes from start

    # ------ p04 — Kalman trace ------
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    axes[0].plot(t, mid, color="#1f77b4", linewidth=1.2)
    axes[0].set_ylabel("Mid price (USDT)")
    axes[0].set_title("Lo que ocurre en el mercado")

    axes[1].plot(t, pressure, color="#d62728", linewidth=1.0)
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].fill_between(t, 0, pressure, where=np.array(pressure) > 0,
                         alpha=0.18, color="#2ca02c", label="presión compradora")
    axes[1].fill_between(t, 0, pressure, where=np.array(pressure) < 0,
                         alpha=0.18, color="#d62728", label="presión vendedora")
    axes[1].set_ylabel("Presión (Kalman)")
    axes[1].set_title("Lo que el filtro de Kalman estima — dirección de la presión")
    axes[1].legend(loc="upper right", fontsize=9)

    axes[2].plot(t, regime, color="#9467bd", linewidth=1.0)
    axes[2].axhline(0, color="black", linewidth=0.6)
    axes[2].set_ylabel("Régimen (Kalman)")
    axes[2].set_xlabel("Tiempo dentro de la sesión (minutos)")
    axes[2].set_title("Lo que el filtro de Kalman estima — turbulencia / régimen")

    fig.suptitle(f"Filtro de Kalman en operación — sesión S{target_session}, "
                 f"ventana de {n_ticks} ticks (~{n_ticks//60} min)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p04_kalman_trace.png")
    plt.close(fig)

    # ------ p05 — features trace ------
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(t, depth_im, color="#1f77b4", linewidth=1.0)
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[0].set_ylabel("depth_imbalance")
    axes[0].set_title("Las 3 features de entrada (las cosas crudas que mide el sistema)")

    axes[1].plot(t, ofi, color="#ff7f0e", linewidth=1.0)
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_ylabel("ofi_l1")

    axes[2].plot(t, vol30, color="#2ca02c", linewidth=1.0)
    axes[2].set_ylabel("vol_30s")
    axes[2].set_xlabel("Tiempo dentro de la sesión (minutos)")

    fig.suptitle(f"Features observables — entrada del filtro de Kalman, "
                 f"sesión S{target_session}",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p05_features_trace.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Hedge weights evolving over a session (sampled snapshots)
# ---------------------------------------------------------------------------

def plot_hedge_evolution(target_session=7, seed=0, n_snapshots=80):
    kalman_cfg = load_config("configs/kalman.yaml", KalmanConfig)
    policy_cfg = load_config("configs/policy.yaml", PolicyConfig)
    exec_cfg   = load_config("configs/execution.yaml", ExecutionConfig)
    obs        = list(kalman_cfg.obs_features)

    rows = list(iter_parquet_dir(FEATURES_DIR))
    sessions = _detect_sessions(rows)
    a, b = sessions[target_session - 1]
    sub = rows[a:b]

    pcfg = PolicyConfig(
        learning_rate  = policy_cfg.learning_rate,
        initial_weight = policy_cfg.initial_weight,
        mode           = "bucketed",
        pressure_edges = list(policy_cfg.pressure_edges),
    )
    kalman = KalmanFilter(kalman_cfg)
    policy = make_policy(pcfg, seed=seed)

    # We'll snapshot weights of the 4 inner buckets (1, 2, 3, 4) at uniform intervals
    snap_idx = np.linspace(1, len(sub) - 1, n_snapshots, dtype=int)
    snaps = []   # list of dicts {bucket: {action: weight}}

    prev = None
    snap_set = set(snap_idx.tolist())
    for i, row in enumerate(sub):
        if prev is None:
            prev = row; continue
        z = np.array([prev[f] for f in obs], dtype=float)
        ks = kalman.step(prev["ts_ms"], z)
        if ks is None:
            prev = row; continue
        ctx = PolicyContext(prev["ts_ms"], ks.market_pressure, ks.regime)
        d = policy.select(ctx)
        a_e = Action(d.action)
        f = simulate_fill(action=a_e, ts_ms=prev["ts_ms"],
                          curr_mid=prev["mid_price"],
                          curr_best_bid=prev["best_bid_px"],
                          curr_best_ask=prev["best_ask_px"],
                          next_mid=row["mid_price"])
        L = compute_loss(f, exec_cfg.lambda_)
        policy.update(a_e, L)
        if i in snap_set:
            snaps.append({
                b["bucket"]: {
                    "WAIT":       b["weights"][Action.WAIT],
                    "PASSIVE":    b["weights"][Action.PASSIVE],
                    "AGGRESSIVE": b["weights"][Action.AGGRESSIVE],
                }
                for b in policy.bucket_summary()
            })
        prev = row

    # Plot 4 panels (buckets 1-4 are the well-visited ones)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    bucket_labels = {
        1: "Bucket 1: presión vendedora moderada (-0.5, -0.2]",
        2: "Bucket 2: presión vendedora ligera (-0.2, 0]",
        3: "Bucket 3: presión compradora ligera (0, +0.2]",
        4: "Bucket 4: presión compradora moderada (+0.2, +0.5]",
    }
    x = np.linspace(0, 100, n_snapshots)   # % of session

    for ax, b_idx in zip(axes.flat, [1, 2, 3, 4]):
        wait_arr = [s[b_idx]["WAIT"]       for s in snaps]
        pass_arr = [s[b_idx]["PASSIVE"]    for s in snaps]
        aggr_arr = [s[b_idx]["AGGRESSIVE"] for s in snaps]
        ax.fill_between(x, 0, wait_arr, color=ACTION_COLORS["WAIT"], alpha=0.85,
                        label="P(WAIT)")
        ax.fill_between(x, wait_arr,
                        [w + p for w, p in zip(wait_arr, pass_arr)],
                        color=ACTION_COLORS["PASSIVE"], alpha=0.85, label="P(PASSIVE)")
        ax.fill_between(x,
                        [w + p for w, p in zip(wait_arr, pass_arr)],
                        [1.0 for _ in x],
                        color=ACTION_COLORS["AGGRESSIVE"], alpha=0.85, label="P(AGGRESSIVE)")
        ax.set_title(bucket_labels[b_idx], fontsize=10.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probabilidad de cada acción")
        ax.set_xlim(0, 100)

    axes[1, 0].set_xlabel("Avance de la sesión (%)")
    axes[1, 1].set_xlabel("Avance de la sesión (%)")
    axes[0, 0].legend(loc="center right", fontsize=8.5, framealpha=0.9)

    fig.suptitle(f"Convergencia del Hedge — pesos por bucket a lo largo de S{target_session} "
                 f"(semilla = {seed})\n"
                 "Cada panel = una región del estado del mercado. "
                 "El algoritmo aprende qué acción funciona mejor en cada región.",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p06_hedge_weights_evolution.png")
    plt.close(fig)

    return policy   # so we can reuse for plot 7 if needed


# ---------------------------------------------------------------------------
# 7. Per-bucket final weights (loaded from the frozen weights JSON)
# ---------------------------------------------------------------------------

def plot_per_bucket_final_weights():
    fz_path = next(Path("data/derived/frozen_weights").glob("1d_session*_seed*.json"))
    with open(fz_path) as f:
        fz = json.load(f)

    n = len(fz["buckets"])
    x = np.arange(n)
    w_wait = [b["weights"]["WAIT"]       for b in fz["buckets"]]
    w_pass = [b["weights"]["PASSIVE"]    for b in fz["buckets"]]
    w_aggr = [b["weights"]["AGGRESSIVE"] for b in fz["buckets"]]
    visits = [b["visits"] for b in fz["buckets"]]

    fig, ax = plt.subplots(figsize=(11, 6))
    bw = 0.7
    ax.bar(x, w_wait, bw, color=ACTION_COLORS["WAIT"],       label="P(WAIT)")
    ax.bar(x, w_pass, bw, bottom=w_wait, color=ACTION_COLORS["PASSIVE"], label="P(PASSIVE)")
    ax.bar(x, w_aggr, bw, bottom=[w + p for w, p in zip(w_wait, w_pass)],
           color=ACTION_COLORS["AGGRESSIVE"], label="P(AGGRESSIVE)")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Probabilidad")
    edges = fz["pressure_edges"]
    labels = []
    for i in range(n):
        lo = "-∞" if i == 0           else f"{edges[i-1]:+.2f}"
        hi = "+∞" if i == n - 1       else f"{edges[i]:+.2f}"
        labels.append(f"Bucket {i}\n({lo}, {hi}]\n{visits[i]:,} visitas")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title(f"Política aprendida — pesos finales por bucket de presión\n"
                 f"(sesión {fz['session_idx']}, semilla {fz['seed']}, {fz['ticks']:,} ticks)\n"
                 "Cada barra suma 1 — es la distribución sobre WAIT/PASSIVE/AGGRESSIVE en ese bucket",
                 fontsize=11.5)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p07_per_bucket_final_weights.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Action mix marginal vs 1D
# ---------------------------------------------------------------------------

def plot_action_mix(harness):
    sessions = sorted({r["session_idx"] for r in harness["results"]})

    fig, axes = plt.subplots(1, len(sessions), figsize=(15.0, 4.0), sharey=True)

    for ax, s in zip(axes, sessions):
        marg = [r for r in harness["results"]
                if r["mode"] == "marginal" and r["session_idx"] == s]
        onel = [r for r in harness["results"]
                if r["mode"] == "1D_bucketed" and r["session_idx"] == s]

        m_w = np.mean([r["wait_pct"]       for r in marg])
        m_p = np.mean([r["passive_pct"]    for r in marg])
        m_a = np.mean([r["aggressive_pct"] for r in marg])
        o_w = np.mean([r["wait_pct"]       for r in onel])
        o_p = np.mean([r["passive_pct"]    for r in onel])
        o_a = np.mean([r["aggressive_pct"] for r in onel])

        x = np.arange(2)
        ax.bar(x, [m_w, o_w], color=ACTION_COLORS["WAIT"],       label="WAIT")
        ax.bar(x, [m_p, o_p], bottom=[m_w, o_w],
               color=ACTION_COLORS["PASSIVE"], label="PASSIVE")
        ax.bar(x, [m_a, o_a], bottom=[m_w + m_p, o_w + o_p],
               color=ACTION_COLORS["AGGRESSIVE"], label="AGGRESSIVE")
        ax.set_xticks(x); ax.set_xticklabels(["marginal", "1D"])
        ax.set_title(f"S{s}")
        ax.set_ylim(0, 100)

    axes[0].set_ylabel("Porcentaje de acciones (%)")
    axes[-1].legend(loc="upper right", bbox_to_anchor=(1.5, 1.0), framealpha=0.95)
    fig.suptitle("Distribución de acciones — ¿en qué se diferencian las dos políticas?",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p08_action_mix_marginal_vs_1d.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 9. Cumulative savings vs always-AGGR — replay the policy on one session
# ---------------------------------------------------------------------------

def plot_savings_curve(target_session=7, seed=0):
    kalman_cfg = load_config("configs/kalman.yaml", KalmanConfig)
    policy_cfg = load_config("configs/policy.yaml", PolicyConfig)
    exec_cfg   = load_config("configs/execution.yaml", ExecutionConfig)
    obs        = list(kalman_cfg.obs_features)

    rows = list(iter_parquet_dir(FEATURES_DIR))
    sessions = _detect_sessions(rows)
    a, b = sessions[target_session - 1]
    sub = rows[a:b]

    pcfg = PolicyConfig(
        learning_rate  = policy_cfg.learning_rate,
        initial_weight = policy_cfg.initial_weight,
        mode           = "bucketed",
        pressure_edges = list(policy_cfg.pressure_edges),
    )
    kalman = KalmanFilter(kalman_cfg)
    policy = make_policy(pcfg, seed=seed)

    cum_loss = [0.0]
    cum_aggr = [0.0]
    t        = [0]

    prev = None
    tick_idx = 0
    for i, row in enumerate(sub):
        if prev is None:
            prev = row; continue
        z = np.array([prev[f] for f in obs], dtype=float)
        ks = kalman.step(prev["ts_ms"], z)
        if ks is None:
            prev = row; continue
        ctx = PolicyContext(prev["ts_ms"], ks.market_pressure, ks.regime)
        d = policy.select(ctx)
        a_e = Action(d.action)
        f = simulate_fill(action=a_e, ts_ms=prev["ts_ms"],
                          curr_mid=prev["mid_price"],
                          curr_best_bid=prev["best_bid_px"],
                          curr_best_ask=prev["best_ask_px"],
                          next_mid=row["mid_price"])
        L = compute_loss(f, exec_cfg.lambda_)
        policy.update(a_e, L)

        # AGGRESSIVE counterfactual at this tick
        aggr_slip = prev["best_ask_px"] - prev["mid_price"]
        aggr_adv  = max(0.0, prev["best_ask_px"] - row["mid_price"])
        aggr_loss = aggr_slip + exec_cfg.lambda_ * aggr_adv

        tick_idx += 1
        cum_loss.append(cum_loss[-1] + L)
        cum_aggr.append(cum_aggr[-1] + aggr_loss)
        t.append(tick_idx)
        prev = row

    cum_loss = np.array(cum_loss)
    cum_aggr = np.array(cum_aggr)
    t        = np.array(t)
    savings  = cum_aggr - cum_loss

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    axes[0].plot(t, cum_aggr, "--", color=C_SPREAD, linewidth=1.4,
                 label="Always-AGGRESSIVE (línea base trivial)")
    axes[0].plot(t, cum_loss, color=C_1D, linewidth=1.6,
                 label="Política 1D bucketed")
    axes[0].set_ylabel("Pérdida acumulada (USDT)")
    axes[0].legend(loc="upper left", framealpha=0.95)
    axes[0].set_title(f"Pérdida acumulada — política vs línea base trivial — S{target_session}")

    axes[1].plot(t, savings, color=C_GAIN, linewidth=2.0)
    axes[1].fill_between(t, 0, savings, color=C_GAIN, alpha=0.25)
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_ylabel("$ ahorrados vs always-AGGR")
    axes[1].set_xlabel("Tick")
    axes[1].set_title(f"Ahorro acumulado contra la línea base — total final: ${savings[-1]:+,.2f}")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "p09_savings_vs_aggr.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 10. Loss decomposition — show the toxicity barrier
# ---------------------------------------------------------------------------

def plot_loss_decomposition():
    """Use the §25 harness output to show how each variant trades slippage
    for adverse — the toxicity-barrier visual."""
    h = _load_harness(HARNESS_OLD)
    sessions = sorted({r["session_idx"] for r in h["results"]})
    modes = ["1D_baseline", "vol_delta", "ofi_window", "spread_delta"]

    means_slip = {m: [] for m in modes}
    means_adv  = {m: [] for m in modes}

    for s in sessions:
        for m in modes:
            rows = [r for r in h["results"]
                    if r["mode"] == m and r["session_idx"] == s]
            means_slip[m].append(np.mean([r["total_slippage"] for r in rows]))
            means_adv[m].append(np.mean([r["total_adverse"]   for r in rows]))

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = {"1D_baseline": "o", "vol_delta": "s",
               "ofi_window": "^", "spread_delta": "D"}
    colors  = {"1D_baseline": C_1D,   "vol_delta": C_VOL,
               "ofi_window":  C_OFI,  "spread_delta": C_SPREAD}
    for m in modes:
        ax.scatter(means_slip[m], means_adv[m],
                   marker=markers[m], color=colors[m], s=140,
                   edgecolor="black", linewidth=0.7, zorder=3,
                   label={"1D_baseline":  "1D bucketed (baseline)",
                          "vol_delta":    "+ vol_delta",
                          "ofi_window":   "+ ofi_window",
                          "spread_delta": "+ spread_delta"}[m])
    # connect points of same session for clarity
    for i, s in enumerate(sessions):
        xs = [means_slip[m][i] for m in modes]
        ys = [means_adv[m][i] for m in modes]
        ax.plot(xs, ys, color="#999999", linewidth=0.5, alpha=0.5)
        ax.text(xs[0], ys[0], f"  S{s}", fontsize=9, color="#444444")

    ax.set_xlabel("Slippage total ($, menor = mejor)")
    ax.set_ylabel("Adverse total (unidades, menor = mejor)")
    ax.set_title("La barrera de toxicidad — ahorrar slippage cuesta más adverse\n"
                 "Cada línea conecta los 4 modos sobre la MISMA sesión. Todas suben de "
                 "izq.→der.: cualquier modo que ahorra slippage paga más adverse.")
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p10_loss_decomposition.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading harness output...")
    h_new = _load_harness(HARNESS_NEW)

    print("[1/10] p01 — marginal vs 1D bars")
    plot_marginal_vs_1d_bars(h_new)
    print("[2/10] p02 — paired diff with |t|")
    plot_paired_diff(h_new)
    print("[3/10] p03 — per-seed scatter")
    plot_scatter(h_new)
    print("[4/10] p04+p05 — Kalman + features traces")
    plot_kalman_and_features(target_session=2, n_ticks=2000, start_offset=2000)
    print("[6/10] p06 — Hedge weights evolution")
    plot_hedge_evolution(target_session=7, seed=0, n_snapshots=80)
    print("[7/10] p07 — per-bucket final weights")
    plot_per_bucket_final_weights()
    print("[8/10] p08 — action mix marginal vs 1D")
    plot_action_mix(h_new)
    print("[9/10] p09 — savings curve")
    plot_savings_curve(target_session=7, seed=0)
    print("[10/10] p10 — toxicity barrier (Pareto)")
    plot_loss_decomposition()

    print(f"\nDone — see {FIG_DIR}")


if __name__ == "__main__":
    main()
