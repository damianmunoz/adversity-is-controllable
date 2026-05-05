"""
scripts/_plots_section25.py — figures + numeric summary for §25.

Reads data/derived/ab_adverse_3sessions_10seed.json (overwritten by the
re-run of scripts/_ab_adverse_3sessions.py with TARGET_SESSIONS = [1,2,3,7])
and produces:

  data/derived/figures/section25/
    fig01_total_loss_per_session.png
    fig02_paired_diff_vs_1d.png
    fig03_per_seed_scatter_ofi_window_vs_1d.png
    fig04_action_mix_per_session.png
    fig05_slippage_vs_adverse.png

It also prints a numeric summary block to stdout that is meant to be
pasted into sanity.txt §25.

Real numbers only — every plot and every printed statistic is computed
from the JSON. Nothing fabricated.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


JSON_PATH = Path("data/derived/ab_adverse_3sessions_10seed.json")
OUT_DIR   = Path("data/derived/figures/section25")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Modes in canonical display order
MODES = ["1D_baseline", "vol_delta", "ofi_window", "spread_delta"]
LABELS = {
    "1D_baseline":  "1D baseline",
    "vol_delta":    "A: vol_delta",
    "ofi_window":   "B: ofi_window",
    "spread_delta": "C: spread_delta",
}
COLORS = {
    "1D_baseline":  "#444444",
    "vol_delta":    "#1f77b4",
    "ofi_window":   "#2ca02c",
    "spread_delta": "#d62728",
}


def paired_t(diffs: np.ndarray) -> tuple[float, float, float]:
    """Returns (mean, sem, t-stat) for a paired-difference vector."""
    n = len(diffs)
    if n < 2:
        return float(diffs.mean()) if n else 0.0, 0.0, 0.0
    m = float(diffs.mean())
    s = float(diffs.std(ddof=1))
    sem = s / math.sqrt(n)
    t = m / sem if sem > 0 else 0.0
    return m, sem, t


def load_records(path: Path) -> tuple[list[dict], dict]:
    j = json.load(open(path))
    return j["results"], j.get("config", {})


def organize(rows: list[dict]):
    """Return nested dict[session_idx][mode][seed] = row, plus session meta."""
    by = defaultdict(lambda: defaultdict(dict))
    sess_meta = {}
    for r in rows:
        s = r["session_idx"]
        by[s][r["mode"]][r["seed"]] = r
        sess_meta[s] = (
            r["session_start_iso"][:19],
            r["session_end_iso"][:19],
            r["session_n_rows"],
            r["session_duration_h"],
        )
    return by, sess_meta


# --------------------------------------------------------------------------
# Plot 1 — mean total_loss per (session, mode) with stdev error bars
# --------------------------------------------------------------------------

def fig01(by, sess_meta, sessions):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.20
    x = np.arange(len(sessions))
    for i, mode in enumerate(MODES):
        means, stds = [], []
        for s in sessions:
            losses = np.array([by[s][mode][seed]["total_loss"] for seed in by[s][mode]])
            means.append(losses.mean()); stds.append(losses.std(ddof=1))
        ax.bar(x + (i - 1.5) * width, means, width, yerr=stds, capsize=3,
               label=LABELS[mode], color=COLORS[mode])
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}\n{sess_meta[s][3]:.1f}h\n{sess_meta[s][2]:,} ticks"
                        for s in sessions], fontsize=9)
    ax.set_ylabel("total_loss  (slippage + λ·adverse_move, λ=0.1)")
    ax.set_title("§25 — Mean total_loss per session × mode (10 seeds, error bars = stdev)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig01_total_loss_per_session.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# Plot 2 — paired diff (mode − 1D) per session, with t-stat annotations
# --------------------------------------------------------------------------

def fig02(by, sess_meta, sessions):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    adverse_modes = [m for m in MODES if m != "1D_baseline"]
    width = 0.25
    x = np.arange(len(sessions))
    for i, mode in enumerate(adverse_modes):
        diffs_means, diffs_sems, ts = [], [], []
        for s in sessions:
            seeds = sorted(set(by[s]["1D_baseline"]) & set(by[s][mode]))
            d = np.array([by[s][mode][k]["total_loss"]
                          - by[s]["1D_baseline"][k]["total_loss"] for k in seeds])
            m, sem, t = paired_t(d)
            diffs_means.append(m); diffs_sems.append(sem); ts.append(t)
        bars = ax.bar(x + (i - 1) * width, diffs_means, width,
                      yerr=diffs_sems, capsize=3,
                      label=LABELS[mode], color=COLORS[mode])
        for bar, t, m in zip(bars, ts, diffs_means):
            label = f"|t|={abs(t):.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    m + (5 if m >= 0 else -15),
                    label, ha="center", fontsize=8,
                    color="black" if abs(t) >= 2.0 else "gray")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}" for s in sessions])
    ax.set_ylabel("Paired diff vs 1D baseline (mode − 1D)\nnegative = mode WINS")
    ax.set_title("§25 — Paired diff in total_loss vs 1D baseline (10 seeds, λ=0.1)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig02_paired_diff_vs_1d.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# Plot 3 — per-seed scatter ofi_window vs 1D, one panel per session
# --------------------------------------------------------------------------

def fig03(by, sess_meta, sessions):
    fig, axes = plt.subplots(1, len(sessions),
                             figsize=(4.0 * len(sessions), 4.4), sharey=False)
    if len(sessions) == 1:
        axes = [axes]
    for ax, s in zip(axes, sessions):
        seeds = sorted(set(by[s]["1D_baseline"]) & set(by[s]["ofi_window"]))
        x = np.array([by[s]["1D_baseline"][k]["total_loss"] for k in seeds])
        y = np.array([by[s]["ofi_window"][k]["total_loss"]   for k in seeds])
        wins = int((y < x).sum())
        ax.scatter(x, y, s=60, c="#2ca02c", edgecolor="black")
        for k, xi, yi in zip(seeds, x, y):
            ax.annotate(str(k), (xi, yi), textcoords="offset points",
                        xytext=(4, 4), fontsize=7)
        lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
        pad = (hi - lo) * 0.05
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                "k--", lw=1, label="y=x")
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("1D baseline total_loss")
        ax.set_ylabel("ofi_window total_loss")
        ax.set_title(f"S{s} ({sess_meta[s][3]:.1f}h, {sess_meta[s][2]:,} ticks)\n"
                     f"ofi_window wins {wins}/{len(seeds)} seeds")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("§25 — Per-seed paired comparison: ofi_window (B) vs 1D baseline\n"
                 "dot below dashed line = ofi_window beat 1D for that seed",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "fig03_per_seed_scatter_ofi_window_vs_1d.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# Plot 4 — action mix per (session × mode), mean across seeds
# --------------------------------------------------------------------------

def fig04(by, sess_meta, sessions):
    fig, axes = plt.subplots(1, len(sessions),
                             figsize=(4.0 * len(sessions), 4.4), sharey=True)
    if len(sessions) == 1:
        axes = [axes]
    actions = ["wait_pct", "passive_pct", "aggressive_pct"]
    action_names = ["WAIT", "PASSIVE", "AGGRESSIVE"]
    width = 0.20
    for ax, s in zip(axes, sessions):
        x = np.arange(len(actions))
        for i, mode in enumerate(MODES):
            seeds = sorted(by[s][mode])
            mix = [np.mean([by[s][mode][k][a] for k in seeds]) for a in actions]
            ax.bar(x + (i - 1.5) * width, mix, width,
                   label=LABELS[mode], color=COLORS[mode])
        ax.set_xticks(x); ax.set_xticklabels(action_names)
        ax.set_title(f"S{s} ({sess_meta[s][3]:.1f}h)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("mean fraction of ticks (%)")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("§25 — Mean action mix per session × mode (10 seeds)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "fig04_action_mix_per_session.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# Plot 5 — slip vs adverse scatter (each dot = one (session × mode) mean)
# --------------------------------------------------------------------------

def fig05(by, sess_meta, sessions):
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = {1: "o", 2: "s", 3: "^", 7: "D"}
    for mode in MODES:
        for s in sessions:
            seeds = sorted(by[s][mode])
            slip = np.mean([by[s][mode][k]["total_slippage"] for k in seeds])
            adv  = np.mean([by[s][mode][k]["total_adverse"]  for k in seeds])
            ax.scatter(slip, adv, s=140, marker=markers.get(s, "o"),
                       facecolor=COLORS[mode],
                       edgecolor="black", linewidth=1,
                       label=f"{LABELS[mode]} (S{s})")
            ax.annotate(f"{LABELS[mode][:1]}-S{s}", (slip, adv),
                        textcoords="offset points", xytext=(6, 6), fontsize=7)
    ax.set_xlabel("total_slippage  ($, mean across 10 seeds)")
    ax.set_ylabel("total_adverse_move  ($, mean across 10 seeds)")
    ax.set_title("§25 — Slippage vs adverse-move trade-off per (session × mode)\n"
                 "lower-left is better on BOTH axes (Pareto)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig05_slippage_vs_adverse.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# Numeric summary printed to stdout for sanity.txt §25
# --------------------------------------------------------------------------

def summary(by, sess_meta, sessions, config):
    print()
    print("=" * 80)
    print(" §25 NUMERIC SUMMARY  (paste into sanity.txt)")
    print("=" * 80)
    print(f"  config: λ={config.get('lambda')}, η={config.get('eta')}, "
          f"pressure_edges={config.get('pressure_edges')}")
    print(f"  seeds: {config.get('seeds')}  ({len(config.get('seeds', []))} per cell)")
    print(f"  modes: {config.get('modes')}")
    print()

    # Per-session table
    for s in sessions:
        meta = sess_meta[s]
        print(f"--- Session {s}  ({meta[0]} → {meta[1]} UTC,  "
              f"{meta[2]:,} ticks,  {meta[3]:.2f}h) ---")
        seeds_b = sorted(by[s]["1D_baseline"])
        base = np.array([by[s]["1D_baseline"][k]["total_loss"] for k in seeds_b])
        print(f"  {'mode':<14} {'mean_loss':>10} {'std':>8} "
              f"{'mean_diff':>10} {'sem':>7} {'|t|':>6} {'wins':>6} "
              f"{'mean_slip$':>11} {'mean_adv$':>10} {'fill_rate':>9}")
        for mode in MODES:
            seeds = sorted(by[s][mode])
            losses = np.array([by[s][mode][k]["total_loss"] for k in seeds])
            slip   = np.array([by[s][mode][k]["total_slippage"] for k in seeds])
            adv    = np.array([by[s][mode][k]["total_adverse"]  for k in seeds])
            fr     = np.array([by[s][mode][k]["fill_rate"]     for k in seeds])
            if mode == "1D_baseline":
                m_d, sem_d, t_d, wins = 0.0, 0.0, 0.0, 0
            else:
                k_common = sorted(set(seeds_b) & set(seeds))
                d = np.array([by[s][mode][k]["total_loss"]
                              - by[s]["1D_baseline"][k]["total_loss"] for k in k_common])
                m_d, sem_d, t_d = paired_t(d)
                wins = int((d < 0).sum())
            tag = LABELS[mode].split(":")[0] if ":" in LABELS[mode] else mode
            print(f"  {tag:<14} {losses.mean():>10.2f} {losses.std(ddof=1):>8.2f} "
                  f"{m_d:>+10.2f} {sem_d:>7.2f} {abs(t_d):>6.2f} "
                  f"{wins:>3}/{len(seeds):<2} "
                  f"{slip.mean():>11.2f} {adv.mean():>10.2f} {fr.mean():>9.4f}")
        print()


def main():
    rows, config = load_records(JSON_PATH)
    by, sess_meta = organize(rows)
    sessions = sorted(by)
    print(f"Loaded {len(rows)} rows — sessions {sessions}")
    fig01(by, sess_meta, sessions)
    fig02(by, sess_meta, sessions)
    fig03(by, sess_meta, sessions)
    fig04(by, sess_meta, sessions)
    fig05(by, sess_meta, sessions)
    print(f"Wrote 5 figures → {OUT_DIR}/")
    summary(by, sess_meta, sessions, config)


if __name__ == "__main__":
    main()
