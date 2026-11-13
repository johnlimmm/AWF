"""
plot-accesstier.py
==================

Figures for the 4-tier access-tier trace-driven experiments
(Wi-Fi / 5G / 4G-LTE / LEO), consuming the CSVs written by
``run-accesstier.py`` under ``results/accesstier/``.

Generated figures (written next to the CSVs):
  fig-at-tiers.png      per-tier measured one-way delay distribution (trace bank)
  fig-at-main.png       accuracy vs decision-latency, all methods (D=50)
  fig-at-frontier.png   AWF accuracy-latency frontier (lambda sweep) vs baselines
  fig-at-rho.png        arrival-belief ablation (linear/tier/state/oracle)
  fig-at-ntn.png        LEO-fraction (NTN) robustness
  fig-at-deadline.png   decision-deadline sensitivity
  fig-at-mobility.png   access-switching dynamics (fixed/iid/markov/replay)

Run:  C:\\python\\python.exe plot-accesstier.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
AT = os.path.join(HERE, "results", "accesstier")

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11,
})

TIER_ORDER = ["wifi", "5g", "lte", "leo"]
TIER_LABEL = {"wifi": "Wi-Fi", "5g": "5G", "lte": "4G-LTE",
              "leo": "LEO"}
METHOD_LABEL = {"weighted": "Full-Wait (linear)", "fastest_only": "First-Arrival",
                "earliest_k:3": "First-k", "anytime_cbci": "Confidence-Stop",
                "anytime_cbci_cw": "Anytime-cBCI", "da_os": "Throughput-Stop",
                "deadline_partial": "DeadlinePartial", "awf": "AWF",
                "oracle": "Oracle", "reliability_weighted": "Rel-Weighted",
                "neurobci_sc": "Full-wait SC", "deadline_aware": "Arrival-Only",
                "generic_voi": "Entropy-Gain", "logpool": "Full-Wait"}
CLR = {"awf": "#d1495b", "anytime_cbci": "#e8a33d", "earliest_k:3": "#3fa34d",
       "earliest_k": "#3fa34d",
       "weighted": "#6c757d", "fastest_only": "#e8a33d", "oracle": "#8134af",
       "neurobci_sc": "#1b998b", "deadline_aware": "#5c4d7d",
       "generic_voi": "#c9a227", "logpool": "#1b998b",
       "anytime_cbci_cw": "#2e86ab", "da_os": "#3b429f"}

#: Canonical baseline roster shared by every multi-method comparison figure so
#: the same set of methods appears consistently across the paper.
BASELINE_ORDER = ["earliest_k:3", "anytime_cbci_cw", "da_os", "anytime_cbci",
                  "deadline_aware", "generic_voi", "logpool", "awf"]


def _lbl(m):
    return METHOD_LABEL.get(m, m)


def _load(name):
    return pd.read_csv(os.path.join(AT, name))


def _save(fig, name):
    path = os.path.join(AT, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# 0. Trace-bank per-tier delay distribution
# ---------------------------------------------------------------------------

def fig_tiers():
    spec = importlib.util.spec_from_file_location(
        "bcinet_tracebank", os.path.join(HERE, "bcinet-tracebank.py"))
    tb = importlib.util.module_from_spec(spec)
    sys.modules["bcinet_tracebank"] = tb
    spec.loader.exec_module(tb)
    bank = tb.load_tracebank()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    # (a) delay ECDF per tier
    for t in TIER_ORDER:
        d = np.sort(np.asarray(bank[t].all_delays(), dtype=float))
        d = d[d < np.percentile(d, 99)]
        y = np.linspace(0, 1, len(d))
        ax1.plot(d, y, label=TIER_LABEL[t], lw=2)
    ax1.set_xlabel("measured one-way delay (ms)")
    ax1.set_ylabel("empirical CDF")
    ax1.set_title("(a) Per-tier arrival-delay distribution")
    ax1.set_xlim(0, 120)
    ax1.legend(frameon=False)
    # (b) overall loss per tier
    loss = [bank[t].overall_loss() for t in TIER_ORDER]
    ax2.bar([TIER_LABEL[t] for t in TIER_ORDER], loss,
            color=["#2e86ab", "#3fa34d", "#d1495b", "#e8a33d", "#8134af"])
    ax2.set_ylabel("measured packet-loss probability")
    ax2.set_title("(b) Per-tier loss")
    for i, v in enumerate(loss):
        ax2.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    fig.suptitle("Access-tier trace bank (Wi-Fi / 5G / 4G-LTE / LEO)")
    _save(fig, "fig-at-tiers.png")


# ---------------------------------------------------------------------------
# 1. Main accuracy vs latency (D=50)
# ---------------------------------------------------------------------------

def fig_main():
    d = _load("table-main-d50.csv")
    order = list(BASELINE_ORDER)
    g = (d[d.method.isin(order)].groupby("method")
         .agg(acc=("accuracy", "mean"), lat=("decision_latency", "mean")))
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for m in order:
        if m not in g.index:
            continue
        is_awf = (m == "awf")
        ax.scatter(g.loc[m, "lat"], g.loc[m, "acc"],
                   s=430 if is_awf else 260,
                   color=CLR.get(m, "#444"),
                   marker="*" if is_awf else "o",
                   zorder=4 if is_awf else 3,
                   edgecolor="k", linewidth=0.9, label=_lbl(m))
    ax.set_xlabel("mean decision latency (ms)")
    ax.set_ylabel("accuracy")
    ax.set_title("Main: accuracy vs latency (D=50\u2009ms, 5-tier, 3 datasets)")
    ax.legend(frameon=False, fontsize=10, loc="lower right",
              handletextpad=0.2, borderpad=0.3)
    _save(fig, "fig-at-main.png")


# ---------------------------------------------------------------------------
# 2. Frontier (lambda sweep) on hard datasets
# ---------------------------------------------------------------------------

def fig_frontier():
    f = _load("table-frontier.csv")
    # methods whose control knob is swept -> draw one operating-point curve
    swept = [("earliest_k", "First-k", "k", (0, (1, 1))),
             ("deadline_aware", "Arrival-Only", "tau", "--"),
             ("anytime_cbci", "Confidence-Stop", "theta", (0, (5, 1))),
             ("anytime_cbci_cw", "Anytime-cBCI", "theta", ":"),
             ("da_os", "Throughput-Stop", "beta", (0, (3, 1, 1, 1))),
             ("generic_voi", "Entropy-Gain", "lambda", "-."),
             ("awf", "AWF", "lambda", "-")]

    def _rng(sym, v_a, v_b):
        if sym == "k":
            return r"$k\!\in\![%d,%d]$" % (int(v_a), int(v_b))
        if sym == "lambda":
            if v_a <= 0:
                return r"$\lambda\!\in\![0,10^{%d}]$" % int(round(np.log10(v_b)))
            return r"$\lambda\!\in\![10^{%d},10^{%d}]$" % (
                int(round(np.log10(v_a))), int(round(np.log10(v_b))))
        if sym == "beta":
            if v_a <= 0:
                return r"$\beta\!\in\![0,10^{%d}]$" % int(round(np.log10(v_b)))
            return r"$\beta\!\in\![10^{%d},10^{%d}]$" % (
                int(round(np.log10(v_a))), int(round(np.log10(v_b))))
        if sym == "theta":
            return r"$\theta\!\in\![%g,%g]$" % (v_a, v_b)
        return r"$\tau\!\in\![%g,%g]$" % (v_a, v_b)

    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    for m, base, sym, ls in swept:
        b = f[f.method == m]
        if b.empty:
            continue
        if m == "earliest_k":
            # Committees have N=4,9,10.  Keep only k values represented by
            # every dataset in this cross-dataset average; larger k values
            # remain in the CSV for per-dataset analysis but must not be mixed
            # into a curve whose dataset composition changes with k.
            coverage = b.groupby("sweep")["dataset"].nunique()
            common = coverage[coverage == b["dataset"].nunique()].index
            b = b[b["sweep"].isin(common)]
        # 3-dataset average per knob value
        g = (b.groupby("sweep")
             .agg(acc=("accuracy", "mean"), lat=("latency_mean", "mean"))
             .sort_values("lat"))
        is_awf = (m == "awf")
        clr = CLR.get(m, "#444")
        vals = sorted(g.index.tolist())
        lab = "%s, %s" % (base, _rng(sym, vals[0], vals[-1]))
        ax.plot(g["lat"], g["acc"] * 100.0, marker="o", linestyle=ls,
                color=clr,
                lw=3.4 if is_awf else 2.4, ms=9 if is_awf else 7,
                mfc=clr if is_awf else "white",
                mec=clr, mew=1.8,
                label=lab, zorder=6 if is_awf else 3)
    ax.set_xlabel("mean decision latency (ms)", fontsize=25)
    ax.set_ylabel("accuracy (%)", fontsize=25)
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(alpha=0.22)
    ax.margins(x=0.08, y=0.10)
    ax.legend(frameon=True, framealpha=0.92, edgecolor="0.8",
              fontsize=16, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig-at-frontier.png")


# ---------------------------------------------------------------------------
# 3. rho-belief ablation
# ---------------------------------------------------------------------------

def fig_rho():
    r = _load("table-rho-ablation.csv")
    order = ["accesstier__belief-linear", "accesstier__belief-tier",
             "accesstier__belief-state", "accesstier__belief-oracle"]
    lab = ["Linear\n(parametric)", "Tier-CDF", "Tier+State\nCDF", "Oracle"]
    g = r.groupby("preset")["accuracy"].agg(["mean", "std"]).reindex(order)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    bars = ax.bar(lab, g["mean"],
                  yerr=g["std"] / np.sqrt(r.groupby("preset").size().reindex(order)),
                  color=["#6c757d", "#2e86ab", "#d1495b", "#8134af"], capsize=4)
    ax.set_ylabel("AWF accuracy")
    ax.set_ylim(g["mean"].min() - 0.03, g["mean"].max() + 0.02)
    ax.set_title("Arrival-probability belief ablation (D=50, 3 datasets)")
    for b, v in zip(bars, g["mean"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}",
                ha="center", fontsize=9)
    _save(fig, "fig-at-rho.png")


# ---------------------------------------------------------------------------
# 4. NTN robustness (LEO fraction)
# ---------------------------------------------------------------------------

def fig_ntn():
    n = _load("table-ntn.csv")
    order = ["accesstier_leo00", "accesstier_leo10", "accesstier",
             "accesstier_leo30"]
    xfrac = [0, 10, 20, 30]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for m in BASELINE_ORDER:
        y = [n[(n.preset == p) & (n.method == m)]["accuracy"].mean()
             for p in order]
        is_awf = (m == "awf")
        ax.plot(xfrac, y, "-o", color=CLR.get(m, "#444"),
                lw=2.6 if is_awf else 1.8, ms=8 if is_awf else 6,
                zorder=6 if is_awf else 3, label=_lbl(m))
    ax.set_xlabel("LEO (non-terrestrial) user fraction (%)")
    ax.set_ylabel("accuracy")
    ax.set_title("NTN robustness: accuracy vs LEO fraction (D=50)")
    ax.legend(frameon=False)
    _save(fig, "fig-at-ntn.png")


# ---------------------------------------------------------------------------
# 5. Deadline sensitivity (method comparison, split into two figures)
# ---------------------------------------------------------------------------

_DEADLINE_METHODS = ["anytime_cbci", "deadline_aware", "logpool", "awf"]

#: Per-method linestyle (round markers everywhere; colour + linestyle keep
#: overlapping deadline curves distinguishable without odd marker shapes).
_DL_STYLE = {
    "anytime_cbci":    ("o", "--"),
    "deadline_aware":  ("o", "-."),
    "logpool":         ("o", "-"),
    "awf":             ("o", "-"),
}


def _deadline_metric_fig(col, ylabel, fname, legend_loc, scale=1.0):
    d = _load("table-deadline.csv")
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    ys = []
    for m in _DEADLINE_METHODS:
        g = d[d.method == m].groupby("deadline_ms")[col].mean()
        marker, ls = _DL_STYLE.get(m, ("o", "-"))
        is_awf = (m == "awf")
        yv = g.values * scale
        ys.append(yv)
        if m == "logpool":
            # Full-wait is the full-evidence reference: draw it as a wide
            # translucent band so AWF, which nearly coincides in accuracy,
            # visibly rides on top of it instead of being hidden behind it.
            ax.plot(g.index.values, yv, linestyle="-", color=CLR.get(m, "#444"),
                    lw=7.5, alpha=0.30, solid_capstyle="round",
                    label=_lbl(m), zorder=1)
            continue
        ax.plot(g.index.values, yv, marker=marker, linestyle=ls,
                color=CLR.get(m, "#444"),
                lw=3.0 if is_awf else 2.2, ms=8.5 if is_awf else 7,
                mfc=CLR.get(m, "#444") if is_awf else "white",
                mec=CLR.get(m, "#444"), mew=1.8,
                label=_lbl(m), zorder=6 if is_awf else 3)
    # Head-room so near-coincident curves near the top are not clipped.
    lo = min(v.min() for v in ys)
    hi = max(v.max() for v in ys)
    pad = 0.10 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(r"deadline $D$ (ms)", fontsize=_AX_LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=_AX_LABEL_FS)
    ax.tick_params(axis="both", labelsize=_TICK_FS)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=15, loc=legend_loc)
    fig.tight_layout()
    _save(fig, fname)


def fig_deadline_acc():
    _deadline_metric_fig("accuracy", "accuracy (%)", "fig-at-deadline-acc.png",
                         "lower right", scale=100.0)


def fig_deadline_lat():
    _deadline_metric_fig("decision_latency", "decision latency (ms)",
                         "fig-at-deadline-lat.png", "upper left")


# ---------------------------------------------------------------------------
# 6. Mobility / access-switching dynamics
# ---------------------------------------------------------------------------

def fig_mobility():
    m = _load("table-mobility.csv")
    modes = ["fixed", "iid", "markov", "replay"]
    mlab = ["Fixed", "i.i.d.", "Markov", "Trace-replay"]
    methods = list(BASELINE_ORDER)
    x = np.arange(len(modes))
    n = len(methods)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    vals = []
    for i, meth in enumerate(methods):
        y = [m[(m.mobility == mo) & (m.method == meth)]["accuracy"].mean()
             for mo in modes]
        vals.extend(y)
        ax.bar(x + (i - (n - 1) / 2) * w, y, w, label=_lbl(meth),
               color=CLR.get(meth, "#444"))
    ax.set_xticks(x); ax.set_xticklabels(mlab)
    ax.set_ylabel("accuracy")
    lo, hi = min(vals), max(vals)
    ax.set_ylim(max(0.0, lo - 0.04), hi + 0.02)
    ax.set_title("Access-switching dynamics (D=50, 3 datasets)")
    ax.legend(frameon=False, ncol=4, fontsize=9, loc="lower center")
    _save(fig, "fig-at-mobility.png")


# ---------------------------------------------------------------------------
# 7. Margin-threshold sensitivity (alpha / gamma sweep)
# ---------------------------------------------------------------------------

#: hyperparameter-sensitivity palette.
_SWEEP_CLR = "#1f4e79"  # unified sweep-line color
# Saturated single-hue navy gradient.  The range stays visually cohesive but
# has enough contrast to avoid the washed-out look of a pale gradient.
_SWEEP_BAR_COLORS = ["#8FAECC", "#7194B8", "#537AA3", "#365F8C", "#1F456E"]
_AX_LABEL_FS = 22      # axis-label font size (unified across sweep figures)
_TICK_FS = 17          # tick-label font size


def _sweep_metric_fig(x, y, xlabel, ylabel, marker, fname, logx=True,
                      figsize=(7.8, 5.8), label_fs=None):
    """Single-axis sweep figure for one metric.
    Unified format shared by the alpha and lambda sensitivity panels."""
    if label_fs is None:
        label_fs = _AX_LABEL_FS
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, marker, color=_SWEEP_CLR, lw=2.7, ms=7, zorder=4)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel(ylabel, fontsize=label_fs)
    ax.tick_params(axis="x", labelsize=_TICK_FS)
    ax.tick_params(axis="y", labelsize=_TICK_FS)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    _save(fig, fname)


def _sweep_pair(g, xlabel, stem):
    """Emit two single-axis figures (accuracy + latency) for a sweep table."""
    x = g.index.values
    _sweep_metric_fig(x, g["acc"].values * 100.0,
                      xlabel, "accuracy (%)", "-o",
                      "fig-at-%s-acc.png" % stem, logx=True,
                      figsize=(7.8, 3.9), label_fs=18)
    _sweep_metric_fig(x, g["lat"].values,
                      xlabel, "decision latency (ms)", "-s",
                      "fig-at-%s-lat.png" % stem, logx=True,
                      figsize=(7.8, 3.9), label_fs=18)


def _sweep_bar_metric_fig(x, y, xlabel, ylabel, fname,
                          ylim_pad=(0.10, 0.18)):
    """Clean categorical bar chart shared by alpha and lambda sweeps."""
    labels = [r"$10^{%d}$" % int(np.round(np.log10(v))) for v in x]
    positions = np.arange(len(x))
    colors = [_SWEEP_BAR_COLORS[i % len(_SWEEP_BAR_COLORS)]
              for i in range(len(x))]

    fig, ax = plt.subplots(figsize=(7.8, 3.9))
    ax.bar(positions, y, width=0.68, color=colors, hatch="//",
           edgecolor="#18364F", linewidth=1.15, zorder=3)

    low = float(np.min(y))
    high = float(np.max(y))
    span = max(high - low, 0.25)
    ax.set_ylim(low - ylim_pad[0] * span, high + ylim_pad[1] * span)
    ax.set_xticks(positions, labels)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=_TICK_FS)
    ax.grid(axis="y", alpha=0.22, zorder=0)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, fname)


def fig_alpha():
    a = _load("table-alpha.csv")
    ga = a.groupby("alpha").agg(
        acc=("accuracy", "mean"), lat=("latency_mean", "mean"))
    x = ga.index.to_numpy(dtype=float)
    _sweep_bar_metric_fig(x, ga["acc"].to_numpy() * 100.0, r"$\alpha$",
                          "accuracy (%)", "fig-at-alpha-acc.png")
    _sweep_bar_metric_fig(x, ga["lat"].to_numpy(), r"$\alpha$",
                          "decision latency (ms)", "fig-at-alpha-lat.png")


def fig_lambda():
    f = _load("table-lambda.csv")
    gl = f.groupby("lambda").agg(
        acc=("accuracy", "mean"), lat=("latency_mean", "mean"))
    x = gl.index.to_numpy(dtype=float)
    _sweep_bar_metric_fig(x, gl["acc"].to_numpy() * 100.0, r"$\lambda$",
                          "accuracy (%)", "fig-at-lambda-acc.png")
    _sweep_bar_metric_fig(x, gl["lat"].to_numpy(), r"$\lambda$",
                          "decision latency (ms)", "fig-at-lambda-lat.png")


def main():
    fig_tiers()
    fig_main()
    fig_frontier()
    fig_rho()
    fig_ntn()
    fig_deadline_acc()
    fig_deadline_lat()
    fig_mobility()
    fig_alpha()
    fig_lambda()
    print("all access-tier figures done")


if __name__ == "__main__":
    main()
