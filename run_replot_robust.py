"""Regenerate fig-trace-robust.png as a 2x2 panel: accuracy AND latency for both
the arrival-model mismatch and the within-session drift sweeps."""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "accesstier")
mm = pd.read_csv(os.path.join(OUT, "table-mismatch.csv"))
dr = pd.read_csv(os.path.join(OUT, "table-drift.csv"))

STY = {"awf": ("AWF", "#d1495b", "-", "o"),
       "logpool": ("Full-Wait (log)", "#1b998b", "-", "s"),
       "anytime_cbci": ("Confidence-Stop", "#2e86ab", ":", "^"),
       "deadline_aware": ("Arrival-Only", "#5c4d7d", "--", "D"),
       "generic_voi": ("Entropy-Gain", "#c9a227", "-.", "v")}
order = list(STY.keys())

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})


def _panel(ax, df, key, col, ylabel, title, band_deadline=None):
    g = df.groupby([key, "method"])[col].mean().unstack()
    if col == "accuracy":
        g = g * 100
    xs = g.index.values
    for m in order:
        lab, clr, ls, mrk = STY[m]
        isa = m == "awf"
        if m == "logpool" and col == "accuracy":
            ax.plot(xs, g[m], color=clr, lw=7, alpha=0.30,
                    solid_capstyle="round", label=lab, zorder=1)
            continue
        ax.plot(xs, g[m], color=clr, ls=ls, marker=mrk,
                lw=3 if isa else 2.0, ms=8.5 if isa else 6,
                mfc=clr if isa else "white", mec=clr, mew=1.5,
                label=lab, zorder=6 if isa else 3)
    if band_deadline is not None:
        ax.axhline(band_deadline, color="0.4", ls=":", lw=1.3)
    ax.set_xlabel(xlabel_map[key], fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.tick_params(labelsize=11)


xlabel_map = {"scale": "arrival-delay mismatch factor",
              "drift": "within-session drift (last-trial extra delay)"}

fig, ax = plt.subplots(2, 2, figsize=(13.5, 10))
_panel(ax[0, 0], mm, "scale", "accuracy", "accuracy (%)", "(a) Mismatch: accuracy")
_panel(ax[0, 1], mm, "scale", "latency", "mean latency (ms)", "(b) Mismatch: latency", 50)
_panel(ax[1, 0], dr, "drift", "accuracy", "accuracy (%)", "(c) Drift: accuracy")
_panel(ax[1, 1], dr, "drift", "latency", "mean latency (ms)", "(d) Drift: latency", 50)
ax[0, 0].legend(fontsize=11, loc="lower left", frameon=False)
ax[0, 1].legend(fontsize=11, loc="upper left", frameon=False)
fig.suptitle("Robustness to arrival-model mismatch and within-session drift "
             "(accuracy and latency)", fontsize=16, y=1.0)
fig.tight_layout()
p = os.path.join(OUT, "fig-trace-robust.png")
fig.savefig(p)
plt.close(fig)
print("wrote", p)
