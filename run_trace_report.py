"""Trace-bank statistical characterization figures for the report.

Generates:
  fig-trace-stats.png    (2x2) delay ECDF, deadline feasibility, dispersion,
                                temporal structure
  fig-trace-robust.png   (1x2) accuracy vs arrival-model mismatch and drift
Both under results/accesstier/.  Numbers are printed for the written report.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "accesstier")


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


tb = _load("tb", "bcinet-tracebank.py")
bank = tb.load_tracebank()

TIERS = ["ble", "wifi", "5g", "lte", "leo"]
LABEL = {"ble": "BLE", "wifi": "Wi-Fi", "5g": "5G", "lte": "4G", "leo": "LEO"}
CLR = {"ble": "#2e86ab", "wifi": "#3fa34d", "5g": "#d1495b",
       "lte": "#e8a33d", "leo": "#8134af"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})
AX_FS, TK_FS, LG_FS, TT_FS = 16, 13, 13, 16

delays = {t: np.asarray(bank[t].all_delays(), float) for t in TIERS}
loss = {t: bank[t].overall_loss() for t in TIERS}


def lag1(x):
    x = np.asarray(x, float)
    if x.size < 3:
        return np.nan
    x = x - x.mean()
    d = np.dot(x, x)
    return float(np.dot(x[:-1], x[1:]) / d) if d > 0 else np.nan


# per-tier scalar stats
cv = {t: delays[t].std() / delays[t].mean() for t in TIERS}
tailr = {t: np.percentile(delays[t], 99) / np.percentile(delays[t], 50)
         for t in TIERS}
acf1, drift = {}, {}
for t in TIERS:
    seqs = [np.asarray(s, float) for s in bank[t].sequences
            if np.asarray(s).size >= 6]
    if seqs:
        acf1[t] = np.nanmean([lag1(s) for s in seqs])
        dd = []
        for s in seqs:
            h = s.size // 2
            b = s.mean()
            dd.append(abs(s[h:].mean() - s[:h].mean()) / b * 100 if b > 0 else np.nan)
        drift[t] = np.nanmedian(dd)
    else:
        acf1[t], drift[t] = np.nan, np.nan

# ===========================================================================
# Figure 1: trace characterization (2x2)
# ===========================================================================
fig, ax = plt.subplots(2, 2, figsize=(13, 10))

# (a) delay ECDF (log-x) with deadline markers
a = ax[0, 0]
for t in TIERS:
    d = np.sort(delays[t])
    y = np.linspace(0, 1, d.size)
    a.plot(d, y, color=CLR[t], lw=2.4, label=LABEL[t])
for D in (50, 100):
    a.axvline(D, color="0.4", ls=":", lw=1.5)
    a.text(D, 0.03, "D=%d" % D, fontsize=11, color="0.3", rotation=90,
           ha="right", va="bottom")
a.set_xscale("log")
a.set_xlim(3, 300)
a.set_xlabel("one-way delay (ms, log)", fontsize=AX_FS)
a.set_ylabel("empirical CDF", fontsize=AX_FS)
a.set_title("(a) Per-tier delay distribution", fontsize=TT_FS)
a.tick_params(labelsize=TK_FS)
a.legend(fontsize=LG_FS, loc="lower right", frameon=False)

# (b) deadline feasibility
b = ax[0, 1]
Ds = np.arange(5, 101, 2.5)
for t in TIERS:
    p = [(delays[t] <= D).mean() * (1 - loss[t]) for D in Ds]
    b.plot(Ds, p, color=CLR[t], lw=2.4, label=LABEL[t])
b.axvspan(50, 100, color="0.85", alpha=0.35, zorder=0)
b.set_xlabel("deadline D (ms)", fontsize=AX_FS)
b.set_ylabel(r"P(arrive $\leq$ D)", fontsize=AX_FS)
b.set_title("(b) Deadline feasibility (arrival is binding)", fontsize=TT_FS)
b.tick_params(labelsize=TK_FS)
b.set_ylim(-0.02, 1.02)
b.legend(fontsize=LG_FS, loc="lower right", frameon=False)

# (c) dispersion: CV and tail ratio grouped bars
c = ax[1, 0]
x = np.arange(len(TIERS))
w = 0.38
c.bar(x - w / 2, [cv[t] for t in TIERS], w, color="#4c72b0", label="CV (std/mean)")
c.bar(x + w / 2, [tailr[t] for t in TIERS], w, color="#dd8452",
      label="tail ratio p99/p50")
c.axhline(1.0, color="0.5", ls="--", lw=1)
c.set_xticks(x)
c.set_xticklabels([LABEL[t] for t in TIERS], fontsize=TK_FS)
c.set_ylabel("dispersion / heavy-tail", fontsize=AX_FS)
c.set_title("(c) Variability and tail heaviness", fontsize=TT_FS)
c.tick_params(axis="y", labelsize=TK_FS)
c.legend(fontsize=LG_FS, frameon=False)
for i, t in enumerate(TIERS):
    c.text(i - w / 2, cv[t] + 0.05, "%.2f" % cv[t], ha="center", fontsize=10)
    c.text(i + w / 2, tailr[t] + 0.05, "%.1f" % tailr[t], ha="center", fontsize=10)

# (d) temporal structure: lag-1 acf and within-source drift
d = ax[1, 1]
c1 = "#55a868"
c2 = "#c44e52"
d.bar(x - w / 2, [acf1[t] for t in TIERS], w, color=c1, label="lag-1 autocorr.")
d2 = d.twinx()
d2.bar(x + w / 2, [drift[t] for t in TIERS], w, color=c2,
       label="split-half drift (%)")
d.axhline(0.0, color="0.5", ls="--", lw=1)
d.set_xticks(x)
d.set_xticklabels([LABEL[t] for t in TIERS], fontsize=TK_FS)
d.set_ylabel("lag-1 autocorrelation", fontsize=AX_FS - 1, color=c1)
d2.set_ylabel("split-half drift (%)", fontsize=AX_FS - 1, color=c2)
d.set_title("(d) Temporal structure (memory / drift)", fontsize=TT_FS)
d.tick_params(axis="y", labelsize=TK_FS, colors=c1)
d2.tick_params(axis="y", labelsize=TK_FS, colors=c2)
d.tick_params(axis="x", labelsize=TK_FS)
d.set_ylim(-0.05, 0.30)
d2.set_ylim(0, 18)
d.grid(False)
d2.grid(False)

fig.suptitle("Access-tier trace bank: statistical characterization",
             fontsize=19, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
p1 = os.path.join(OUT, "fig-trace-stats.png")
fig.savefig(p1)
plt.close(fig)
print("wrote", p1)

# ===========================================================================
# Figure 2: robustness to mismatch / drift
# ===========================================================================
MM = {"awf": ("AWF", "#d1495b", "-", "o"),
      "logpool": ("Full-Wait (log)", "#1b998b", "-", "s"),
      "anytime_cbci": ("Confidence-Stop", "#2e86ab", ":", "^"),
      "deadline_aware": ("Arrival-Only", "#5c4d7d", "--", "D"),
      "generic_voi": ("Entropy-Gain", "#c9a227", "-.", "v")}

mm = pd.read_csv(os.path.join(OUT, "table-mismatch.csv"))
dr = pd.read_csv(os.path.join(OUT, "table-drift.csv"))

fig2, ax2 = plt.subplots(1, 2, figsize=(13.5, 5.4))

ga = mm.groupby(["scale", "method"])["accuracy"].mean().unstack() * 100
for mk, (lab, clr, ls, mrk) in MM.items():
    is_awf = mk == "awf"
    if mk == "logpool":
        ax2[0].plot(ga.index, ga[mk], color=clr, lw=7, alpha=0.30,
                    solid_capstyle="round", label=lab, zorder=1)
        continue
    ax2[0].plot(ga.index, ga[mk], color=clr, ls=ls, marker=mrk,
                lw=3 if is_awf else 2.1, ms=9 if is_awf else 6.5,
                mfc=clr if is_awf else "white", mec=clr, mew=1.6,
                label=lab, zorder=6 if is_awf else 3)
ax2[0].axvline(1.0, color="0.5", ls=":", lw=1.4)
ax2[0].set_xlabel("arrival-delay mismatch factor", fontsize=AX_FS)
ax2[0].set_ylabel("accuracy (%)", fontsize=AX_FS)
ax2[0].set_title("(a) Arrival-model mismatch", fontsize=TT_FS)
ax2[0].tick_params(labelsize=TK_FS)
ax2[0].legend(fontsize=12, loc="lower left", frameon=False)

gd = dr.groupby(["drift", "method"])["accuracy"].mean().unstack() * 100
for mk, (lab, clr, ls, mrk) in MM.items():
    is_awf = mk == "awf"
    if mk == "logpool":
        ax2[1].plot(gd.index, gd[mk], color=clr, lw=7, alpha=0.30,
                    solid_capstyle="round", label=lab, zorder=1)
        continue
    ax2[1].plot(gd.index, gd[mk], color=clr, ls=ls, marker=mrk,
                lw=3 if is_awf else 2.1, ms=9 if is_awf else 6.5,
                mfc=clr if is_awf else "white", mec=clr, mew=1.6,
                label=lab, zorder=6 if is_awf else 3)
ax2[1].set_xlabel("within-session drift (last-trial extra delay)", fontsize=AX_FS)
ax2[1].set_ylabel("accuracy (%)", fontsize=AX_FS)
ax2[1].set_title("(b) Non-stationary drift", fontsize=TT_FS)
ax2[1].tick_params(labelsize=TK_FS)
ax2[1].legend(fontsize=12, loc="lower left", frameon=False)

fig2.suptitle("Robustness when AWF's arrival belief is wrong "
              "(AWF tracks the full-wait ceiling)", fontsize=17, y=1.02)
fig2.tight_layout()
p2 = os.path.join(OUT, "fig-trace-robust.png")
fig2.savefig(p2)
plt.close(fig2)
print("wrote", p2)

# ---- print the numbers the written report cites --------------------------
print("\n--- per-tier scalar stats ---")
for t in TIERS:
    print("%-5s CV=%.2f tail=%.2f acf1=%.3f drift=%.1f%% loss=%.2f%%" % (
        LABEL[t], cv[t], tailr[t], acf1[t], drift[t], loss[t] * 100))
print("\n--- AWF vs full-wait ceiling gap ---")
for s in sorted(mm["scale"].unique()):
    g = ga.loc[s]
    print("scale %.2f: AWF=%.2f ceiling=%.2f gap=%+.2f" % (
        s, g["awf"], g["logpool"], g["awf"] - g["logpool"]))
for gval in sorted(dr["drift"].unique()):
    g = gd.loc[gval]
    print("drift %.2f: AWF=%.2f ceiling=%.2f gap=%+.2f" % (
        gval, g["awf"], g["logpool"], g["awf"] - g["logpool"]))
