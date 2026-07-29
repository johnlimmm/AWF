"""Deeper trace diagnostics for the report:
  (1) Nonstationarity: first/second-half quantiles, two-sample KS, Wasserstein
      distance, block-bootstrap CI of the median shift, rolling median.
  (2) The quantity AWF actually uses: conditional residual arrival probability
      rho_j(t) = P(a <= D | a > t) and E[Delta_j(t)] vs decision time t.
Writes fig-nonstat.png, fig-rho-t.png and nonstat-stats.txt.
"""
import importlib.util
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "accesstier")
spec = importlib.util.spec_from_file_location("tb", os.path.join(HERE, "bcinet-tracebank.py"))
tb = importlib.util.module_from_spec(spec); sys.modules["tb"] = tb
spec.loader.exec_module(tb)
bank = tb.load_tracebank()

TIERS = ["ble", "wifi", "5g", "lte", "leo"]
LABEL = {"ble": "BLE", "wifi": "Wi-Fi", "5g": "5G", "lte": "4G", "leo": "LEO"}
CLR = {"ble": "#2e86ab", "wifi": "#3fa34d", "5g": "#d1495b", "lte": "#e8a33d", "leo": "#8134af"}
rng = np.random.RandomState(0)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})

seqs = {t: [np.asarray(s, float) for s in bank[t].sequences
            if np.asarray(s).size >= 8 and np.isfinite(np.asarray(s)).all()]
        for t in TIERS}
delays = {t: np.asarray(bank[t].all_delays(), float) for t in TIERS}
loss = {t: float(bank[t].overall_loss()) for t in TIERS}


def halves(t):
    a, b = [], []
    for s in seqs[t]:
        m = s.size // 2
        a.append(s[:m]); b.append(s[m:])
    return np.concatenate(a), np.concatenate(b)


out = ["=" * 78, "NONSTATIONARITY DIAGNOSTICS (within-session, ordered replay)", "=" * 78,
       "%-6s %8s %8s %8s   %8s %8s %8s   %6s %8s %14s" % (
           "tier", "p50_1", "p50_2", "d_p50", "p95_1", "p95_2", "d_p95",
           "KS", "Wass", "med_shift_CI")]
ks_w = {}
for t in TIERS:
    h1, h2 = halves(t)
    p50_1, p95_1 = np.percentile(h1, [50, 95])
    p50_2, p95_2 = np.percentile(h2, [50, 95])
    ks = stats.ks_2samp(h1, h2).statistic
    wass = stats.wasserstein_distance(h1, h2)
    ks_w[t] = (ks, wass)
    # block bootstrap CI of the median shift (resample whole sequences)
    shifts = []
    S = seqs[t]
    for _ in range(400):
        idx = rng.randint(0, len(S), len(S))
        bs = [S[i] for i in idx]
        a = np.concatenate([s[:s.size // 2] for s in bs])
        b = np.concatenate([s[s.size // 2:] for s in bs])
        shifts.append(np.median(b) - np.median(a))
    lo, hi = np.percentile(shifts, [2.5, 97.5])
    out.append("%-6s %8.1f %8.1f %8.2f   %8.1f %8.1f %8.2f   %6.3f %8.2f  [%+.2f,%+.2f]" % (
        LABEL[t], p50_1, p50_2, p50_2 - p50_1, p95_1, p95_2, p95_2 - p95_1,
        ks, wass, lo, hi))

# ---- Figure A: rolling median + KS/Wasserstein ----------------------------
figA, ax = plt.subplots(1, 2, figsize=(13, 5))
for t in ["ble", "wifi", "leo"]:
    s = max(seqs[t], key=lambda a: a.size)
    w = max(s.size // 25, 5)
    rm = np.array([np.median(s[max(0, i - w):i + w + 1]) for i in range(s.size)])
    x = np.linspace(0, 1, s.size)
    ax[0].plot(x, rm, color=CLR[t], lw=2.2, label=LABEL[t])
ax[0].set_xlabel("normalized position within session", fontsize=15)
ax[0].set_ylabel("rolling median delay (ms)", fontsize=15)
ax[0].set_title("(a) Rolling median within a session", fontsize=15)
ax[0].tick_params(labelsize=12); ax[0].legend(fontsize=12, frameon=False)

x = np.arange(len(TIERS)); wd = 0.38
ax[1].bar(x - wd / 2, [ks_w[t][0] for t in TIERS], wd, color="#4c72b0", label="KS (first vs second half)")
ax[1].bar(x + wd / 2, [ks_w[t][1] for t in TIERS], wd, color="#dd8452", label="Wasserstein (ms)")
ax[1].set_xticks(x); ax[1].set_xticklabels([LABEL[t] for t in TIERS], fontsize=12)
ax[1].set_title("(b) Distribution change between halves", fontsize=15)
ax[1].tick_params(axis="y", labelsize=12); ax[1].legend(fontsize=12, frameon=False)
figA.suptitle("Within-session nonstationarity of the access traces", fontsize=16, y=1.01)
figA.tight_layout(); figA.savefig(os.path.join(OUT, "fig-nonstat.png")); plt.close(figA)

# ---- Figure B: rho_j(t) and E[Delta_j(t)] vs t ----------------------------
D = 50.0
ts = np.linspace(0, D - 1, 60)
figB, ax = plt.subplots(1, 2, figsize=(13, 5))
out.append("")
out.append("CONDITIONAL RESIDUAL ARRIVAL rho_j(t)=P(a<=D|a>t), D=50 (selected t)")
out.append("%-6s %8s %8s %8s %8s" % ("tier", "t=0", "t=20", "t=35", "t=45"))
for t in TIERS:
    d = delays[t]
    rho, edl = [], []
    for tv in ts:
        surv = d[d > tv]
        if surv.size == 0:
            rho.append(0.0); edl.append(0.0); continue
        rho.append((1 - loss[t]) * np.mean(surv <= D))
        edl.append(np.mean(np.minimum(surv - tv, D - tv)))
    rho = np.array(rho); edl = np.array(edl)
    ax[0].plot(ts, rho, color=CLR[t], lw=2.4, label=LABEL[t])
    ax[1].plot(ts, edl, color=CLR[t], lw=2.4, label=LABEL[t])

    def at(tv):
        surv = d[d > tv]
        return (1 - loss[t]) * np.mean(surv <= D) if surv.size else 0.0
    out.append("%-6s %8.2f %8.2f %8.2f %8.2f" % (LABEL[t], at(0), at(20), at(35), at(45)))
ax[0].set_xlabel("decision time t (ms)", fontsize=15)
ax[0].set_ylabel(r"$\rho_j(t)=P(a\leq D\mid a>t)$", fontsize=15)
ax[0].set_title(r"(a) Conditional arrival probability, $D=50$", fontsize=15)
ax[0].tick_params(labelsize=12); ax[0].legend(fontsize=12, frameon=False, loc="lower left")
ax[1].set_xlabel("decision time t (ms)", fontsize=15)
ax[1].set_ylabel(r"$E[\Delta_j(t)]$ (ms)", fontsize=15)
ax[1].set_title(r"(b) Expected residual wait, $D=50$", fontsize=15)
ax[1].tick_params(labelsize=12); ax[1].legend(fontsize=12, frameon=False)
figB.suptitle(r"What AWF actually uses: residual arrival model vs decision time", fontsize=16, y=1.01)
figB.tight_layout(); figB.savefig(os.path.join(OUT, "fig-rho-t.png")); plt.close(figB)

rep = "\n".join(out)
print(rep)
with open(os.path.join(OUT, "nonstat-stats.txt"), "w", encoding="utf-8") as f:
    f.write(rep)
print("\nwrote fig-nonstat.png, fig-rho-t.png, nonstat-stats.txt")
