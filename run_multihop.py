"""Multi-hop robustness: AWF believes only the single access-hop delay, but the
realised arrival traverses extra unmodeled network hops (internet / multi-hop
path) that add shifted, heavy-tailed transit delay.  Shows AWF still tracks the
full-wait ceiling, i.e. the policy does not require a single-hop path.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "accesstier")


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


R = _load("runexp", "run-bcinet-experiment.py")
metrics = R.bcinet_metrics
DATASETS = list(R.DATASETS)
SEEDS = 8
PRESET = "accesstier"
D = 50.0
METHODS = {"awf": "AWF", "logpool": "Full-Wait (log)",
           "anytime_cbci": "Confidence-Stop", "deadline_aware": "Arrival-Only",
           "generic_voi": "Entropy-Gain"}
HOPS = [0, 1, 2, 3, 4]

users_by_ds = {ds: R._ensure_users(ds) for ds in DATASETS}

rows = []
eff = []  # effective end-to-end delay stats per hop count
for H in HOPS:
    for ds, users in users_by_ds.items():
        for seed in range(SEEDS):
            for mk in METHODS:
                res, arr, y, nu = R.run_one(
                    mk, seed, ds, PRESET, users, deadline_ms=D,
                    alpha=1e-3, lambda_delay=1e-3, extra_hops=H)
                lat = np.array([d.decision_time_ms for d in res], float)
                rows.append(dict(hops=H, dataset=ds, seed=seed, method=mk,
                                 accuracy=metrics.accuracy(res, y),
                                 latency=float(lat.mean())))
                if mk == "awf":
                    a = arr[np.isfinite(arr)]
                    eff.append(dict(hops=H, dataset=ds, seed=seed,
                                    p50=float(np.percentile(a, 50)),
                                    p99=float(np.percentile(a, 99)),
                                    cv=float(a.std() / a.mean())))
    print("hops=%d done" % H, flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "table-multihop.csv"), index=False)
edf = pd.DataFrame(eff).groupby("hops").mean(numeric_only=True)

pa = (df.groupby(["hops", "method"])["accuracy"].mean().unstack() * 100)
pl = df.groupby(["hops", "method"])["latency"].mean().unstack()
order = list(METHODS.keys())

lines = ["=" * 78,
         "MULTI-HOP ROBUSTNESS (AWF believes single access hop; reality adds",
         "extra unmodeled hops).  D=50, %d seeds, 3 tasks." % SEEDS,
         "=" * 78, "",
         "effective end-to-end one-way delay seen by responses:"]
for H in HOPS:
    lines.append("  hops=%d : p50=%.1f ms  p99=%.1f ms  CV=%.2f" % (
        H, edf.loc[H, "p50"], edf.loc[H, "p99"], edf.loc[H, "cv"]))
lines += ["", "accuracy (%)",
          "%-6s " % "hops" + " ".join("%-16s" % METHODS[m] for m in order)]
for H in HOPS:
    lines.append("%-6d " % H + " ".join("%-16.2f" % pa.loc[H, m] for m in order))
lines += ["", "mean latency (ms)",
          "%-6s " % "hops" + " ".join("%-16s" % METHODS[m] for m in order)]
for H in HOPS:
    lines.append("%-6d " % H + " ".join("%-16.2f" % pl.loc[H, m] for m in order))
lines += ["", "AWF vs Full-Wait(log) ceiling"]
for H in HOPS:
    lines.append("  hops=%d : AWF=%.2f ceiling=%.2f gap=%+.2f" % (
        H, pa.loc[H, "awf"], pa.loc[H, "logpool"],
        pa.loc[H, "awf"] - pa.loc[H, "logpool"]))
rep = "\n".join(lines)
print("\n" + rep)
with open(os.path.join(OUT, "multihop-summary.txt"), "w", encoding="utf-8") as f:
    f.write(rep)

# figure
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200,
                     "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})
STY = {"awf": ("#d1495b", "-", "o"), "logpool": ("#1b998b", "-", "s"),
       "anytime_cbci": ("#2e86ab", ":", "^"),
       "deadline_aware": ("#5c4d7d", "--", "D"),
       "generic_voi": ("#c9a227", "-.", "v")}
fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
for m in order:
    clr, ls, mrk = STY[m]
    isa = m == "awf"
    if m == "logpool":
        ax[0].plot(HOPS, [pa.loc[H, m] for H in HOPS], color=clr, lw=7,
                   alpha=0.30, solid_capstyle="round", label=METHODS[m], zorder=1)
        continue
    ax[0].plot(HOPS, [pa.loc[H, m] for H in HOPS], color=clr, ls=ls, marker=mrk,
               lw=3 if isa else 2.1, ms=10 if isa else 7,
               mfc=clr if isa else "white", mec=clr, mew=1.6,
               label=METHODS[m], zorder=6 if isa else 3)
ax[0].set_xlabel("number of unmodeled extra hops", fontsize=16)
ax[0].set_ylabel("accuracy (%)", fontsize=16)
ax[0].set_title("(a) Accuracy vs multi-hop transit", fontsize=16)
ax[0].set_xticks(HOPS)
ax[0].tick_params(labelsize=13)
ax[0].legend(fontsize=12, loc="lower left", frameon=False)

for m in order:
    clr, ls, mrk = STY[m]
    isa = m == "awf"
    ax[1].plot(HOPS, [pl.loc[H, m] for H in HOPS], color=clr, ls=ls, marker=mrk,
               lw=3 if isa else 2.1, ms=10 if isa else 7,
               mfc=clr if isa else "white", mec=clr, mew=1.6,
               label=METHODS[m], zorder=6 if isa else 3)
ax[1].axhline(D, color="0.4", ls=":", lw=1.4)
ax[1].set_xlabel("number of unmodeled extra hops", fontsize=16)
ax[1].set_ylabel("mean decision latency (ms)", fontsize=16)
ax[1].set_title("(b) Decision latency vs multi-hop transit", fontsize=16)
ax[1].set_xticks(HOPS)
ax[1].tick_params(labelsize=13)
ax[1].legend(fontsize=12, loc="upper left", frameon=False)

fig.suptitle("Multi-hop robustness: AWF models only the access hop yet tracks "
             "the full-wait ceiling", fontsize=16, y=1.01)
fig.tight_layout()
p = os.path.join(OUT, "fig-multihop.png")
fig.savefig(p)
plt.close(fig)
print("\nwrote table-multihop.csv, multihop-summary.txt,", p)
