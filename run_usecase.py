"""Deployment use-case robustness: does AWF work across realistic networked-cBCI
deployments (not synthetic parameter sweeps)?

Personas are realistic access compositions:
  clinic : co-located rehab/lab on BLE/Wi-Fi
  metro  : metropolitan Wi-Fi/5G edge
  wide   : wide-area 5G/4G cellular
  field  : remote/disaster LEO-satellite + cellular fallback
  hetero : fully mixed 5-tier (default)
Runs AWF + key baselines over 3 tasks (datasets) x 15 seeds at D=50.
Writes table-usecase.csv, usecase-summary.txt, fig-usecase.png.
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
SEEDS = 15
D = 50.0
PERSONAS = [
    ("accesstier_clinic", "Co-located clinic"),
    ("accesstier_metro", "Metro edge"),
    ("accesstier_wide", "Wide-area mobile"),
    ("accesstier_field", "Non-terrestrial field"),
    ("accesstier", "Heterogeneous"),
]
METHODS = {
    "awf": "AWF",
    "logpool": "Full-Wait (log)",
    "anytime_cbci": "Confidence-Stop",
    "deadline_aware": "Arrival-Only",
    "generic_voi": "Entropy-Gain",
}

users_by_ds = {}
for ds in DATASETS:
    try:
        users_by_ds[ds] = R._ensure_users(ds)
    except Exception as exc:
        print("skip", ds, exc, flush=True)

rows = []
for preset, pname in PERSONAS:
    for ds, users in users_by_ds.items():
        for seed in range(SEEDS):
            for mk in METHODS:
                res, arr, y, nu = R.run_one(
                    mk, seed, ds, preset, users, deadline_ms=D,
                    alpha=1e-3, lambda_delay=1e-3)
                lat = np.array([d.decision_time_ms for d in res], float)
                rows.append(dict(persona=pname, preset=preset, dataset=ds,
                                 seed=seed, method=mk,
                                 accuracy=metrics.accuracy(res, y),
                                 latency=float(lat.mean())))
    print("done", pname, flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "table-usecase.csv"), index=False)

pa = (df.groupby(["persona", "method"])["accuracy"].mean().unstack() * 100)
pl = df.groupby(["persona", "method"])["latency"].mean().unstack()
order = [p for _, p in PERSONAS]
mk_order = list(METHODS.keys())

# ---- summary text ---------------------------------------------------------
lines = []
lines.append("=" * 82)
lines.append("DEPLOYMENT USE-CASE ROBUSTNESS  (D=50, %d seeds, 3 tasks)" % SEEDS)
lines.append("=" * 82)
lines.append("")
lines.append("accuracy (%)")
lines.append("%-22s " % "persona" + " ".join("%-16s" % METHODS[m] for m in mk_order))
for p in order:
    lines.append("%-22s " % p +
                 " ".join("%-16.2f" % pa.loc[p, m] for m in mk_order))
lines.append("")
lines.append("mean latency (ms)")
lines.append("%-22s " % "persona" + " ".join("%-16s" % METHODS[m] for m in mk_order))
for p in order:
    lines.append("%-22s " % p +
                 " ".join("%-16.2f" % pl.loc[p, m] for m in mk_order))
lines.append("")
lines.append("AWF vs Full-Wait(log) ceiling")
for p in order:
    gap = pa.loc[p, "awf"] - pa.loc[p, "logpool"]
    red = (1 - pl.loc[p, "awf"] / pl.loc[p, "logpool"]) * 100
    lines.append("  %-22s acc gap=%+.2f pt, latency reduction=%.0f%% "
                 "(%.1f -> %.1f ms)" % (
                     p, gap, red, pl.loc[p, "logpool"], pl.loc[p, "awf"]))
rep = "\n".join(lines)
print("\n" + rep)
with open(os.path.join(OUT, "usecase-summary.txt"), "w", encoding="utf-8") as f:
    f.write(rep)

# ---- figure ---------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200,
                     "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})
CLR = {"awf": "#d1495b", "logpool": "#1b998b", "anytime_cbci": "#2e86ab",
       "deadline_aware": "#5c4d7d", "generic_voi": "#c9a227"}
fig, ax = plt.subplots(1, 2, figsize=(15, 5.6))
x = np.arange(len(order))
show = ["awf", "logpool", "anytime_cbci", "deadline_aware", "generic_voi"]
w = 0.16
for i, m in enumerate(show):
    ax[0].bar(x + (i - 2) * w, [pa.loc[p, m] for p in order], w,
              color=CLR[m], label=METHODS[m])
ax[0].set_xticks(x)
ax[0].set_xticklabels(order, rotation=18, ha="right", fontsize=12)
ax[0].set_ylabel("accuracy (%)", fontsize=16)
ax[0].set_ylim(55, 100)
ax[0].set_title("(a) Accuracy across deployment use cases", fontsize=16)
ax[0].tick_params(axis="y", labelsize=13)
ax[0].legend(fontsize=12, ncol=2, loc="lower left", frameon=False)

for i, m in enumerate(show):
    ax[1].bar(x + (i - 2) * w, [pl.loc[p, m] for p in order], w,
              color=CLR[m], label=METHODS[m])
ax[1].axhline(D, color="0.4", ls=":", lw=1.5)
ax[1].text(len(order) - 0.5, D + 1, "deadline", fontsize=11, color="0.3",
           ha="right")
ax[1].set_xticks(x)
ax[1].set_xticklabels(order, rotation=18, ha="right", fontsize=12)
ax[1].set_ylabel("mean decision latency (ms)", fontsize=16)
ax[1].set_title("(b) Decision latency across use cases", fontsize=16)
ax[1].tick_params(axis="y", labelsize=13)
ax[1].legend(fontsize=12, ncol=2, loc="upper left", frameon=False)

fig.suptitle("AWF across realistic networked-cBCI deployments "
             "(near full-wait accuracy at much lower latency)", fontsize=17, y=1.0)
fig.tight_layout()
p = os.path.join(OUT, "fig-usecase.png")
fig.savefig(p)
plt.close(fig)
print("\nwrote table-usecase.csv, usecase-summary.txt,", p)
