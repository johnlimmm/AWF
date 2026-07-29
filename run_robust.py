"""Robustness to arrival-model mismatch and non-stationarity (drift).

Directly addresses the circularity concern: in the main setup AWF's arrival
belief equals the generative distribution.  Here the realised arrivals are
biased (scale) or drift across trials while AWF keeps the nominal (stationary)
belief, so AWF acts on a WRONG arrival model.  We measure how AWF and the
key baselines degrade.  Writes CSVs + a summary to results/accesstier/.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load("runexp", "run-bcinet-experiment.py")
metrics = R.bcinet_metrics
OUT = os.path.join(HERE, "results", "accesstier")

DATASETS = list(R.DATASETS)
SEEDS = 8
PRESET = "accesstier"
D = 50.0
# method key -> display label
METHODS = {
    "awf": "AWF",
    "logpool": "Full-Wait (log)",
    "anytime_cbci": "Confidence-Stop",
    "deadline_aware": "Arrival-Only",
    "generic_voi": "Entropy-Gain",
}
SCALES = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
DRIFTS = [0.0, 0.25, 0.5, 0.75, 1.0]

users_by_ds = {}
for ds in DATASETS:
    try:
        users_by_ds[ds] = R._ensure_users(ds)
    except Exception as exc:
        print("skip", ds, exc, flush=True)


def _run(scale, drift):
    rows = []
    for ds, users in users_by_ds.items():
        for seed in range(SEEDS):
            for mkey in METHODS:
                res, arr, y, nu = R.run_one(
                    mkey, seed, ds, PRESET, users, deadline_ms=D,
                    alpha=1e-3, lambda_delay=1e-3,
                    arrival_scale=scale, arrival_drift=drift)
                lat = np.array([d.decision_time_ms for d in res], dtype=float)
                rows.append(dict(dataset=ds, seed=seed, method=mkey,
                                 scale=scale, drift=drift,
                                 accuracy=metrics.accuracy(res, y),
                                 latency=float(lat.mean())))
    return rows


# --- E7 mismatch (fixed multiplicative bias, no drift) ---------------------
print("[mismatch] sweeping arrival_scale ...", flush=True)
mm = []
for s in SCALES:
    mm += _run(s, 0.0)
    print("  scale=%.2f done" % s, flush=True)
mm = pd.DataFrame(mm)
mm.to_csv(os.path.join(OUT, "table-mismatch.csv"), index=False)

# --- E8 drift (linear ramp across trials) ----------------------------------
print("[drift] sweeping arrival_drift ...", flush=True)
dr = []
for g in DRIFTS:
    dr += _run(1.0, g)
    print("  drift=%.2f done" % g, flush=True)
dr = pd.DataFrame(dr)
dr.to_csv(os.path.join(OUT, "table-drift.csv"), index=False)


def _summ(df, key, grid):
    lines = []
    piv_a = (df.groupby([key, "method"])["accuracy"].mean().unstack() * 100.0)
    piv_l = df.groupby([key, "method"])["latency"].mean().unstack()
    order = list(METHODS.keys())
    lines.append("accuracy (%%) by %s" % key)
    lines.append("%-7s " % key + " ".join("%-16s" % METHODS[m] for m in order))
    for v in grid:
        lines.append("%-7.2f " % v +
                     " ".join("%-16.2f" % piv_a.loc[v, m] for m in order))
    lines.append("")
    lines.append("mean latency (ms) by %s" % key)
    lines.append("%-7s " % key + " ".join("%-16s" % METHODS[m] for m in order))
    for v in grid:
        lines.append("%-7.2f " % v +
                     " ".join("%-16.2f" % piv_l.loc[v, m] for m in order))
    return "\n".join(lines)


rep = []
rep.append("=" * 78)
rep.append("ROBUSTNESS TO ARRIVAL-MODEL MISMATCH AND DRIFT  (D=50, %d seeds)" % SEEDS)
rep.append("AWF uses the NOMINAL (stationary) arrival belief; realised arrivals")
rep.append("are scaled (mismatch) or drift across trials (non-stationarity).")
rep.append("=" * 78)
rep.append("")
rep.append("[E7] Multiplicative mismatch  (scale>1 => true delays worse than believed)")
rep.append("-" * 78)
rep.append(_summ(mm, "scale", SCALES))
rep.append("")
rep.append("[E8] Within-session drift  (drift=1.0 => last trial +100%% delay vs first)")
rep.append("-" * 78)
rep.append(_summ(dr, "drift", DRIFTS))
rep = "\n".join(rep)
print("\n" + rep)
with open(os.path.join(OUT, "robust-summary.txt"), "w", encoding="utf-8") as f:
    f.write(rep)
print("\nwrote table-mismatch.csv, table-drift.csv, robust-summary.txt")
