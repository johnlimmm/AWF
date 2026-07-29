"""Quick check: fixed per-seed deployment (user->tier fixed across all trials,
only delay/loss resampled per trial) vs the current i.i.d. re-draw. 5 seeds."""
import importlib.util
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


rat = _load("rat", "run-accesstier.py")
metrics = rat.metrics

METHODS = ["weighted", "logpool", "anytime_cbci", "deadline_aware",
           "generic_voi", "awf"]
LAB = {"weighted": "WeightedSum", "logpool": "Full-wait",
       "anytime_cbci": "Anytime", "deadline_aware": "Deadline-aware",
       "generic_voi": "Entropy-gain", "awf": "AWF"}
N_SEEDS = 5

rows = []
for ds in rat.ALL_DATASETS:
    users = rat.R._ensure_users(ds)
    print("dataset", ds, flush=True)
    for seed in range(N_SEEDS):
        for m in METHODS:
            res, arr, y, nu = rat._mobility_run(
                m, seed, ds, users, "fixed", "accesstier", 50.0, 1e-5, 1e-5)
            lat = float(np.mean([r.decision_time_ms for r in res]))
            rows.append({"dataset": ds, "seed": seed, "method": m,
                         "accuracy": metrics.accuracy(res, y),
                         "decision_latency": lat})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, "results", "accesstier", "table-fixed-deploy-quick.csv"),
          index=False)

lines = ["FIXED deployment (per-seed user->tier), D=50, %d seeds" % N_SEEDS]
for m in METHODS:
    sub = df[df.method == m]
    g = sub.groupby("seed").agg(acc=("accuracy", "mean"),
                                lat=("decision_latency", "mean"))
    a = g["acc"].values * 100
    l = g["lat"].values
    lines.append("  %-14s acc %5.1f +/- %.1f   lat %5.1f +/- %.1f"
                 % (LAB[m], a.mean(), a.std(ddof=1), l.mean(), l.std(ddof=1)))
txt = "\n".join(lines)
with open(os.path.join(HERE, "results", "accesstier", "fixed_deploy_quick.txt"), "w") as f:
    f.write(txt + "\n")
print(txt, flush=True)
