"""Diagnostic after the Entropy-gain fix: per-method accuracy, latency, and
mean number of arrived (used) responses at the stop time, D=50, fixed
deployment, 5 seeds. Also prints simple sanity relations."""
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


R = _load("runexp", "run-bcinet-experiment.py")
metrics = R.bcinet_metrics

METHODS = ["weighted", "logpool", "anytime_cbci", "deadline_aware",
           "generic_voi", "awf"]
LAB = {"weighted": "WeightedSum", "logpool": "Full-wait",
       "anytime_cbci": "Anytime", "deadline_aware": "Deadline-aware",
       "generic_voi": "Entropy-gain", "awf": "AWF"}

rows = []
for ds in R.DATASETS:
    users = R._ensure_users(ds)
    print("dataset", ds, flush=True)
    for seed in range(5):
        for m in METHODS:
            res, arr, y, nu = R.run_one(
                m, seed, ds, "accesstier", users, deadline_ms=50.0,
                alpha=1e-5, lambda_delay=1e-5)
            lat = float(np.mean([r.decision_time_ms for r in res]))
            used = float(np.mean([int(r.used_mask.sum()) for r in res]))
            rows.append({"dataset": ds, "seed": seed, "method": m,
                         "accuracy": metrics.accuracy(res, y),
                         "decision_latency": lat, "mean_used": used})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, "results", "accesstier", "diag_baselines.csv"),
          index=False)

lines = ["D=50 fixed, 5 seeds (after Entropy-gain fix)",
         "%-14s  acc     lat    mean_used" % "method"]
for m in METHODS:
    sub = df[df.method == m]
    g = sub.groupby("seed").agg(acc=("accuracy", "mean"),
                                lat=("decision_latency", "mean"),
                                used=("mean_used", "mean"))
    lines.append("%-14s  %5.1f   %5.1f   %.2f"
                 % (LAB[m], g["acc"].mean() * 100, g["lat"].mean(),
                    g["used"].mean()))
txt = "\n".join(lines)
with open(os.path.join(HERE, "results", "accesstier", "diag_baselines.txt"), "w") as f:
    f.write(txt + "\n")
print(txt, flush=True)
