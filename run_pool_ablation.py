"""2x2 ablation: pooling rule (log vs linear) x stopping (full-wait vs AWF).
 - logpool   = log-pool  + full wait  (Full-wait log-pool)
 - weighted  = linear    + full wait  (WeightedSum)
 - awf       = log-pool  + AWF stop
 - awf_linear= linear    + AWF stop
D=50, fixed deployment, 15 seeds."""
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

METHODS = ["logpool", "weighted", "awf", "awf_linear"]
LAB = {"logpool": "Full-wait log-pool", "weighted": "WeightedSum (linear, full-wait)",
       "awf": "AWF (log-pool + stop)", "awf_linear": "AWF + linear pool"}

rows = []
for ds in R.DATASETS:
    users = R._ensure_users(ds)
    print("dataset", ds, flush=True)
    for seed in range(15):
        for m in METHODS:
            res, arr, y, nu = R.run_one(
                m, seed, ds, "accesstier", users, deadline_ms=50.0,
                alpha=1e-5, lambda_delay=1e-5)
            lat = float(np.mean([r.decision_time_ms for r in res]))
            rows.append({"dataset": ds, "seed": seed, "method": m,
                         "accuracy": metrics.accuracy(res, y),
                         "decision_latency": lat})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, "results", "accesstier", "table-pool-ablation.csv"),
          index=False)

lines = ["2x2 pooling x stopping ablation, D=50, 15 seeds"]
for m in METHODS:
    sub = df[df.method == m]
    g = sub.groupby("seed").agg(acc=("accuracy", "mean"),
                                lat=("decision_latency", "mean"))
    a = g["acc"].values * 100
    l = g["lat"].values
    lines.append("%-34s acc %5.1f +/- %.1f   lat %5.1f +/- %.1f"
                 % (LAB[m], a.mean(), a.std(ddof=1), l.mean(), l.std(ddof=1)))
txt = "\n".join(lines)
with open(os.path.join(HERE, "results", "accesstier", "pool_ablation.txt"), "w") as f:
    f.write(txt + "\n")
print(txt, flush=True)
