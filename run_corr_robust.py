"""Correlation-robustness sweep: inject cross-user error correlation (rho_err)
and measure how the AWF vs Full-wait accuracy gap changes. D=50 ms, 15 seeds."""
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

RHOS = [0.0, 0.2, 0.4, 0.6, 0.8]
METHODS = ["awf", "logpool", "weighted", "anytime_cbci"]
LAB = {"awf": "AWF", "logpool": "Full-wait", "weighted": "WeightedSum",
       "anytime_cbci": "Anytime"}

rows = []
for ds in R.DATASETS:
    users = R._ensure_users(ds)
    print("dataset", ds, flush=True)
    for seed in range(15):
        for rho in RHOS:
            for m in METHODS:
                res, arr, y, nu = R.run_one(
                    m, seed, ds, "accesstier", users, deadline_ms=50.0,
                    alpha=1e-5, lambda_delay=1e-5, rho_err=rho)
                lat = float(np.mean([d.decision_time_ms for d in res]))
                rows.append({"dataset": ds, "seed": seed, "rho_err": rho,
                             "method": m, "accuracy": metrics.accuracy(res, y),
                             "decision_latency": lat})

df = pd.DataFrame(rows)
out_csv = os.path.join(HERE, "results", "accesstier", "table-corr-robust.csv")
df.to_csv(out_csv, index=False)

lines = ["DONE rows %d" % len(df)]
for rho in RHOS:
    lines.append("--- rho_err=%.1f ---" % rho)
    for m in METHODS:
        sub = df[(df.method == m) & (df.rho_err == rho)]
        g = sub.groupby("seed").agg(acc=("accuracy", "mean"),
                                    lat=("decision_latency", "mean"))
        a = g["acc"].values * 100
        l = g["lat"].values
        lines.append("  %-11s acc %5.1f +/- %.1f   lat %5.1f"
                     % (LAB[m], a.mean(), a.std(ddof=1), l.mean()))
    aw = df[(df.method == "awf") & (df.rho_err == rho)].groupby("seed")["accuracy"].mean().mean() * 100
    fw = df[(df.method == "logpool") & (df.rho_err == rho)].groupby("seed")["accuracy"].mean().mean() * 100
    lines.append("  GAP Full-wait - AWF = %.2f pp" % (fw - aw))

summary = "\n".join(lines)
with open(os.path.join(HERE, "results", "accesstier", "corr_summary.txt"), "w") as f:
    f.write(summary + "\n")
print(summary, flush=True)
