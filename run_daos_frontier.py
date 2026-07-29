"""Recompute the DA-OS frontier with a correctly scaled beta range.

The throughput value u*sum(rho_j) is on the order of the missing-user count
(roughly 1..9), while the delay cost beta*E_delay uses E_delay of tens of ms.
For DA-OS to trade throughput against latency, beta must reach the scale where
beta*E_delay is comparable to sum(rho_j), i.e. beta ~ (1..9)/(30 ms).  We
therefore sweep beta over a larger, properly scaled range and overwrite the
da_os rows in table-frontier.csv.
"""
import importlib.util
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AT = os.path.join(HERE, "results", "accesstier")


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


R = _load("runexp", "run-bcinet-experiment.py")
metrics = R.bcinet_metrics
ALL_DATASETS = list(R.DATASETS)
SEEDS = 15
DA_OS_BETAS = [3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0, 3e0]
preset = "accesstier"


def _pct(results):
    lat = np.array([d.decision_time_ms for d in results], float)
    return {"latency_mean": float(lat.mean()),
            "latency_p95": float(np.percentile(lat, 95)),
            "latency_p99": float(np.percentile(lat, 99))}


rows = []
for dataset in ALL_DATASETS:
    users = R._ensure_users(dataset)
    for seed in range(SEEDS):
        for bt in DA_OS_BETAS:
            res, arr, y, nu = R.run_one("da_os:%g" % bt, seed, dataset, preset,
                                        users, deadline_ms=50.0, alpha=1e-5,
                                        lambda_delay=1e-5)
            row = {"dataset": dataset, "seed": seed, "method": "da_os",
                   "sweep": bt, "accuracy": metrics.accuracy(res, y)}
            row.update(_pct(res))
            rows.append(row)
    print("done", dataset, flush=True)

new = pd.DataFrame(rows)
path = os.path.join(AT, "table-frontier.csv")
f = pd.read_csv(path)
f = f[f.method != "da_os"]           # drop old flat da_os rows
f = pd.concat([f, new], ignore_index=True)
f.to_csv(path, index=False)
print("rewrote", path)

g = new.groupby("sweep").agg(acc=("accuracy", "mean"),
                             lat=("latency_mean", "mean")).sort_values("lat")
for sw, r in g.iterrows():
    print("  beta=%-8g acc=%5.1f lat=%5.1f" % (sw, r["acc"] * 100, r["lat"]))
