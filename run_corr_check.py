"""Verify baseline consistency: rho=0 (uniform bootstrap path) vs rho~=0
(copula path). If they match, the rho=0.2 bump in the main sweep is genuine
sampling noise; if the copula-consistent ~0 point aligns with the rho>=0.2
trend, the bump is a code-path artifact and rho~=0 should be used as the
baseline."""
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

rows = []
for ds in R.DATASETS:
    u = R._ensure_users(ds)
    print("dataset", ds, flush=True)
    for seed in range(15):
        for rho in [0.0, 1e-6]:
            for m in ["awf", "logpool"]:
                res, a, y, nu = R.run_one(
                    m, seed, ds, "accesstier", u, deadline_ms=50.0,
                    alpha=1e-5, lambda_delay=1e-5, rho_err=rho)
                rows.append({"ds": ds, "seed": seed, "rho": rho, "m": m,
                             "acc": metrics.accuracy(res, y)})

df = pd.DataFrame(rows)
lines = []
for rho in [0.0, 1e-6]:
    for m in ["awf", "logpool"]:
        v = df[(df.rho == rho) & (df.m == m)].groupby("seed")["acc"].mean().values * 100
        lines.append("rho=%.0e %-8s acc %.2f +/- %.2f" % (rho, m, v.mean(), v.std(ddof=1)))
txt = "\n".join(lines)
with open(os.path.join(HERE, "results", "accesstier", "corr_check.txt"), "w") as f:
    f.write(txt + "\n")
print(txt, flush=True)
