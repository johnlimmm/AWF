"""Loss=0 ablation: rerun AWF and Full-wait with the per-trial packet loss
removed (arrivals never dropped), to check whether the headline accuracy and
latency behavior depends on the trace loss rates. D=50 and D=100, 15 seeds."""
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
net = R.bcinet_network
metrics = R.bcinet_metrics

_orig = net.sample_arrivals


def _noloss_sample_arrivals(profiles, n_trials, rng):
    n_users = len(profiles)
    arrival_ms = np.zeros((n_users, n_trials), dtype=np.float64)
    lost = np.zeros((n_users, n_trials), dtype=bool)
    for ui, p in enumerate(profiles):
        if p.trace_samples is not None:
            idx = rng.randint(0, p.trace_samples.size, size=n_trials)
            arrival_ms[ui] = p.trace_samples[idx].astype(np.float64)
            continue
        d_lo, d_hi = p.delay_range
        base = rng.uniform(d_lo, d_hi, size=n_trials)
        jitter = rng.normal(0.0, 0.5 * float(np.mean(p.jitter_range)), size=n_trials)
        arrival_ms[ui] = np.clip(base + jitter, 0.0, None)
    return arrival_ms, lost


net.sample_arrivals = _noloss_sample_arrivals

LAB = {"logpool": "Full-wait", "awf": "AWF"}
lines = ["Loss=0 ablation (per-trial loss removed), 15 seeds"]
for D in (50.0, 100.0):
    for m in ("logpool", "awf"):
        accs, lats = [], []
        for ds in R.DATASETS:
            users = R._ensure_users(ds)
            for seed in range(15):
                res, arr, y, nu = R.run_one(m, seed, ds, "accesstier", users,
                                            deadline_ms=D, alpha=1e-5,
                                            lambda_delay=1e-5)
                accs.append(metrics.accuracy(res, y))
                lats.append(float(np.mean([r.decision_time_ms for r in res])))
        lines.append("D=%3.0f  %-10s acc %5.1f   lat %5.1f"
                     % (D, LAB[m], np.mean(accs) * 100, np.mean(lats)))

net.sample_arrivals = _orig
txt = "\n".join(lines)
with open(os.path.join(HERE, "results", "accesstier", "loss0_ablation.txt"), "w") as f:
    f.write(txt + "\n")
print(txt, flush=True)
