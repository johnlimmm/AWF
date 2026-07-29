"""DA-OS at the throughput-latency exchange rate beta whose D=50 latency is
closest to AWF's operating latency, so it enters the main tables as an
iso-latency comparison rather than a duplicate of Full-Wait.  The beta is
selected automatically by sweeping a small grid and picking the point closest
to AWF_TARGET_LAT, then the same beta is reported at D=50 and D=100 and per
dataset.  Output is written to results/accesstier/daos-point.txt."""
import importlib.util
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AT = os.path.join(HERE, "results", "accesstier")
AWF_TARGET_LAT = 18.2   # AWF D=50 latency at the 1e-3 default (rebuilt 4-tier bank)
BETA_GRID = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
DEFAULT = 1e-3


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
preset = "accesstier"


def run(D, beta):
    per_seed_acc, per_seed_lat = {}, {}
    perds = {}
    for dataset in DATASETS:
        users = R._ensure_users(dataset)
        accs, lats = [], []
        for seed in range(SEEDS):
            res, arr, y, nu = R.run_one("da_os:%g" % beta, seed, dataset,
                                        preset, users, deadline_ms=D,
                                        alpha=DEFAULT, lambda_delay=DEFAULT)
            a = metrics.accuracy(res, y)
            l = float(np.mean([d.decision_time_ms for d in res]))
            accs.append(a); lats.append(l)
            per_seed_acc.setdefault(seed, []).append(a)
            per_seed_lat.setdefault(seed, []).append(l)
        perds[dataset] = (np.mean(accs) * 100, np.std(accs) * 100, np.mean(lats))
    sa = np.array([np.mean(per_seed_acc[s]) for s in range(SEEDS)])
    sl = np.array([np.mean(per_seed_lat[s]) for s in range(SEEDS)])
    return sa.mean() * 100, sa.std() * 100, sl.mean(), sl.std(), perds


buf = []


def emit(s=""):
    buf.append(str(s))


# select beta at D=50 closest to AWF target latency
cand = []
for b in BETA_GRID:
    am, asd, lm, lsd, perds = run(50.0, b)
    cand.append((abs(lm - AWF_TARGET_LAT), b, am, asd, lm, lsd, perds))
    emit("scan beta=%-5g  D=50  acc=%.1f_%.1f  lat=%.1f_%.1f" %
         (b, am, asd, lm, lsd))
cand.sort()
_, best_beta, am, asd, lm, lsd, perds = cand[0]
emit("")
emit("SELECTED beta=%g (closest to AWF %.1f ms)" % (best_beta, AWF_TARGET_LAT))
emit("D=50  DA-OS(beta=%g):  acc=%.1f_%.1f  lat=%.1f_%.1f" %
     (best_beta, am, asd, lm, lsd))
for ds, (a, s, l) in perds.items():
    emit("   %-14s %.1f_%.1f / %.1f" % (ds, a, s, l))

am2, asd2, lm2, lsd2, _ = run(100.0, best_beta)
emit("D=100 DA-OS(beta=%g):  acc=%.1f_%.1f  lat=%.1f_%.1f" %
     (best_beta, am2, asd2, lm2, lsd2))

with open(os.path.join(AT, "daos-point.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(buf))
print("\n".join(buf))
