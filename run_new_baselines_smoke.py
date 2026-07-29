"""Smoke test + quick head-to-head for the two new prior-work baselines:
Anytime cBCI (confidence-weighted) and DA-OS (He and Zhang).
Runs on the real access-tier pipeline, small scale, to confirm they work and
place them against AWF / Full-Wait / First-Arrival.
"""
import importlib.util
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


R = _load("runexp", "run-bcinet-experiment.py")
net = R.bcinet_network
data = R.bcinet_data
fusion = R.bcinet_fusion
DATASETS = list(R.DATASETS)
LAM = 1e-5
ALPHA = 1e-5

METHODS = [
    ("awf", "AWF"),
    ("anytime_cbci_cw", "Anytime-cBCI (CW)"),
    ("da_os", "DA-OS (He-Zhang)"),
    ("logpool", "Full-Wait"),
    ("fastest_only", "First-Arrival"),
    ("anytime_cbci", "Calib-Conf-Stop"),
]

# Build access-tier profiles the same way the main experiment does.
TIERS = ["ble", "wifi", "5g", "lte", "leo"]
tb = _load("tb", "bcinet-tracebank.py")
bank = tb.load_tracebank()


def _profile(tier, rng):
    b = bank[tier]
    samples = np.sort(b.all_delays().astype(float))
    p10, p90 = float(np.percentile(samples, 10)), float(np.percentile(samples, 90))
    if p90 <= p10:
        p90 = p10 + 1.0
    loss = float(b.overall_loss())
    return net.NetworkProfile(tier=tier, delay_range=(p10, p90),
                              jitter_range=(0.0, 0.0), loss_prob=loss,
                              state=None, est_samples=samples, est_loss=loss), samples, loss


rows = {m: {"acc": [], "lat": []} for m, _ in METHODS}
D = 100.0
ds = DATASETS[0]
users = R._ensure_users(ds)
val_acc = np.array([u.val_accuracy for u in users])
confusions = [u.confusion for u in users]
prof_cache = {t: _profile(t, None) for t in TIERS}

for seed in range(5):
    rng = np.random.RandomState(seed)
    P_cube, y = data.align_test_trials(users, rng)
    nu, ntr = len(users), P_cube.shape[1]
    tiers = [TIERS[int(rng.randint(len(TIERS)))] for _ in range(nu)]
    profs = [prof_cache[t][0] for t in tiers]
    arrivals = np.empty((nu, ntr), float)
    for uix, t in enumerate(tiers):
        s = prof_cache[t][1]
        a = rng.choice(s, ntr).astype(float)
        a[rng.random(ntr) < prof_cache[t][2]] = np.inf
        arrivals[uix] = a
    for mname, _ in METHODS:
        fn = fusion.get_method(mname)
        acc, lat = [], []
        for tt in range(ntr):
            r = fn(P=P_cube[:, tt, :], arrivals_ms=arrivals[:, tt], deadline_ms=D,
                   val_acc=val_acc, profiles=profs, confusions=confusions,
                   alpha=ALPHA, lambda_delay=LAM, y_true=int(y[tt]))
            acc.append(int(r.pred == int(y[tt])))
            lat.append(r.decision_time_ms)
        rows[mname]["acc"].append(np.mean(acc) * 100)
        rows[mname]["lat"].append(np.mean(lat))

print("%-22s %8s %8s" % ("method", "acc%", "lat(ms)"))
print("-" * 40)
for m, lab in METHODS:
    print("%-22s %8.2f %8.1f" % (lab, np.mean(rows[m]["acc"]), np.mean(rows[m]["lat"])))
