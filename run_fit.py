"""Do the per-tier delays match conventional latency distribution families?

Fits candidate families (lognormal, gamma, Weibull, exponential, Pareto) to
each tier's one-way delay and ranks them by KS statistic and AIC.  Reports the
body vs tail behaviour so we can state whether the traces are 'typical'.
"""
import importlib.util
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("tb", os.path.join(HERE, "bcinet-tracebank.py"))
tb = importlib.util.module_from_spec(spec)
sys.modules["tb"] = tb
spec.loader.exec_module(tb)
bank = tb.load_tracebank()

TIERS = ["ble", "wifi", "5g", "lte", "leo"]
LABEL = {"ble": "BLE", "wifi": "Wi-Fi", "5g": "5G", "lte": "4G", "leo": "LEO"}
CANDS = {
    "lognormal": stats.lognorm,
    "gamma": stats.gamma,
    "Weibull": stats.weibull_min,
    "exponential": stats.expon,
    "Pareto": stats.pareto,
}
rng = np.random.RandomState(0)
out = []


def w(s=""):
    out.append(s)
    print(s, flush=True)


w("=" * 74)
w("PER-TIER DELAY vs CONVENTIONAL DISTRIBUTION FAMILIES")
w("(KS = Kolmogorov-Smirnov statistic, lower = closer; AIC lower = better)")
w("=" * 74)

for t in TIERS:
    d = np.asarray(bank[t].all_delays(), float)
    d = d[d > 0]
    if d.size > 20000:
        d = rng.choice(d, 20000, replace=False)
    n = d.size
    w("")
    w("%s  (n=%d, mean=%.2f ms, CV=%.2f, p99/p50=%.2f)" % (
        LABEL[t], n, d.mean(), d.std() / d.mean(),
        np.percentile(d, 99) / np.percentile(d, 50)))
    rows = []
    for name, dist in CANDS.items():
        try:
            if name in ("gamma", "Weibull", "lognormal", "Pareto"):
                par = dist.fit(d, floc=0)   # location fixed at 0 (delays>0)
            else:
                par = dist.fit(d)
            ks = stats.kstest(d, dist.cdf, args=par).statistic
            ll = np.sum(dist.logpdf(d, *par))
            aic = 2 * len(par) - 2 * ll
            rows.append((name, ks, aic))
        except Exception as exc:
            rows.append((name, np.nan, np.nan))
    rows.sort(key=lambda r: (np.inf if np.isnan(r[1]) else r[1]))
    w("  %-12s %-9s %-12s" % ("family", "KS", "AIC"))
    for name, ks, aic in rows:
        mark = "  <== best" if name == rows[0][0] else ""
        w("  %-12s %8.4f %12.0f%s" % (name, ks, aic, mark))

w("")
w("Interpretation")
w("-" * 74)
w("BLE delay is calibrated uniform(2,12) ms by construction (raw trace carries")
w("reachability, not application RTT), so it is NOT an empirical shape.")
w("Wi-Fi/cellular/LEO delays are measured RTT/2 and are expected to be right-")
w("skewed heavy-tailed, which is the conventional latency shape (lognormal body,")
w("heavier-than-lognormal tail).")

dst = os.path.join(HERE, "results", "accesstier", "dist-fit.txt")
with open(dst, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("\nwrote", dst)
