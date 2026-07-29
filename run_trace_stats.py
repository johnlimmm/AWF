"""Statistical characterization of the access-tier trace bank.

Answers the advisor's question: how static/stationary are the traces, and
therefore how much of AWF's arrival model is 'true by construction'?
Writes a plain-text report to results/accesstier/trace-stats.txt.
"""
import importlib.util
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "tb", os.path.join(HERE, "bcinet-tracebank.py"))
tb = importlib.util.module_from_spec(spec)
sys.modules["tb"] = tb
spec.loader.exec_module(tb)

bank = tb.load_tracebank()
TIERS = ["ble", "wifi", "5g", "lte", "leo"]
LABEL = {"ble": "BLE", "wifi": "Wi-Fi", "5g": "5G", "lte": "4G", "leo": "LEO"}

out = []


def w(s=""):
    out.append(s)


def lag1_acf(x):
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return np.nan
    x = x - x.mean()
    d = np.dot(x, x)
    if d <= 0:
        return np.nan
    return float(np.dot(x[:-1], x[1:]) / d)


w("=" * 70)
w("ACCESS-TIER TRACE BANK  --  STATISTICAL CHARACTERIZATION")
w("=" * 70)

# ---- 1. Marginal delay shape per tier -------------------------------------
w("")
w("[1] One-way delay marginal (ms) and loss per tier")
w("-" * 70)
w(f"{'tier':6} {'n':>8} {'mean':>7} {'std':>7} {'CV':>6} {'p50':>7} "
  f"{'p95':>7} {'p99':>7} {'p99/p50':>8} {'skew':>7} {'kurt':>8} {'loss%':>7}")
tier_delays = {}
for t in TIERS:
    d = np.asarray(bank[t].all_delays(), dtype=float)
    tier_delays[t] = d
    mean, std = d.mean(), d.std()
    cv = std / mean if mean > 0 else np.nan
    p50, p95, p99 = np.percentile(d, [50, 95, 99])
    tailr = p99 / p50 if p50 > 0 else np.nan
    sk = stats.skew(d)
    ku = stats.kurtosis(d)  # excess kurtosis
    loss = bank[t].overall_loss() * 100.0
    w(f"{LABEL[t]:6} {d.size:8d} {mean:7.2f} {std:7.2f} {cv:6.2f} {p50:7.2f} "
      f"{p95:7.2f} {p99:7.2f} {tailr:8.2f} {sk:7.2f} {ku:8.2f} {loss:7.2f}")

# ---- 2. Temporal structure / stationarity of ordered sequences ------------
w("")
w("[2] Temporal structure of contiguous replay sequences per tier")
w("    (lag-1 autocorrelation ~0 => memoryless/i.i.d.-like arrivals;")
w("     split-half mean drift => non-stationarity within a source)")
w("-" * 70)
w(f"{'tier':6} {'#seq':>5} {'len_med':>8} {'acf1_med':>9} {'acf1_mean':>10} "
  f"{'|drift|%_med':>13}")
for t in TIERS:
    seqs = [np.asarray(s, dtype=float) for s in bank[t].sequences
            if np.asarray(s).size >= 6]
    if not seqs:
        w(f"{LABEL[t]:6} {'--':>5}  (no contiguous sequences; bootstrap tier)")
        continue
    acfs, drifts, lens = [], [], []
    for s in seqs:
        acfs.append(lag1_acf(s))
        h = s.size // 2
        m1, m2 = s[:h].mean(), s[h:].mean()
        base = s.mean()
        drifts.append(abs(m2 - m1) / base * 100.0 if base > 0 else np.nan)
        lens.append(s.size)
    acfs = np.array(acfs, float)
    drifts = np.array(drifts, float)
    w(f"{LABEL[t]:6} {len(seqs):5d} {np.median(lens):8.0f} "
      f"{np.nanmedian(acfs):9.3f} {np.nanmean(acfs):10.3f} "
      f"{np.nanmedian(drifts):13.1f}")

# ---- 3. Cross-source (between-deployment) heterogeneity --------------------
w("")
w("[3] Cross-source heterogeneity of per-sequence mean delay")
w("    (how different are individual measured links within one tier;")
w("     high spread => a fixed marginal hides strong per-user variation)")
w("-" * 70)
w(f"{'tier':6} {'#seq':>5} {'grand_mean':>11} {'between_CV':>11} "
  f"{'min_mean':>9} {'max_mean':>9}")
for t in TIERS:
    seqs = [np.asarray(s, dtype=float) for s in bank[t].sequences
            if np.asarray(s).size >= 6]
    if not seqs:
        w(f"{LABEL[t]:6} {'--':>5}  (no per-source sequences)")
        continue
    means = np.array([s.mean() for s in seqs])
    gm = means.mean()
    bcv = means.std() / gm if gm > 0 else np.nan
    w(f"{LABEL[t]:6} {len(seqs):5d} {gm:11.2f} {bcv:11.2f} "
      f"{means.min():9.2f} {means.max():9.2f}")

# ---- 4. Deadline feasibility as a function of D ---------------------------
w("")
w("[4] Fraction of responses arriving within deadline D (per tier)")
w("    (how binding the deadline is; if ~1.0 for all D the timing")
w("     decision is trivial, if it varies the stop/wait choice matters)")
w("-" * 70)
Ds = [10, 20, 30, 40, 50, 75, 100]
w(f"{'tier':6} " + " ".join(f"D={d:<4}" for d in Ds))
for t in TIERS:
    d = tier_delays[t]
    row = []
    for D in Ds:
        # include loss: a lost response never arrives
        p_arr = (d <= D).mean() * (1.0 - bank[t].overall_loss())
        row.append(f"{p_arr:6.2f}")
    w(f"{LABEL[t]:6} " + " ".join(row))

# ---- 5. Committee-level arrival spread ------------------------------------
w("")
w("[5] Summary read")
w("-" * 70)
cvs = {t: tier_delays[t].std() / tier_delays[t].mean() for t in TIERS}
w(f"delay CV range across tiers: "
  f"{min(cvs.values()):.2f} (min) .. {max(cvs.values()):.2f} (max)")
tailr = {t: np.percentile(tier_delays[t], 99) / np.percentile(tier_delays[t], 50)
         for t in TIERS}
w(f"tail ratio p99/p50 range   : "
  f"{min(tailr.values()):.2f} .. {max(tailr.values()):.2f}")

report = "\n".join(out)
print(report)
dst = os.path.join(HERE, "results", "accesstier", "trace-stats.txt")
with open(dst, "w", encoding="utf-8") as f:
    f.write(report)
print("\nwrote", dst)
