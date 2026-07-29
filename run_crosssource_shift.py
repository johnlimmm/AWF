"""Quantify the cross-source distribution shift available in the raw traces,
to confirm a cross-dataset (not train/test split) evaluation is meaningful.

  5G / 4G : Static condition  vs  Driving condition   (mobility shift)
  LEO     : Enschede (NL)      vs  Osnabruck (DE)      (cross-city)

Reports per source the per-sample one-way delay percentiles and the
Kolmogorov-Smirnov distance between the two sources.  No modelling.
"""
import csv
import glob
import os
import numpy as np
from scipy.stats import ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))
CELL = os.path.join(HERE, "traces", "cellular", "5G-production-dataset")
LEO = os.path.join(HERE, "traces", "leo")

CELL_5G = {"5G"}
CELL_LTE = {"LTE", "HSPA+", "HSPA", "HSUPA", "HSDPA", "UMTS", "EDGE", "GPRS"}


def cell_delays(condition, rat_modes):
    """One-way PINGAVG/2 samples for files under a mobility condition + RAT."""
    out = []
    files = glob.glob(os.path.join(CELL, "**", "*.csv"), recursive=True)
    for fp in files:
        if os.sep + condition + os.sep not in fp:
            continue
        with open(fp, encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                nm = (row.get("NetworkMode") or "").strip()
                if nm not in rat_modes:
                    continue
                pv = row.get("PINGAVG", "-")
                if pv in ("-", "", None):
                    continue
                try:
                    ow = float(pv) / 2.0
                except ValueError:
                    continue
                if 0 < ow < 5000:
                    out.append(ow)
    return np.asarray(out, float)


def leo_delays(city):
    fp = os.path.join(LEO, f"wetlinks_{city}_net_ping.csv")
    out = []
    with open(fp, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                a = float(row["ping_avg"])
            except (KeyError, ValueError):
                continue
            if a > 0:
                out.append(a / 2.0)
    return np.asarray(out, float)


def report(name, a, b, la, lb):
    ks = ks_2samp(a, b).statistic if a.size and b.size else float("nan")
    print(f"\n{name}  (KS = {ks:.3f})")
    print(f"  {'source':<12} {'N':>7} {'p50':>8} {'p90':>8} {'p95':>8} {'p99':>8} {'mean':>8}")
    for lab, s in ((la, a), (lb, b)):
        if s.size:
            print(f"  {lab:<12} {s.size:>7} "
                  f"{np.percentile(s,50):>8.2f} {np.percentile(s,90):>8.2f} "
                  f"{np.percentile(s,95):>8.2f} {np.percentile(s,99):>8.2f} "
                  f"{s.mean():>8.2f}")
    return ks


print("=" * 70)
print("CROSS-SOURCE DISTRIBUTION SHIFT (raw one-way delays, ms)")
print("=" * 70)
s5, d5 = cell_delays("Static", CELL_5G), cell_delays("Driving", CELL_5G)
report("5G  Static vs Driving", s5, d5, "5G-Static", "5G-Driving")
sl, dl = cell_delays("Static", CELL_LTE), cell_delays("Driving", CELL_LTE)
report("4G  Static vs Driving", sl, dl, "4G-Static", "4G-Driving")
en, os_ = leo_delays("enschede"), leo_delays("osnabruck")
report("LEO Enschede vs Osnabruck", en, os_, "LEO-Ensch", "LEO-Osnab")
print()
