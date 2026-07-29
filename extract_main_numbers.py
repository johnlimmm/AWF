"""Extract paper-ready numbers for the new prior-work baselines and AWF from
the regenerated main tables, formatted as mean_std over the 15 seeds."""
import os
import numpy as np
import pandas as pd

AT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "accesstier")
_buf = []


def out(s=""):
    _buf.append(str(s))
    print(s)

METHODS = ["fastest_only", "earliest_k:3", "anytime_cbci", "anytime_cbci_cw",
           "da_os", "deadline_aware", "generic_voi", "logpool", "weighted", "awf"]
LAB = {"fastest_only": "First-Arrival", "earliest_k:3": "First-k",
       "anytime_cbci": "Confidence-Stop", "anytime_cbci_cw": "Anytime-cBCI",
       "da_os": "DA-OS", "deadline_aware": "Arrival-Only",
       "generic_voi": "Entropy-Gain", "logpool": "Full-Wait(log)",
       "weighted": "Full-Wait(lin)", "awf": "AWF"}
DS = {"Zhou2016": "Zhou", "BNCI2014_001": "BNCI", "Weibo2014": "Weibo"}


def seed_means(df, m, col):
    d = df[df.method == m]
    # per-seed mean over datasets, then mean/std across seeds
    g = d.groupby("seed")[col].mean()
    return g.mean(), g.std()


for D in (50, 100):
    df = pd.read_csv(os.path.join(AT, "table-main-d%d.csv" % D))
    acol = "accuracy"
    lcol = "decision_latency"
    out("\n===== MAIN  D=%d =====" % D)
    out("%-16s %14s %14s" % ("method", "acc%", "lat"))
    for m in METHODS:
        am, asd = seed_means(df, m, acol)
        lm, lsd = seed_means(df, m, lcol)
        out("%-16s  %5.1f_%4.1f    %5.1f_%4.1f" % (LAB[m], am * 100, asd * 100, lm, lsd))

# per-dataset at D=50
df = pd.read_csv(os.path.join(AT, "table-main-d50.csv"))
acol = "accuracy"
lcol = "decision_latency"
out("\n===== PER-DATASET  D=50 (acc%_std / lat) =====")
out("%-16s %18s %18s %18s" % ("method", "Zhou", "BNCI", "Weibo"))
for m in METHODS:
    cells = []
    for ds in ("Zhou2016", "BNCI2014_001", "Weibo2014"):
        d = df[(df.method == m) & (df.dataset == ds)]
        a = d.groupby("seed")[acol].mean()
        l = d.groupby("seed")[lcol].mean()
        cells.append("%5.1f_%4.1f/%4.1f" % (a.mean() * 100, a.std() * 100, l.mean()))
    out("%-16s %18s %18s %18s" % (LAB[m], cells[0], cells[1], cells[2]))

with open(os.path.join(AT, "main-numbers.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(_buf))
