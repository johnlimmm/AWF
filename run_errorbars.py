import os
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "accesstier")

order = [("fastest_only", "Fastest-only"), ("earliest_k:3", "Earliest-k"),
         ("anytime_cbci", "Anytime"), ("deadline_aware", "Deadline-aware"),
         ("generic_voi", "Entropy-gain"), ("logpool", "Full-wait"),
         ("weighted", "WeightedSum"), ("awf", "AWF")]

lines = []
for D in (50, 100):
    d = pd.read_csv(os.path.join(OUT, "table-main-d%d.csv" % D))
    lines.append("==== tab_main D=%d (mean +/- std over 15 seeds of 3-dataset mean) ====" % D)
    for m, lab in order:
        sub = d[d.method == m]
        g = sub.groupby("seed").agg(acc=("accuracy", "mean"),
                                    lat=("decision_latency", "mean"))
        a = g["acc"].values * 100
        l = g["lat"].values
        lines.append("%-14s acc %5.1f +/- %.1f   lat %5.1f +/- %.1f"
                     % (lab, a.mean(), a.std(ddof=1), l.mean(), l.std(ddof=1)))
    lines.append("")

d = pd.read_csv(os.path.join(OUT, "table-main-d50.csv"))
lines.append("==== tab_perds D=50 per-dataset (acc mean+/-std / lat mean) ====")
for m, lab in order:
    row = "%-14s" % lab
    for ds in ["Zhou2016", "BNCI2014_001", "Weibo2014"]:
        sub = d[(d.method == m) & (d.dataset == ds)]
        a = sub["accuracy"].values * 100
        l = sub["decision_latency"].values
        row += "  %s: %.1f+/-%.1f / %.1f(+/-%.1f)" % (ds[:4], a.mean(), a.std(ddof=1), l.mean(), l.std(ddof=1))
    lines.append(row)

txt = "\n".join(lines)
with open(os.path.join(OUT, "errorbars_fixed.txt"), "w") as f:
    f.write(txt + "\n")
print(txt)
