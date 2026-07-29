"""Summarise the swept operating frontiers per method from table-frontier.csv."""
import os
import pandas as pd

AT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "results", "accesstier")
df = pd.read_csv(os.path.join(AT, "table-frontier.csv"))
buf = []


def emit(s=""):
    buf.append(str(s))


for m in ["awf", "generic_voi", "anytime_cbci_cw", "da_os", "deadline_aware"]:
    d = df[(df.method == m) & df["sweep"].notna()]
    if d.empty:
        continue
    g = d.groupby("sweep").agg(acc=("accuracy", "mean"),
                               lat=("latency_mean", "mean")).reset_index()
    g = g.sort_values("lat")
    g["acc"] *= 100.0
    emit("=== %s ===" % m)
    for _, r in g.iterrows():
        emit("  knob=%-8g  lat=%6.2f  acc=%6.2f" %
             (r["sweep"], r["lat"], r["acc"]))
    emit("  acc range: %.2f .. %.2f | lat range: %.2f .. %.2f" %
         (g["acc"].min(), g["acc"].max(), g["lat"].min(), g["lat"].max()))
    emit()

with open(os.path.join(AT, "frontier-summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(buf))
print("\n".join(buf))
