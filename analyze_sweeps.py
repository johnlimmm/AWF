"""Summarise alpha and lambda sensitivity sweeps into per-knob aggregates."""
import pandas as pd
import numpy as np
import sys

OUT = r"e:\bci\results\accesstier"
_buf = []


def emit(s=""):
    _buf.append(str(s))


def summarise(path, knob):
    df = pd.read_csv(path)
    lat = "latency_mean"
    g = df.groupby(knob).agg(
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
        lat_mean=(lat, "mean"),
        lat_std=(lat, "std"),
    ).reset_index()
    g["acc_mean"] *= 100.0
    g["acc_std"] *= 100.0
    return g


def per_dataset(path, knob):
    df = pd.read_csv(path)
    g = df.groupby(["dataset", knob]).agg(
        acc=("accuracy", "mean"),
        lat=("latency_mean", "mean"),
    ).reset_index()
    g["acc"] *= 100.0
    return g


for knob, path in [("alpha", OUT + r"\table-alpha.csv"),
                   ("lambda", OUT + r"\table-lambda.csv")]:
    emit("=" * 60)
    emit("%s SWEEP (pooled over 3 datasets x 15 seeds)" % knob.upper())
    emit("=" * 60)
    g = summarise(path, knob)
    emit(f"{knob:>8}  {'acc%':>8} {'+/-':>6}  {'lat_ms':>8} {'+/-':>6}")
    for _, r in g.iterrows():
        emit(f"{r[knob]:>8.0e}  {r['acc_mean']:>8.2f} {r['acc_std']:>6.2f}  "
             f"{r['lat_mean']:>8.2f} {r['lat_std']:>6.2f}")
    emit()
    emit("per-dataset:")
    pd_g = per_dataset(path, knob)
    piv_a = pd_g.pivot(index=knob, columns="dataset", values="acc")
    piv_l = pd_g.pivot(index=knob, columns="dataset", values="lat")
    emit("  accuracy%:")
    emit(piv_a.to_string(float_format=lambda x: f"{x:6.2f}"))
    emit("  latency ms:")
    emit(piv_l.to_string(float_format=lambda x: f"{x:6.2f}"))
    emit()

with open(OUT + r"\sweep-summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(_buf))
