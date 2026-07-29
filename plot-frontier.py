"""
plot-frontier.py
================

Accuracy--latency operating-point frontier. Each baseline is swept across
its *own* control knob and the resulting curve is drawn on the
accuracy(y)--latency(x) plane, so the comparison no longer rests on a
single operating point:

    * AWF          -- swept over the delay weight lambda,
    * Earliest-k   -- swept over k = 1..4,
    * Anytime cBCI -- swept over the confidence threshold theta,
    * WeightedSum  -- single full-wait point,
    * Fastest-only -- single point.

The narrative: the AWF lambda-frontier sits up and to the left of every
baseline knob-curve across *all* operating points, i.e. it Pareto-dominates
them not just at one tuning but along the entire trade-off.

Reads results/frontier/frontier-raw.csv produced by run-frontier-sweep.py
(schema: knob,value,label,dataset,seed,accuracy,decision_latency).

Usage
-----
    python plot-frontier.py
    python plot-frontier.py --csv results/frontier/frontier-raw.csv
    python plot-frontier.py --out results/fig-frontier.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))

# knob key -> (display label, marker, color, is_curve)
KNOB_INFO = {
    "awf":          ("AWF (제안, λ 스윕)",     "*", "#d62728", True),
    "anytime":      ("Anytime cBCI (θ 스윕)",  "^", "#e377c2", True),
    "earliest_k":   ("Earliest-k (k 스윕)",    "s", "#ff7f0e", True),
    "weighted":     ("WeightedSum",            "D", "#1f77b4", False),
    "fastest_only": ("Fastest-only",           "v", "#9467bd", False),
}


def _korean_font() -> str:
    candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "UnDotum"]
    available = {f.name for f in _fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    return "DejaVu Sans"


_KR_FONT = _korean_font()

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": _KR_FONT,
    "font.size": 12,
})


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- SEM of accuracy and latency per (knob, value), averaged over
    the dataset x seed grid."""
    rows = []
    for knob in KNOB_INFO:
        sub = df[df["knob"] == knob]
        if sub.empty:
            continue
        # ``value`` may be NaN for single-point methods; treat them as one
        # group so groupby keeps the row.
        grouper = sub["value"].fillna(-1.0)
        for val, g in sub.groupby(grouper):
            n = len(g)
            rows.append({
                "knob": knob,
                "value": np.nan if val == -1.0 else val,
                "accuracy": g["accuracy"].mean(),
                "acc_sem": g["accuracy"].std(ddof=1) / np.sqrt(n) if n > 1 else 0.0,
                "latency": g["decision_latency"].mean(),
                "lat_sem": g["decision_latency"].std(ddof=1) / np.sqrt(n) if n > 1 else 0.0,
            })
    return pd.DataFrame(rows)


def plot_frontier(agg: pd.DataFrame, out_path: str,
                  title: str = "정확도--지연 작동점 프론티어 (uniform_all, 6개 데이터셋 평균)") -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    for knob, (label, marker, color, is_curve) in KNOB_INFO.items():
        sub = agg[agg["knob"] == knob].sort_values("latency")
        if sub.empty:
            continue
        ms = 15 if marker == "*" else 9
        if is_curve:
            ax.errorbar(sub["latency"], sub["accuracy"] * 100,
                        xerr=sub["lat_sem"], yerr=sub["acc_sem"] * 100,
                        marker=marker, markersize=ms, color=color,
                        capsize=3, linewidth=1.8, label=label, zorder=3)
        else:
            ax.errorbar(sub["latency"], sub["accuracy"] * 100,
                        xerr=sub["lat_sem"], yerr=sub["acc_sem"] * 100,
                        marker=marker, markersize=ms, color=color,
                        capsize=3, linestyle="none", label=label, zorder=4)

    ax.set_xlabel("평균 판단 지연 (ms)")
    ax.set_ylabel("정확도 (%)")
    ax.set_title(title,
                 fontsize=12)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9,
              edgecolor="#cccccc")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[frontier] saved → {out_path}")

    print("\n── 작동점 프론티어 요약 ──")
    for knob, (label, _, _, _) in KNOB_INFO.items():
        sub = agg[agg["knob"] == knob].sort_values("latency")
        if sub.empty:
            continue
        pts = ", ".join(
            f"{r.accuracy*100:.1f}%@{r.latency:.1f}ms" for r in sub.itertuples())
        print(f"  {label}: {pts}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv",
                   default=os.path.join(_HERE, "results", "frontier",
                                        "frontier-raw.csv"))
    p.add_argument("--out",
                   default=os.path.join(_HERE, "results", "fig-frontier.png"))
    p.add_argument("--title",
                   default="정확도--지연 작동점 프론티어 (uniform_all, 6개 데이터셋 평균)")
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv)
    agg = aggregate(df)
    plot_frontier(agg, args.out, args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
