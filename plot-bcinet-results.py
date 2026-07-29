"""
plot-bcinet-results.py
======================

Generate the figures and summary tables for the cBCI study.

Inputs
------
``--input``  default ``results/summary-results.csv``  (the per-seed table)
``--out-dir`` default ``results/``

Generated figures
-----------------
* ``fig-accuracy-latency.png``      accuracy vs. decision latency scatter
* ``fig-used-response-ratio.png``   bar chart of used-response ratio
* ``fig-network-preset-latency.png``  latency per network preset
* ``fig-stopping-reason.png``        AWF stopping-reason proportions

The script also prints a small console summary so you can sanity-check
the run without opening the CSVs.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Pretty plot defaults
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _short_method(m: str) -> str:
    return m.replace("reliability_", "rel_").replace("deadline_partial", "DP")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_accuracy_latency(df: pd.DataFrame, out_path: str) -> None:
    """Scatter every (method) at its mean (latency, accuracy)."""
    agg = (df.groupby("method")
             .agg(accuracy=("accuracy", "mean"),
                  decision_latency=("decision_latency", "mean"),
                  acc_std=("accuracy", "std"),
                  lat_std=("decision_latency", "std"))
             .reset_index())
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in agg.iterrows():
        ax.errorbar(row["decision_latency"], row["accuracy"],
                    xerr=row["lat_std"], yerr=row["acc_std"],
                    fmt="o", capsize=2, alpha=0.7)
        ax.annotate(_short_method(row["method"]),
                    (row["decision_latency"], row["accuracy"]),
                    xytext=(5, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Decision latency (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. decision-latency tradeoff")
    fig.savefig(out_path)
    plt.close(fig)


def plot_used_response_ratio(df: pd.DataFrame, out_path: str) -> None:
    agg = (df.groupby("method")["used_response_ratio"]
             .agg(["mean", "std"]).reset_index()
             .sort_values("mean"))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(len(agg)), agg["mean"], yerr=agg["std"], capsize=2)
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels([_short_method(m) for m in agg["method"]],
                       rotation=40, ha="right")
    ax.set_ylabel("Used response ratio")
    ax.set_title("Used response ratio by method")
    fig.savefig(out_path)
    plt.close(fig)


def plot_network_preset_latency(df: pd.DataFrame, out_path: str) -> None:
    agg = (df.groupby(["preset", "method"])["decision_latency"]
             .mean().reset_index())
    presets = list(agg["preset"].unique())
    methods = list(agg["method"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / max(len(methods), 1)
    x = np.arange(len(presets))
    for i, m in enumerate(methods):
        ys = [agg[(agg["preset"] == p) & (agg["method"] == m)]
                ["decision_latency"].mean()
              for p in presets]
        ax.bar(x + i * width, ys, width=width, label=_short_method(m))
    ax.set_xticks(x + 0.4)
    ax.set_xticklabels(presets, rotation=20, ha="right")
    ax.set_ylabel("Decision latency (ms)")
    ax.set_title("Latency per network preset")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.savefig(out_path)
    plt.close(fig)


def plot_stopping_reason(df: pd.DataFrame, out_path: str) -> None:
    """AWF (and ablations) stopping-reason composition."""
    awf_methods = [m for m in df["method"].unique() if m.startswith("awf")]
    if not awf_methods:
        return
    cols = [c for c in df.columns if c.startswith("reason_")]
    if not cols:
        return
    agg = df[df["method"].isin(awf_methods)].groupby("method")[cols].mean()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottoms = np.zeros(len(agg))
    for c in cols:
        ax.bar(agg.index, agg[c], bottom=bottoms,
               label=c.replace("reason_", ""))
        bottoms += agg[c].values
    ax.set_ylabel("Fraction of trials")
    ax.set_title("AWF stopping reasons")
    ax.set_xticks(range(len(agg.index)))
    ax.set_xticklabels(agg.index, rotation=15)
    ax.legend(fontsize=7, loc="upper right")
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Plot cBCI experiment results.")
    p.add_argument("--input", default="results/summary-results.csv",
                   help="Input CSV (per-seed summary).")
    p.add_argument("--out-dir", default="results",
                   help="Output directory for figures.")
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"[plot] missing input: {args.input}")
        return 1
    df = pd.read_csv(args.input)
    if df.empty:
        print("[plot] empty results, nothing to plot")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)

    plot_accuracy_latency(df, os.path.join(args.out_dir,
                                           "fig-accuracy-latency.png"))
    plot_used_response_ratio(df, os.path.join(args.out_dir,
                                              "fig-used-response-ratio.png"))
    plot_network_preset_latency(df, os.path.join(args.out_dir,
                                                 "fig-network-preset-latency.png"))
    plot_stopping_reason(df, os.path.join(args.out_dir,
                                          "fig-stopping-reason.png"))

    print("[plot] wrote figures to", args.out_dir)
    print(df.groupby("method")
            [["accuracy", "decision_latency", "used_response_ratio",
              "late_lost_response_ratio", "waiting_reduction"]]
            .mean()
            .round(3)
            .to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
