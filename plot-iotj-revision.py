"""Paper-ready plots from ``run-iotj-revision.py`` outputs."""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "results", "iotj-revision-final")
OUT = DEFAULT_OUT

POLICIES = ["First-k", "Confidence-Stop", "Anytime-cBCI", "Arrival-Only",
            "Throughput-Stop", "Entropy-Gain", "Full-Wait", "AWF"]
COLORS = {
    "First-k": "#4C956C", "Confidence-Stop": "#E07A5F",
    "Anytime-cBCI": "#3D5A80", "Arrival-Only": "#8E6C88",
    "Throughput-Stop": "#577590", "Entropy-Gain": "#D4A72C",
    "Full-Wait": "#6C757D", "AWF": "#C44536",
}
STYLES = {
    "First-k": (0, (1, 1)), "Confidence-Stop": (0, (5, 1)),
    "Anytime-cBCI": ":", "Arrival-Only": "--",
    "Throughput-Stop": (0, (3, 1, 1, 1)), "Entropy-Gain": "-.",
    "Full-Wait": "--", "AWF": "-",
}

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 250, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "font.size": 11,
})


def _save(fig, filename: str) -> None:
    path = os.path.join(OUT, filename)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def deadline_figure() -> None:
    """All policies with 95% seed-level bands, averaged across datasets."""
    df = pd.read_csv(os.path.join(OUT, "deadline-all-policies.csv"))
    # First average the three datasets within each deployment seed, then form
    # the uncertainty band across 15 paired seeds.
    per_seed = (df.groupby(["policy", "deadline_ms", "seed"])
                .agg(accuracy=("accuracy", "mean"),
                     latency=("latency_mean_ms", "mean"))
                .reset_index())
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.5))
    for policy in POLICIES:
        p = per_seed[per_seed.policy == policy]
        if p.empty:
            continue
        grouped = p.groupby("deadline_ms")
        x = np.asarray(sorted(p.deadline_ms.unique()), dtype=float)
        acc = grouped.accuracy.mean().reindex(x).to_numpy() * 100.0
        acc_se = grouped.accuracy.sem().reindex(x).to_numpy() * 100.0
        lat = grouped.latency.mean().reindex(x).to_numpy()
        lat_se = grouped.latency.sem().reindex(x).to_numpy()
        lw = 3.0 if policy == "AWF" else 1.8
        z = 8 if policy == "AWF" else 3
        for ax, y, se in ((axes[0], acc, acc_se),
                          (axes[1], lat, lat_se)):
            ax.plot(x, y, marker="o", ms=6, lw=lw,
                    linestyle=STYLES[policy], color=COLORS[policy],
                    label=policy, zorder=z)
            ax.fill_between(x, y - 1.96 * se, y + 1.96 * se,
                            color=COLORS[policy], alpha=0.08, linewidth=0)
    axes[0].set_ylabel("accuracy (%)")
    axes[1].set_ylabel("mean decision latency (ms)")
    for ax in axes:
        ax.set_xlabel("decision deadline (ms)")
        ax.set_xscale("log")
        ax.set_xticks([20, 50, 100, 200], ["20", "50", "100", "200"])
    axes[0].set_title("(a) Accuracy")
    axes[1].set_title("(b) Decision latency")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "fig-revision-deadline-all.png")


def tuned_tradeoff_figure() -> None:
    df = pd.read_csv(os.path.join(OUT, "tuned-test-summary.csv"))
    seed = (df.groupby(["method", "seed"])
            .agg(accuracy=("accuracy", "mean"),
                 latency=("latency_mean_ms", "mean"))
            .reset_index())
    g = seed.groupby("method").agg(
        accuracy=("accuracy", "mean"), accuracy_se=("accuracy", "sem"),
        latency=("latency", "mean"), latency_se=("latency", "sem"))
    fig, ax = plt.subplots(figsize=(8.2, 6.1))
    for policy in POLICIES:
        if policy not in g.index:
            continue
        row = g.loc[policy]
        ax.errorbar(row.latency, row.accuracy * 100.0,
                    xerr=1.96 * row.latency_se,
                    yerr=1.96 * row.accuracy_se * 100.0,
                    fmt="o", ms=10 if policy == "AWF" else 7,
                    color=COLORS[policy], capsize=3, label=policy,
                    zorder=8 if policy == "AWF" else 3)
    ax.set_xlabel("mean decision latency (ms)")
    ax.set_ylabel("accuracy (%)")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, "fig-revision-tuned-tradeoff.png")


def calibration_figure() -> None:
    df = pd.read_csv(os.path.join(OUT, "calibration-per-user.csv"))
    g = (df.groupby(["dataset", "variant"])[["nll", "ece10"]]
         .mean().reset_index())
    datasets = ["BNCI2014_001", "Zhou2016", "Weibo2014"]
    labels = ["BNCI", "Zhou", "Weibo"]
    x = np.arange(len(datasets)); width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    for i, (variant, label, color) in enumerate(
            (("raw", "Raw", "#A7A9AC"),
             ("temperature", "Temperature-scaled", "#3D5A80"))):
        q = g[g.variant == variant].set_index("dataset").reindex(datasets)
        axes[0].bar(x + (i - 0.5) * width, q.nll, width,
                    color=color, label=label)
        axes[1].bar(x + (i - 0.5) * width, q.ece10, width,
                    color=color, label=label)
    axes[0].set_ylabel("negative log-likelihood")
    axes[1].set_ylabel("ECE (10 bins)")
    for ax, title in zip(axes, ("(a) NLL", "(b) Calibration error")):
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "fig-revision-calibration.png")


def calibration_awf_figure() -> None:
    """Effect of posterior calibration on AWF accuracy, latency, and stops."""
    path = os.path.join(OUT, "calibration-awf-seeds.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    datasets = ["BNCI2014_001", "Zhou2016", "Weibo2014"]
    labels = ["BNCI", "Zhou", "Weibo"]
    variants = [("raw", "Raw", "#A7A9AC"),
                ("temperature", "Temperature-scaled", "#3D5A80")]
    x = np.arange(len(datasets)); width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5))
    for i, (variant, label, color) in enumerate(variants):
        q = (df[df.variant == variant].groupby("dataset")
             .agg(accuracy=("accuracy", "mean"),
                  latency=("latency_mean_ms", "mean"),
                  margin=("reason_margin", "mean"))
             .reindex(datasets))
        pos = x + (i - 0.5) * width
        axes[0].bar(pos, q.accuracy * 100.0, width,
                    color=color, label=label)
        axes[1].bar(pos, q.latency, width, color=color, label=label)
        axes[2].bar(pos, q.margin * 100.0, width, color=color, label=label)
    axes[0].set_ylabel("AWF accuracy (%)")
    axes[1].set_ylabel("AWF decision latency (ms)")
    axes[2].set_ylabel("margin-stop rate (%)")
    for ax, title in zip(axes, ("(a) Accuracy", "(b) Latency",
                                "(c) Stopping behavior")):
        ax.set_xticks(x, labels)
        ax.set_title(title)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    _save(fig, "fig-revision-calibration-awf.png")


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=os.path.relpath(DEFAULT_OUT, HERE))
    args = parser.parse_args()
    OUT = (args.out_dir if os.path.isabs(args.out_dir)
           else os.path.join(HERE, args.out_dir))
    deadline_figure()
    tuned_tradeoff_figure()
    calibration_figure()
    calibration_awf_figure()


if __name__ == "__main__":
    main()
