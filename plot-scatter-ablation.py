"""
plot-scatter-ablation.py
========================

Accuracy vs. decision-latency scatter for the AWF ablation study.

Shows that the *full* AWF achieves the best accuracy/latency trade-off
(upper-left) compared with its ablated variants:

    - AWF (full, proposed)
    - AWF w/o arrival-prob    (rho_i(t) = 1)
    - AWF w/o reliability      (uniform confusion matrices)
    - AWF w/o delay-cost       (lambda_delay term removed)

Usage
-----
    python plot-scatter-ablation.py
    python plot-scatter-ablation.py --input results/ablation_uniform/summary-results.csv
    python plot-scatter-ablation.py --out results/fig-scatter-ablation.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Method mapping: CSV name -> display label / marker / color
# (insertion order = legend order; full AWF first)
# ---------------------------------------------------------------------------

METHOD_INFO = {
    "awf":                ("AWF (전체)",          "*", "#d62728"),
    "awf_no_arrival":     ("AWF w/o 도착확률",    "o", "#1f77b4"),
    "awf_no_reliability": ("AWF w/o 신뢰도",      "s", "#2ca02c"),
    "awf_no_delay":       ("AWF w/o 지연비용",    "D", "#ff7f0e"),
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
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.edgecolor": "#444444",
    "font.family": _KR_FONT,
    "font.size": 12,
})


def _agg_seed_sem(df: pd.DataFrame) -> pd.DataFrame:
    """Per-method mean and seed-level standard error (datasets averaged)."""
    per_seed = (df.groupby(["method", "seed"])
                  .agg(accuracy=("accuracy", "mean"),
                       latency=("decision_latency", "mean"))
                  .reset_index())

    def _sem(s):
        return s.std(ddof=1) / np.sqrt(len(s))

    agg = (per_seed.groupby("method")
                   .agg(accuracy=("accuracy", "mean"),
                        acc_sem=("accuracy", _sem),
                        latency=("latency", "mean"),
                        lat_sem=("latency", _sem))
                   .reset_index())
    return agg


def plot_scatter(df: pd.DataFrame, out_path: str) -> None:
    df = df[df["method"].isin(METHOD_INFO.keys())].copy()
    if df.empty:
        print("[ablation] no matching methods in data – check CSV")
        return

    agg = _agg_seed_sem(df)
    agg["accuracy"] *= 100.0
    agg["acc_sem"] *= 100.0
    info = agg.set_index("method")

    fig, ax = plt.subplots(figsize=(7.4, 5.2))

    # --- knee curve through the variants ordered by latency ---------------
    order = [m for m in ("awf_no_reliability", "awf",
                         "awf_no_delay", "awf_no_arrival")
             if m in info.index]
    if len(order) >= 2:
        fx = [info.loc[m, "latency"] for m in order]
        fy = [info.loc[m, "accuracy"] for m in order]
        ax.plot(fx, fy, "-", color="#9aa0a6", lw=1.6, alpha=0.7, zorder=1)

    legend_handles = []
    for m, (label, marker, color) in METHOD_INFO.items():
        if m not in info.index:
            continue
        row = info.loc[m]
        size = 360 if marker == "*" else 130

        ax.errorbar(
            row["latency"], row["accuracy"],
            xerr=row["lat_sem"], yerr=row["acc_sem"],
            fmt="none", ecolor=color, elinewidth=1.2,
            capsize=3, capthick=1.0, alpha=0.7, zorder=2,
        )
        sc = ax.scatter(
            row["latency"], row["accuracy"],
            s=size, marker=marker, color=color,
            zorder=4, edgecolors="white", linewidths=1.0,
            label=label,
        )
        legend_handles.append(sc)

    if "awf" in info.index:
        a = info.loc["awf"]
        ax.annotate("무릎(knee)", xy=(a["latency"], a["accuracy"]),
                    xytext=(a["latency"] + 5, a["accuracy"] - 4.5),
                    fontsize=10, fontweight="bold", color="#d62728",
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))

    ax.legend(
        handles=legend_handles,
        fontsize=10, loc="lower right",
        framealpha=0.95, edgecolor="#cccccc",
    )

    ax.set_xlabel("판단 지연 (ms)", fontsize=12)
    ax.set_ylabel("정확도 (%)", fontsize=12)
    ax.set_title("AWF 구성요소 제거 실험 (정확도 – 지연 트레이드오프)",
                 fontsize=13, pad=12)
    ax.set_xlim(left=0)
    ax.margins(y=0.14)

    ax.annotate("← 빠른 판단", xy=(0.035, 0.10),
                xycoords="axes fraction", fontsize=9, color="gray", ha="left")
    ax.annotate("↑ 높은 정확도", xy=(0.035, 0.16),
                xycoords="axes fraction", fontsize=9, color="gray", ha="left")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[ablation] saved → {out_path}")

    print("\n── AWF ablation 요약 ──")
    for _, row in agg.sort_values("latency").iterrows():
        label = METHOD_INFO[row["method"]][0]
        print(f"  {label:<20} acc={row['accuracy']:.2f}%  "
              f"lat={row['latency']:.1f} ms")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input",
                   default="results/ablation_uniform/summary-results.csv")
    p.add_argument("--preset", default="uniform_all",
                   help="Filter to a network preset. Empty string = all.")
    p.add_argument("--out", default="results/fig-scatter-ablation.png")
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"[ablation] input not found: {args.input}")
        return 1

    df = pd.read_csv(args.input)
    if args.preset:
        df = df[df["preset"] == args.preset]
        if df.empty:
            print(f"[ablation] preset '{args.preset}' not found in data")
            return 1
        print(f"[ablation] using preset='{args.preset}', {len(df)} rows")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_scatter(df, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
