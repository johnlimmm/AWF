"""
plot-scatter-selected.py
========================

Accuracy vs. decision-latency scatter for the five methods discussed
in the patent embodiment section:

    - LogPool                  (full-wait fusion)
    - WeightedSum              (full-wait fusion)
    - Anytime cBCI             (earliest-k early-stopping proxy)
    - Reliability-Delay        (reliability_latency_score_selection)
    - AWF (proposed)

Usage
-----
    python plot-scatter-selected.py
    python plot-scatter-selected.py --input results/summary-results.csv
    python plot-scatter-selected.py --out results/fig-scatter-selected.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Method mapping: CSV name -> display label / group
# ---------------------------------------------------------------------------

METHOD_INFO = {
    # (display_label, marker, color)
    "awf":           ("AWF (제안)",        "*", "#d62728"),
    "weighted":      ("WeightedSum",      "D", "#1f77b4"),
    "fastest_only":  ("Fastest-only",     "v", "#9467bd"),
    "earliest_k:3":  ("Earliest-k (k=3)", "s", "#ff7f0e"),
    "anytime_cbci":  ("Anytime cBCI",     "^", "#e377c2"),
}

import matplotlib.font_manager as _fm

def _korean_font() -> str:
    """Return the first available Korean-capable font name."""
    candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "UnDotum"]
    available = {f.name for f in _fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    return "DejaVu Sans"  # fallback (Korean glyphs may be missing)

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
    """Per-method mean and seed-level standard error.

    Each seed's value is first averaged over datasets, so the reported
    error bar is the standard error across seeds (matching the caption),
    not the much larger between-dataset spread.
    """
    per_seed = (df.groupby(["method", "seed"])
                  .agg(accuracy=("accuracy", "mean"),
                       latency=("decision_latency", "mean"))
                  .reset_index())
    n_seed = per_seed.groupby("method")["seed"].nunique()

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
    # Keep only the five target methods
    df = df[df["method"].isin(METHOD_INFO.keys())].copy()
    if df.empty:
        print("[scatter] no matching methods in data – check CSV")
        return

    agg = _agg_seed_sem(df)
    agg["accuracy"] *= 100.0
    agg["acc_sem"] *= 100.0
    info = agg.set_index("method")

    fig, ax = plt.subplots(figsize=(7.4, 5.2))

    # --- achievable trade-off frontier: Fastest -> AWF -> WeightedSum ------
    frontier = [m for m in ("fastest_only", "awf", "weighted")
                if m in info.index]
    if len(frontier) >= 2:
        fx = [info.loc[m, "latency"] for m in frontier]
        fy = [info.loc[m, "accuracy"] for m in frontier]
        ax.plot(fx, fy, "-", color="#9aa0a6", lw=1.6, alpha=0.7,
                zorder=1, label="_frontier")
        ax.annotate("트레이드오프 경계", xy=(fx[1], fy[1]),
                    xytext=(fx[1] + 6, fy[1] - 7.5),
                    fontsize=9, color="#7a7f85",
                    arrowprops=dict(arrowstyle="-", color="#bcbfc4", lw=0.8))

    legend_handles = []
    for m, (label, marker, color) in METHOD_INFO.items():
        if m not in info.index:
            continue
        row = info.loc[m]
        size = 320 if marker == "*" else 130

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

    # Highlight + annotate the AWF operating point
    if "awf" in info.index:
        a = info.loc["awf"]
        ax.annotate(f"{a['accuracy']:.1f}% @ {a['latency']:.0f} ms",
                    xy=(a["latency"], a["accuracy"]),
                    xytext=(a["latency"] - 1, a["accuracy"] + 4.2),
                    fontsize=10, fontweight="bold", color="#d62728",
                    ha="center")

    ax.legend(
        handles=legend_handles,
        fontsize=10, loc="lower right",
        framealpha=0.95, edgecolor="#cccccc",
    )

    ax.set_xlabel("판단 지연 (ms)", fontsize=12)
    ax.set_ylabel("정확도 (%)", fontsize=12)
    ax.set_title("정확도 – 판단 지연 트레이드오프", fontsize=14, pad=12)
    ax.set_xlim(left=0)
    ax.margins(y=0.12)

    ax.annotate("← 빠른 판단",   xy=(0.02, 0.45), xycoords="axes fraction",
                fontsize=9, color="gray", ha="left")
    ax.annotate("↑ 높은 정확도", xy=(0.02, 0.51), xycoords="axes fraction",
                fontsize=9, color="gray", ha="left")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[scatter] saved → {out_path}")

    # Console summary
    print("\n── 선택 기법 요약 ──")
    for _, row in agg.sort_values("latency").iterrows():
        label = METHOD_INFO[row["method"]][0]
        print(f"  {label:<22} acc={row['accuracy']:.2f}%  "
              f"lat={row['latency']:.1f} ms")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/uniform_all/summary-results.csv")
    p.add_argument("--preset", default="uniform_all",
                   help="Filter to a specific network preset. Empty string = use all.")
    p.add_argument("--out", default="results/fig-scatter-selected.png")
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"[scatter] input not found: {args.input}")
        return 1

    df = pd.read_csv(args.input)
    if args.preset:
        df = df[df["preset"] == args.preset]
        if df.empty:
            print(f"[scatter] preset '{args.preset}' not found in data")
            return 1
        print(f"[scatter] using preset='{args.preset}', {len(df)} rows")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_scatter(df, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
