"""
plot-deadline-sweep.py
======================

Deadline sensitivity analysis. Sweeps the decision deadline D from
20 ms to 100 ms and plots, for each method, how mean accuracy and mean
decision latency respond.

Key narrative: the adaptive methods (AWF, Anytime cBCI) keep their
*actual* decision latency well below the deadline and saturate in
accuracy, whereas the fixed full-wait policy (WeightedSum) pays a
latency that grows linearly with the deadline.

Reads one summary-results.csv per deadline from
    results/deadline_sweep/d{D}/summary-results.csv

Usage
-----
    python plot-deadline-sweep.py
    python plot-deadline-sweep.py --root results/deadline_sweep
    python plot-deadline-sweep.py --out results/fig-deadline-sweep.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import numpy as np
import pandas as pd

# CSV name -> (display label, marker, color)
# insertion order = legend order (AWF first)
METHOD_INFO = {
    "awf":          ("AWF (제안)",       "*", "#d62728"),
    "weighted":     ("WeightedSum",      "D", "#1f77b4"),
    "fastest_only": ("Fastest-only",     "v", "#9467bd"),
    "earliest_k:3": ("Earliest-k (k=3)", "s", "#ff7f0e"),
    "anytime_cbci": ("Anytime cBCI",     "^", "#e377c2"),
}

DEADLINES = [20, 30, 40, 50, 60, 70, 80, 90, 100]

# Additional result roots to merge in (methods run separately).
# Resolve relative to this script's location so cwd does not matter.
_HERE = os.path.dirname(os.path.abspath(__file__))
# The unified driver (run-frontier-sweep.py) regenerates every method for
# every deadline through the same run_one as the main table, so no separate
# "extra" roots are needed (and merging them would re-introduce the
# Table 3 / Figure 5 numeric wobble).
EXTRA_ROOTS: list[str] = []


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


def _read_deadline(root: str, d: int, preset: str) -> pd.DataFrame:
    path = os.path.join(root, f"d{d}", "summary-results.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if preset:
        df = df[df["preset"] == preset]
    return df


def load_sweep(root: str, preset: str) -> pd.DataFrame:
    """Collect per-deadline summaries (merging extra roots) into long form."""
    rows = []
    for d in DEADLINES:
        frames = [_read_deadline(root, d, preset)]
        for extra in EXTRA_ROOTS:
            frames.append(_read_deadline(extra, d, preset))
        df = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
            if any(not f.empty for f in frames) else pd.DataFrame()
        if df.empty:
            print(f"[deadline] no data for D={d} – skipping")
            continue
        for m in METHOD_INFO:
            sub = df[df["method"] == m]
            if sub.empty:
                continue
            rows.append({
                "deadline": d,
                "method": m,
                "accuracy": sub["accuracy"].mean(),
                "acc_sem": sub["accuracy"].std() / np.sqrt(len(sub)),
                "latency": sub["decision_latency"].mean(),
                "lat_sem": sub["decision_latency"].std() / np.sqrt(len(sub)),
            })
    return pd.DataFrame(rows)


def plot_sweep(agg: pd.DataFrame, out_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    for m, (label, marker, color) in METHOD_INFO.items():
        sub = agg[agg["method"] == m].sort_values("deadline")
        if sub.empty:
            continue
        ms = 13 if marker == "*" else 8
        # Left: accuracy vs deadline
        ax1.errorbar(sub["deadline"], sub["accuracy"] * 100,
                     yerr=sub["acc_sem"] * 100,
                     marker=marker, markersize=ms, color=color,
                     capsize=3, linewidth=1.8, label=label)
        # Right: latency vs deadline
        ax2.errorbar(sub["deadline"], sub["latency"],
                     yerr=sub["lat_sem"],
                     marker=marker, markersize=ms, color=color,
                     capsize=3, linewidth=1.8, label=label)

    # y=x reference on the latency panel (full-wait upper bound)
    ax2.plot(DEADLINES, DEADLINES, color="gray", linestyle=":",
             linewidth=1.2, label="마감 = 지연 (상한)")

    ax1.set_xlabel("마감 시간 D (ms)")
    ax1.set_ylabel("정확도 (%)")
    ax1.set_ylim(50, 100)
    ax1.set_title("(a) 마감 시간에 따른 정확도")
    ax1.legend(fontsize=8, loc="lower right", framealpha=0.9,
               edgecolor="#cccccc", ncol=2)

    ax2.set_xlabel("마감 시간 D (ms)")
    ax2.set_ylabel("평균 판단 지연 (ms)")
    ax2.set_title("(b) 마감 시간에 따른 판단 지연")
    ax2.legend(fontsize=8, loc="upper left", framealpha=0.9,
               edgecolor="#cccccc", ncol=2)

    fig.suptitle("마감 시간(deadline) 민감도 분석 (uniform_all, 6개 데이터셋 평균)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[deadline] saved → {out_path}")

    print("\n── 마감 민감도 요약 ──")
    for m, (label, _, _) in METHOD_INFO.items():
        sub = agg[agg["method"] == m].sort_values("deadline")
        if sub.empty:
            continue
        print(f"\n  {label}")
        for _, r in sub.iterrows():
            print(f"    D={int(r['deadline']):3d}ms  "
                  f"acc={r['accuracy']:.4f}  lat={r['latency']:.1f}ms")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/deadline_sweep")
    p.add_argument("--preset", default="uniform_all")
    p.add_argument("--out", default="results/fig-deadline-sweep.png")
    args = p.parse_args()

    agg = load_sweep(args.root, args.preset)
    if agg.empty:
        print("[deadline] no data found – run the deadline sweep first")
        return 1
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_sweep(agg, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
