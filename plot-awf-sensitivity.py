"""
plot-awf-sensitivity.py
=======================
Hyperparameter sensitivity analysis for AWF on the ``uniform_all`` preset.

Two one-at-a-time sweeps over the six cached EEG datasets:

* lambda sweep : delay-cost coefficient lambda varied with alpha fixed.
* alpha  sweep : margin threshold alpha varied with lambda fixed.

For every configuration we report mean accuracy and mean decision latency
(averaged over datasets x seeds). Results are written to CSV and a
two-panel accuracy/latency figure is saved for the paper.

Usage:
    python plot-awf-sensitivity.py
"""
import importlib.util, os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
from matplotlib.ticker import FuncFormatter, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))


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
    "axes.unicode_minus": False,
    "font.family": _KR_FONT,
    "font.size": 12,
    "mathtext.fontset": "cm",
})


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


net = _load("bcinet_network", "bcinet-network.py")
fus = _load("bcinet_fusion", "bcinet-fusion.py")
met = _load("bcinet_metrics", "bcinet-metrics.py")
data = _load("bcinet_data", "bcinet-data.py")

PRESET = "uniform_all"
DEADLINE = net.DEFAULT_DEADLINE_MS  # 100 ms
SEEDS = 30
N_TRIALS = 200

# Default (tuned) operating point.
ALPHA0 = 1e-5
LAMBDA0 = 1e-5

# One-at-a-time sweep grids.
LAMBDAS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
ALPHAS = [1e-5, 1e-3, 1e-2, 3e-2, 1e-1, 2e-1]


# --- read sweeps produced by run-frontier-sweep.py -------------------------
# The unified, deterministic driver (run-frontier-sweep.py) regenerates these
# one-at-a-time sweeps through the *same* run_one used for the main table, so
# the left endpoints (lambda = alpha = 1e-5) coincide exactly with the AWF
# operating point in Table 3 (no Table 3 / figure wobble). This script now
# only reads those CSVs and draws the figure.
out_dir = os.path.join(HERE, "results")
agg_lam = pd.read_csv(os.path.join(out_dir, "table-awf-sensitivity-lambda.csv"))
agg_alp = pd.read_csv(os.path.join(out_dir, "table-awf-sensitivity-alpha.csv"))
print("=== lambda sweep (alpha=1e-5) ===")
print(agg_lam.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("=== alpha sweep (lambda=1e-5) ===")
print(agg_alp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# --- figure -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))

for ax, agg, xs, xlabel, title in [
    (axes[0], agg_lam, LAMBDAS, r"지연 비용 계수 $\lambda$",
     r"(a) 지연 비용 $\lambda$ 스윕 ($\alpha=10^{-5}$)"),
    (axes[1], agg_alp, ALPHAS, r"목표 오류 수준 $\alpha$",
     r"(b) 목표 오류 수준 $\alpha$ 스윕 ($\lambda=10^{-5}$)"),
]:
    a = agg.sort_values("value")
    acc = a["accuracy"].values * 100.0
    sem = a["accuracy_sem"].values * 100.0
    lat = a["decision_latency"].values

    ln1 = ax.errorbar(a["value"], acc, yerr=sem, marker="o", color="C0",
                      capsize=2, label="정확도")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("정확도 (%)", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.set_ylim(80, 95)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _p: r"$10^{%d}$" % round(np.log10(x))))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax2 = ax.twinx()
    ln2, = ax2.plot(a["value"], lat, marker="s", color="C3",
                    label="판단 지연")
    ax2.set_ylabel("판단 지연 (ms)", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax2.set_ylim(0, 45)

fig.tight_layout()
out_png = os.path.join(out_dir, "fig-awf-sensitivity.png")
fig.savefig(out_png, dpi=150)
print(f"\nSaved figure -> {out_png}")
