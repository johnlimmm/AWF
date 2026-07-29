"""Cross-source (cross-dataset) transfer evaluation --- the Level-3 test.

Unlike the held-out split, there is NO train/test division of one source.
The arrival belief is estimated on ONE independent measurement source and the
realised arrivals REPLAY a DIFFERENT independent source, so the two are
independent by construction (different city or different mobility condition):

  5G  : belief from Static  -> deploy on Driving   (and the reverse)
  LEO : belief from Enschede-> deploy on Osnabruck  (and the reverse)   [two cities]

Three policies under identical deployment-source arrivals:
  AWF-Transfer : belief from the OTHER source  (genuine cross-source)
  AWF-Oracle   : belief from the DEPLOYMENT source
  Full-Wait    : waits to the deadline (log-pool reference)

Also records the per-trial stop-time and final-decision agreement between
Transfer and Oracle, so a match is shown to be genuine robustness rather than
a shared input.  Writes table-crosssource.csv, crosssource-summary.txt,
fig-crosssource.png.
"""
import csv
import glob
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "accesstier")
CELLDIR = os.path.join(HERE, "traces", "cellular", "5G-production-dataset")
LEODIR = os.path.join(HERE, "traces", "leo")
CELL_5G = {"5G"}


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


R = _load("runexp", "run-bcinet-experiment.py")
net = R.bcinet_network
data = R.bcinet_data
fusion = R.bcinet_fusion
NetworkProfile = net.NetworkProfile
DATASETS = list(R.DATASETS)
SEEDS = 15
LAM = 1e-3
ALPHA = 1e-3


# ---- raw source parsers (delay pool, per-file sequences, loss) -------------
def cell_source(condition):
    delays, seqs, losses = [], [], []
    for fp in glob.glob(os.path.join(CELLDIR, "**", "*.csv"), recursive=True):
        if os.sep + condition + os.sep not in fp:
            continue
        with open(fp, encoding="utf-8", errors="replace") as f:
            seq = []
            for row in csv.DictReader(f):
                if (row.get("NetworkMode") or "").strip() not in CELL_5G:
                    continue
                pv = row.get("PINGAVG", "-")
                if pv in ("-", "", None):
                    continue
                try:
                    ow = float(pv) / 2.0
                except ValueError:
                    continue
                if not (0 < ow < 5000):
                    continue
                delays.append(ow)
                seq.append(ow)
                pl = row.get("PINGLOSS", "-")
                try:
                    v = float(pl)
                    losses.append(v / 100.0 if v > 1.0 else v)
                except (TypeError, ValueError):
                    pass
            if len(seq) >= 4:
                seqs.append(np.asarray(seq, float))
    pool = np.sort(np.asarray(delays, float))
    loss = float(np.mean(losses)) if losses else 0.02
    return pool, seqs, loss


def leo_source(city):
    fp = os.path.join(LEODIR, f"wetlinks_{city}_net_ping.csv")
    delays, losses = [], []
    with open(fp, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                a = float(row["ping_avg"])
                pl = float(row["packet_loss"])
            except (KeyError, ValueError):
                continue
            if a > 0:
                delays.append(a / 2.0)
                losses.append(min(max(pl, 0.0), 100.0) / 100.0)
    pool = np.sort(np.asarray(delays, float))
    loss = float(np.mean(losses)) if losses else 0.003
    # chop the long ordered sequence into contiguous replay blocks
    seq = np.asarray(delays, float)
    seqs = [seq[i:i + 500] for i in range(0, seq.size, 500) if seq[i:i + 500].size >= 4]
    return pool, seqs, loss


def _profile(tier, belief_pool, loss):
    p10, p90 = float(np.percentile(belief_pool, 10)), float(np.percentile(belief_pool, 90))
    if p90 <= p10:
        p90 = p10 + 1.0
    return NetworkProfile(
        tier=tier, delay_range=(p10, p90), jitter_range=(0.0, 0.0),
        loss_prob=loss, state=None, est_samples=belief_pool, est_loss=loss)


def _replay(seqs, loss, n_users, n_trials, rng):
    arr = np.empty((n_users, n_trials), float)
    for u in range(n_users):
        seq = seqs[int(rng.randint(len(seqs)))]
        start = int(rng.randint(seq.size))
        idx = (start + np.arange(n_trials)) % seq.size
        a = seq[idx].astype(float)
        a[rng.random(n_trials) < loss] = np.inf
        arr[u] = a
    return arr


# ---- scenarios: (label, tier, belief source, deploy source) ---------------
c_static = cell_source("Static")
c_drive = cell_source("Driving")
l_ensch = leo_source("enschede")
l_osnab = leo_source("osnabruck")

SCEN = [
    ("5G Static->Driving", "5g", c_static, c_drive),
    ("5G Driving->Static", "5g", c_drive, c_static),
    ("LEO Enschede->Osnabruck", "leo", l_ensch, l_osnab),
    ("LEO Osnabruck->Enschede", "leo", l_osnab, l_ensch),
]

logpool = fusion.get_method("logpool")
rows = []
for label, tier, (bpool, bseq, bloss), (dpool, dseq, dloss) in SCEN:
    ks = float(ks_2samp(bpool, dpool).statistic)
    prof_tr = _profile(tier, bpool, dloss)   # transfer: belief from source A, deploy loss
    prof_or = _profile(tier, dpool, dloss)   # oracle: belief from deployment source
    for D in (50.0, 100.0):
        for ds in DATASETS:
            try:
                users = R._ensure_users(ds)
            except Exception as exc:
                print("skip", ds, exc, flush=True)
                continue
            val_acc = np.array([u.val_accuracy for u in users])
            confusions = [u.confusion for u in users]
            for seed in range(SEEDS):
                rng = np.random.RandomState(seed)
                P_cube, y = data.align_test_trials(users, rng)
                nu, ntr = len(users), P_cube.shape[1]
                arrivals = _replay(dseq, dloss, nu, ntr, rng)
                profs_tr = [prof_tr] * nu
                profs_or = [prof_or] * nu
                acc = {k: [] for k in ("tr", "or", "fw")}
                lat = {k: [] for k in ("tr", "or", "fw")}
                same_stop = same_dec = 0
                for tt in range(ntr):
                    Pt, at, yt = P_cube[:, tt, :], arrivals[:, tt], int(y[tt])
                    base = dict(P=Pt, arrivals_ms=at, deadline_ms=D,
                                val_acc=val_acc, profiles=profs_tr,
                                confusions=confusions, alpha=ALPHA,
                                lambda_delay=LAM, y_true=yt)
                    rt = fusion.awf_fusion(**base)
                    base_o = dict(base); base_o["profiles"] = profs_or
                    ro = fusion.awf_fusion(**base_o)
                    rf = logpool(**base)
                    for k, r in (("tr", rt), ("or", ro), ("fw", rf)):
                        acc[k].append(int(r.pred == yt))
                        lat[k].append(r.decision_time_ms)
                    same_stop += int(abs(rt.decision_time_ms - ro.decision_time_ms) < 1e-9)
                    same_dec += int(rt.pred == ro.pred)
                row = dict(scenario=label, tier=tier, ks=ks, deadline=D,
                           dataset=ds, seed=seed,
                           stop_agree=same_stop / ntr, dec_agree=same_dec / ntr)
                for k in ("tr", "or", "fw"):
                    row["acc_" + k] = float(np.mean(acc[k])) * 100
                    row["lat_" + k] = float(np.mean(lat[k]))
                rows.append(row)
        print(label, "D=%.0f done" % D, flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "table-crosssource.csv"), index=False)

lines = ["=" * 84,
         "CROSS-SOURCE (CROSS-DATASET) TRANSFER  (%d seeds, 3 tasks)" % SEEDS,
         "Belief estimated on ONE source; arrivals replay a DIFFERENT independent source.",
         "No train/test split --- the two sources are independent by construction.",
         "=" * 84]
for label, tier, _, _ in SCEN:
    d0 = df[df.scenario == label]
    ks = d0["ks"].iloc[0]
    lines.append("")
    lines.append("[%s]  belief-vs-deploy KS = %.3f" % (label, ks))
    lines.append("  %-6s %-12s %8s %8s | agree(stop/dec)" % ("D", "policy", "acc%", "lat"))
    for D in (50.0, 100.0):
        d = d0[d0.deadline == D]
        for k, nm in (("tr", "AWF-Transfer"), ("or", "AWF-Oracle"), ("fw", "Full-Wait")):
            extra = ""
            if k == "tr":
                extra = "  %.1f%% / %.1f%%" % (d["stop_agree"].mean() * 100,
                                               d["dec_agree"].mean() * 100)
            lines.append("  %-6.0f %-12s %8.2f %8.1f%s" % (
                D, nm, d["acc_" + k].mean(), d["lat_" + k].mean(), extra))
        gap = d["acc_tr"].mean() - d["acc_or"].mean()
        Jt = ((100 - d["acc_tr"]) / 100 + LAM * d["lat_tr"]).mean()
        Jo = ((100 - d["acc_or"]) / 100 + LAM * d["lat_or"]).mean()
        lines.append("        Transfer-Oracle acc gap %+.2f pt, regret J %+.4f" % (gap, Jt - Jo))
rep = "\n".join(lines)
print("\n" + rep)
with open(os.path.join(OUT, "crosssource-summary.txt"), "w", encoding="utf-8") as f:
    f.write(rep)

# ---- figure ---------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200,
                     "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})
labels = [s[0] for s in SCEN]
x = np.arange(len(labels)); w = 0.26
fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
D = 100.0
pol = [("tr", "AWF-Transfer", "#d1495b"), ("or", "AWF-Oracle", "#e0a3af"),
       ("fw", "Full-Wait (log)", "#1b998b")]
for i, (k, nm, c) in enumerate(pol):
    vals = [df[(df.scenario == lb) & (df.deadline == D)]["acc_" + k].mean() for lb in labels]
    ax[0].bar(x + (i - 1) * w, vals, w, color=c, label=nm)
ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
ax[0].set_ylabel("accuracy (%)", fontsize=14); ax[0].set_ylim(80, 96)
ax[0].set_title("(a) Cross-source accuracy (D=100 ms)", fontsize=14)
ax[0].legend(fontsize=11, frameon=False, loc="lower right")
for i, (k, nm, c) in enumerate(pol):
    vals = [df[(df.scenario == lb) & (df.deadline == D)]["lat_" + k].mean() for lb in labels]
    ax[1].bar(x + (i - 1) * w, vals, w, color=c, label=nm)
ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
ax[1].set_ylabel("mean decision latency (ms)", fontsize=14)
ax[1].set_title("(b) Cross-source latency (D=100 ms)", fontsize=14)
ax[1].legend(fontsize=11, frameon=False, loc="upper left")
fig.suptitle("Cross-source transfer: belief from one independent source, deployed on another",
             fontsize=14, y=1.02)
fig.tight_layout()
p = os.path.join(OUT, "fig-crosssource.png")
fig.savefig(p); plt.close(fig)
print("\nwrote table-crosssource.csv, crosssource-summary.txt,", p)
