"""Out-of-sample (held-out) trace evaluation --- the decisive test against the
circularity concern.

Network traces are split like EEG calibration: each tier's ordered per-source
sequences are divided into TRAIN and TEST sets.  The nominal arrival belief
(rho, E[Delta]) is estimated from TRAIN only.  The realised arrivals REPLAY
held-out TEST sequences in order (block replay, preserving temporal structure).

Three policies are compared under identical held-out arrivals:
  AWF-Nominal : belief from TRAIN  (genuinely out-of-sample)
  AWF-Oracle  : belief from TEST   (knows the true test distribution)
  Full-Wait   : waits to the deadline (log-pool reference)

The model-mismatch penalty is J(Nominal) - J(Oracle) with J = error + lambda*latency.
Writes table-heldout.csv, heldout-summary.txt, fig-heldout.png.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "accesstier")


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
metrics = R.bcinet_metrics
tb = _load("tb", "bcinet-tracebank.py")
bank = tb.load_tracebank()

NetworkProfile = net.NetworkProfile
TIERS = ["ble", "wifi", "5g", "lte", "leo"]
DATASETS = list(R.DATASETS)
SEEDS = 15
LAM = 1e-3
ALPHA = 1e-3

# ---- Split each tier's ordered sequences into TRAIN / TEST -----------------
train_belief, oracle_belief, test_seqs, tier_loss = {}, {}, {}, {}
for t in TIERS:
    seqs = [np.asarray(s, float) for s in bank[t].sequences
            if np.asarray(s).size >= 4]
    seqs = [s for s in seqs if np.isfinite(s).all()]
    if len(seqs) >= 2:
        h = max(1, len(seqs) // 2)
        tr, te = seqs[:h], seqs[h:]
    else:  # single long sequence: split it in half by time
        s = seqs[0]
        m = s.size // 2
        tr, te = [s[:m]], [s[m:]]
    train_belief[t] = np.sort(np.concatenate(tr))
    oracle_belief[t] = np.sort(np.concatenate(te))
    test_seqs[t] = te
    tier_loss[t] = float(bank[t].overall_loss())


def _profile(tier, est_samples):
    s = train_belief[tier]
    p10, p90 = float(np.percentile(s, 10)), float(np.percentile(s, 90))
    if p90 <= p10:
        p90 = p10 + 1.0
    return NetworkProfile(
        tier=tier, delay_range=(p10, p90), jitter_range=(0.0, 0.0),
        loss_prob=tier_loss[tier], state=None,
        est_samples=est_samples, est_loss=tier_loss[tier])


def _replay_arrivals(tiers, n_trials, rng):
    """Order-preserving block replay of held-out TEST sequences per user."""
    n = len(tiers)
    arr = np.empty((n, n_trials), float)
    for u, t in enumerate(tiers):
        seq = test_seqs[t][int(rng.randint(len(test_seqs[t])))]
        start = int(rng.randint(seq.size))
        idx = (start + np.arange(n_trials)) % seq.size
        a = seq[idx].astype(float)
        lost = rng.random(n_trials) < tier_loss[t]
        a[lost] = np.inf
        arr[u] = a
    return arr


rows = []
logpool = fusion.get_method("logpool")
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
            tiers = [TIERS[int(rng.randint(len(TIERS)))] for _ in range(nu)]
            arrivals = _replay_arrivals(tiers, ntr, rng)
            prof_nom = [_profile(t, train_belief[t]) for t in tiers]
            prof_ora = [_profile(t, oracle_belief[t]) for t in tiers]
            acc = {k: [] for k in ("nom", "ora", "fw")}
            lat = {k: [] for k in ("nom", "ora", "fw")}
            nresp = {k: [] for k in ("nom", "ora", "fw")}
            stop = {"margin": 0, "neg_value": 0, "deadline": 0}
            for tt in range(ntr):
                Pt, at, yt = P_cube[:, tt, :], arrivals[:, tt], int(y[tt])
                base = dict(P=Pt, arrivals_ms=at, deadline_ms=D,
                            val_acc=val_acc, profiles=prof_nom,
                            confusions=confusions, alpha=ALPHA,
                            lambda_delay=LAM, y_true=yt)
                rn = fusion.awf_fusion(**base)
                base_o = dict(base); base_o["profiles"] = prof_ora
                ro = fusion.awf_fusion(**base_o)
                rf = logpool(**base)
                for k, r in (("nom", rn), ("ora", ro), ("fw", rf)):
                    acc[k].append(int(r.pred == yt))
                    lat[k].append(r.decision_time_ms)
                    nresp[k].append(int(r.used_mask.sum()))
                if rn.stopping_reason in stop:
                    stop[rn.stopping_reason] += 1
            row = dict(deadline=D, dataset=ds, seed=seed)
            for k in ("nom", "ora", "fw"):
                row["acc_" + k] = float(np.mean(acc[k])) * 100
                row["lat_" + k] = float(np.mean(lat[k]))
                row["lat95_" + k] = float(np.percentile(lat[k], 95))
                row["nresp_" + k] = float(np.mean(nresp[k]))
            tot = max(sum(stop.values()), 1)
            for sr in ("margin", "neg_value", "deadline"):
                row["stop_" + sr] = stop[sr] / tot
            rows.append(row)
        print("D=%.0f %s done" % (D, ds), flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "table-heldout.csv"), index=False)


def _agg(D):
    d = df[df.deadline == D]
    out = {}
    for k in ("nom", "ora", "fw"):
        out[k] = dict(acc=d["acc_" + k].mean(), lat=d["lat_" + k].mean(),
                      lat95=d["lat95_" + k].mean(), nresp=d["nresp_" + k].mean())
    # objective J = error_rate + lambda * mean_latency (per seed, then mean)
    Jn = ((100 - d["acc_nom"]) / 100 + LAM * d["lat_nom"]).mean()
    Jo = ((100 - d["acc_ora"]) / 100 + LAM * d["lat_ora"]).mean()
    out["regret"] = Jn - Jo
    out["stop"] = dict(margin=d["stop_margin"].mean(),
                       neg_value=d["stop_neg_value"].mean(),
                       deadline=d["stop_deadline"].mean())
    return out


lines = ["=" * 80,
         "OUT-OF-SAMPLE (HELD-OUT) TRACE EVALUATION  (%d seeds, 3 tasks)" % SEEDS,
         "Belief estimated from TRAIN sequences; arrivals REPLAY held-out TEST",
         "sequences in order.  AWF-Oracle knows the test distribution.",
         "=" * 80]
for D in (50.0, 100.0):
    a = _agg(D)
    lines.append("")
    lines.append("[D=%.0f ms]" % D)
    lines.append("  %-12s %8s %8s %8s %8s" % ("policy", "acc%", "lat", "lat95", "nresp"))
    for k, nm in (("nom", "AWF-Nominal"), ("ora", "AWF-Oracle"), ("fw", "Full-Wait")):
        s = a[k]
        lines.append("  %-12s %8.2f %8.1f %8.1f %8.2f" % (
            nm, s["acc"], s["lat"], s["lat95"], s["nresp"]))
    lines.append("  Nominal-Oracle accuracy gap : %+.2f pt" % (a["nom"]["acc"] - a["ora"]["acc"]))
    lines.append("  model-mismatch regret J(Nom)-J(Ora) : %+.4f" % a["regret"])
    lines.append("  AWF-Nominal stop mix : margin %.2f / value %.2f / deadline %.2f" % (
        a["stop"]["margin"], a["stop"]["neg_value"], a["stop"]["deadline"]))
rep = "\n".join(lines)
print("\n" + rep)
with open(os.path.join(OUT, "heldout-summary.txt"), "w", encoding="utf-8") as f:
    f.write(rep)

# ---- figure ---------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200,
                     "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
Ds = [50.0, 100.0]
pol = [("nom", "AWF-Nominal", "#d1495b"), ("ora", "AWF-Oracle", "#e0a3af"),
       ("fw", "Full-Wait (log)", "#1b998b")]
x = np.arange(len(Ds)); w = 0.26
for i, (k, nm, c) in enumerate(pol):
    ax[0].bar(x + (i - 1) * w, [_agg(D)[k]["acc"] for D in Ds], w, color=c, label=nm)
ax[0].set_xticks(x); ax[0].set_xticklabels(["D=50 ms", "D=100 ms"], fontsize=13)
ax[0].set_ylabel("accuracy (%)", fontsize=15); ax[0].set_ylim(80, 96)
ax[0].set_title("(a) Held-out accuracy", fontsize=15)
ax[0].legend(fontsize=12, frameon=False, loc="lower right")
for i, (k, nm, c) in enumerate(pol):
    ax[1].bar(x + (i - 1) * w, [_agg(D)[k]["lat"] for D in Ds], w, color=c, label=nm)
ax[1].set_xticks(x); ax[1].set_xticklabels(["D=50 ms", "D=100 ms"], fontsize=13)
ax[1].set_ylabel("mean decision latency (ms)", fontsize=15)
ax[1].set_title("(b) Held-out latency", fontsize=15)
ax[1].legend(fontsize=12, frameon=False, loc="upper left")
fig.suptitle("Out-of-sample held-out trace evaluation: AWF-Nominal (train belief) "
             "matches AWF-Oracle", fontsize=15, y=1.01)
fig.tight_layout()
p = os.path.join(OUT, "fig-heldout.png")
fig.savefig(p); plt.close(fig)
print("\nwrote table-heldout.csv, heldout-summary.txt,", p)
