"""Diagnostic for the held-out result: is AWF-Nominal == AWF-Oracle a genuine
robustness finding or a bug (same distribution fed to both)?

Checks:
  (1) train vs test belief arrays are actually different (size, mean, KS).
  (2) the rho_j(t) and E[Delta_j(t)] that AWF actually reads differ between
      the nominal and oracle profiles.
  (3) trial-level stop-time agreement and final-decision agreement.
  (4) full-precision accuracy / latency differences.
"""
import importlib.util
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "accesstier")


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


R = _load("runexp", "run-bcinet-experiment.py")
net = R.bcinet_network
data = R.bcinet_data
fusion = R.bcinet_fusion
metrics = R.bcinet_metrics
tb = _load("tb", "bcinet-tracebank.py")
bank = tb.load_tracebank()
NetworkProfile = net.NetworkProfile
arrival_probability = net.arrival_probability

TIERS = ["ble", "wifi", "5g", "lte", "leo"]
LABEL = {"ble": "BLE", "wifi": "Wi-Fi", "5g": "5G", "lte": "4G", "leo": "LEO"}
DATASETS = list(R.DATASETS)
SEEDS = 15
LAM = ALPHA = 1e-3

# ---- Split identically to run_heldout.py ----------------------------------
train_belief, oracle_belief, test_seqs, tier_loss = {}, {}, {}, {}
n_tr_seq, n_te_seq = {}, {}
for t in TIERS:
    seqs = [np.asarray(s, float) for s in bank[t].sequences
            if np.asarray(s).size >= 4 and np.isfinite(np.asarray(s)).all()]
    if len(seqs) >= 2:
        h = max(1, len(seqs) // 2); tr, te = seqs[:h], seqs[h:]
    else:
        s = seqs[0]; m = s.size // 2; tr, te = [s[:m]], [s[m:]]
    train_belief[t] = np.sort(np.concatenate(tr))
    oracle_belief[t] = np.sort(np.concatenate(te))
    test_seqs[t] = te; tier_loss[t] = float(bank[t].overall_loss())
    n_tr_seq[t] = len(tr); n_te_seq[t] = len(te)

out = []
def w(s=""):
    out.append(s); print(s, flush=True)

w("=" * 78)
w("HELD-OUT DIAGNOSTIC")
w("=" * 78)
w("\n[1] Are the train and test belief arrays actually different?")
w("%-6s %8s %8s %9s %9s %7s %10s" % ("tier", "#tr_seq", "#te_seq",
                                     "mean_tr", "mean_te", "KS", "identical?"))
for t in TIERS:
    a, b = train_belief[t], oracle_belief[t]
    ident = (a.size == b.size) and np.array_equal(a, b)
    ks = stats.ks_2samp(a, b).statistic
    w("%-6s %8d %8d %9.2f %9.2f %7.3f %10s" % (
        LABEL[t], n_tr_seq[t], n_te_seq[t], a.mean(), b.mean(), ks,
        "YES(BUG)" if ident else "no"))


def _profile(tier, est):
    s = train_belief[tier]
    p10, p90 = float(np.percentile(s, 10)), float(np.percentile(s, 90))
    if p90 <= p10:
        p90 = p10 + 1.0
    return NetworkProfile(tier=tier, delay_range=(p10, p90),
                          jitter_range=(0.0, 0.0), loss_prob=tier_loss[tier],
                          state=None, est_samples=est, est_loss=tier_loss[tier])


w("\n[2] rho_j(t)=P(a<=D|a>t) that AWF actually reads (D=50), nominal vs oracle")
w("%-6s %-22s %-22s" % ("tier", "nominal rho @t=0,20,35,45", "oracle rho @t=0,20,35,45"))
for t in TIERS:
    pn = _profile(t, train_belief[t]); po = _profile(t, oracle_belief[t])
    rn = [arrival_probability(pn, tv, 50.0) for tv in (0, 20, 35, 45)]
    ro = [arrival_probability(po, tv, 50.0) for tv in (0, 20, 35, 45)]
    w("%-6s %-22s %-22s" % (LABEL[t],
        " ".join("%.3f" % v for v in rn), " ".join("%.3f" % v for v in ro)))


def _replay(tiers, n_trials, rng):
    arr = np.empty((len(tiers), n_trials), float)
    for u, t in enumerate(tiers):
        seq = test_seqs[t][int(rng.randint(len(test_seqs[t])))]
        start = int(rng.randint(seq.size))
        a = seq[(start + np.arange(n_trials)) % seq.size].astype(float)
        a[rng.random(n_trials) < tier_loss[t]] = np.inf
        arr[u] = a
    return arr


w("\n[3][4] Trial-level agreement and full-precision differences")
logpool = fusion.get_method("logpool")
for D in (50.0, 100.0):
    accN = accO = latN = latO = 0.0
    nT = 0; same_stop = same_pred = 0
    for ds in DATASETS:
        users = R._ensure_users(ds)
        val_acc = np.array([u.val_accuracy for u in users])
        conf = [u.confusion for u in users]
        for seed in range(SEEDS):
            rng = np.random.RandomState(seed)
            P, y = data.align_test_trials(users, rng)
            nu, ntr = len(users), P.shape[1]
            tiers = [TIERS[int(rng.randint(len(TIERS)))] for _ in range(nu)]
            arr = _replay(tiers, ntr, rng)
            pn = [_profile(t, train_belief[t]) for t in tiers]
            po = [_profile(t, oracle_belief[t]) for t in tiers]
            for tt in range(ntr):
                base = dict(P=P[:, tt, :], arrivals_ms=arr[:, tt], deadline_ms=D,
                            val_acc=val_acc, confusions=conf, alpha=ALPHA,
                            lambda_delay=LAM, y_true=int(y[tt]))
                rn = fusion.awf_fusion(profiles=pn, **base)
                ro = fusion.awf_fusion(profiles=po, **base)
                accN += (rn.pred == y[tt]); accO += (ro.pred == y[tt])
                latN += rn.decision_time_ms; latO += ro.decision_time_ms
                same_stop += (abs(rn.decision_time_ms - ro.decision_time_ms) < 1e-9)
                same_pred += (rn.pred == ro.pred)
                nT += 1
    w("\n[D=%.0f] over %d trials" % (D, nT))
    w("  accuracy  Nominal=%.4f%%  Oracle=%.4f%%  diff=%.4f%%p" % (
        100 * accN / nT, 100 * accO / nT, 100 * (accN - accO) / nT))
    w("  latency   Nominal=%.4f ms Oracle=%.4f ms diff=%.4f ms" % (
        latN / nT, latO / nT, (latN - latO) / nT))
    w("  stop-time agreement      = %.3f%%" % (100 * same_stop / nT))
    w("  final-decision agreement = %.3f%%" % (100 * same_pred / nT))

rep = "\n".join(out)
with open(os.path.join(OUT, "heldout-diag.txt"), "w", encoding="utf-8") as f:
    f.write(rep)
print("\nwrote heldout-diag.txt")
