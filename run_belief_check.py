"""Corrected input check: compare the rho_j(t) and E[Delta_j(t)] that AWF ACTUALLY
reads (arrival_probability_empirical / expected_remaining_delay, both using
est_samples) between the nominal (train) and oracle (test) beliefs."""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


R = _load("runexp", "run-bcinet-experiment.py")
net = R.bcinet_network
fusion = R.bcinet_fusion
tb = _load("tb", "bcinet-tracebank.py")
bank = tb.load_tracebank()
NetworkProfile = net.NetworkProfile
rho_emp = fusion.arrival_probability_empirical
edelay = fusion.expected_remaining_delay

TIERS = ["ble", "wifi", "5g", "lte", "leo"]
LABEL = {"ble": "BLE", "wifi": "Wi-Fi", "5g": "5G", "lte": "4G", "leo": "LEO"}

train_b, test_b, loss = {}, {}, {}
for t in TIERS:
    seqs = [np.asarray(s, float) for s in bank[t].sequences
            if np.asarray(s).size >= 4 and np.isfinite(np.asarray(s)).all()]
    if len(seqs) >= 2:
        h = max(1, len(seqs) // 2); tr, te = seqs[:h], seqs[h:]
    else:
        s = seqs[0]; m = s.size // 2; tr, te = [s[:m]], [s[m:]]
    train_b[t] = np.sort(np.concatenate(tr))
    test_b[t] = np.sort(np.concatenate(te))
    loss[t] = float(bank[t].overall_loss())


def prof(t, est):
    return NetworkProfile(tier=t, delay_range=(0.0, 1.0), jitter_range=(0.0, 0.0),
                          loss_prob=loss[t], state=None, est_samples=est,
                          est_loss=loss[t])


print("rho_j(t) AWF ACTUALLY reads (arrival_probability_empirical), D=50")
print("%-6s | %-27s | %-27s" % ("tier", "nominal (train) @0,20,35,45",
                                "oracle (test) @0,20,35,45"))
for t in TIERS:
    pn, po = prof(t, train_b[t]), prof(t, test_b[t])
    rn = [rho_emp(pn, tv, 50.0) for tv in (0, 20, 35, 45)]
    ro = [rho_emp(po, tv, 50.0) for tv in (0, 20, 35, 45)]
    print("%-6s | %-27s | %-27s" % (LABEL[t],
        " ".join("%.3f" % v for v in rn), " ".join("%.3f" % v for v in ro)))

print("\nE[Delta_j(t)] AWF ACTUALLY reads (expected_remaining_delay), D=50")
print("%-6s | %-27s | %-27s" % ("tier", "nominal @0,20,35,45", "oracle @0,20,35,45"))
for t in TIERS:
    pn, po = prof(t, train_b[t]), prof(t, test_b[t])
    en = [edelay(pn, tv, 50.0) for tv in (0, 20, 35, 45)]
    eo = [edelay(po, tv, 50.0) for tv in (0, 20, 35, 45)]
    print("%-6s | %-27s | %-27s" % (LABEL[t],
        " ".join("%.2f" % v for v in en), " ".join("%.2f" % v for v in eo)))
