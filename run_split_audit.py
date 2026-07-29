"""Audit the held-out TRAIN/TEST split actually used by run_heldout.py.

Reports, per tier: number of source sequences, total samples, the TRAIN/TEST
sequence and sample counts, the split rule, whether the split is
session-disjoint (whole source sequences held out) or a same-marginal
resample, and the p50/p95/p99 + loss of each half.  No modelling, just facts.
"""
import importlib.util
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


tb = _load("tb", "bcinet-tracebank.py")
bank = tb.load_tracebank()

TIERS = ["ble", "wifi", "5g", "lte", "leo"]
# How each tier's replay sequences are constructed (from bcinet-tracebank.py):
PROVENANCE = {
    "ble":  "synthetic delay (uniform 2-12 ms), real per-beacon cadence length",
    "wifi": "bootstrap resample of one measured M-Lab marginal (independent tests)",
    "5g":   "REAL per-file contiguous PINGAVG/2 sequences (UCC 5G, one CSV = one drive session)",
    "lte":  "REAL per-file contiguous PINGAVG/2 sequences (UCC 5G, one CSV = one drive session)",
    "leo":  "REAL per-file window-mean ping_avg/2 sequences (WetLinks Starlink, one CSV = one measurement file)",
}
SESSION_DISJOINT = {"ble": False, "wifi": False, "5g": True, "lte": True, "leo": True}

lines = []
def w(s=""):
    lines.append(s)
    print(s, flush=True)

w("=" * 78)
w("HELD-OUT SPLIT AUDIT  (mirror of run_heldout.py split logic)")
w("=" * 78)
w(f"{'tier':<5} {'#seq':>5} {'Nall':>8} {'#tr':>5} {'#te':>5} "
  f"{'Ntr':>8} {'Nte':>8} {'ratio':>7}  split")
w("-" * 78)

audit = {}
for t in TIERS:
    seqs = [np.asarray(s, float) for s in bank[t].sequences
            if np.asarray(s).size >= 4]
    seqs = [s for s in seqs if np.isfinite(s).all()]
    n_seq = len(seqs)
    n_all = int(sum(s.size for s in seqs))
    if n_seq >= 2:
        h = max(1, n_seq // 2)
        tr, te = seqs[:h], seqs[h:]
        rule = "session-disjoint (seqs[:h] | seqs[h:])"
    else:
        s = seqs[0]
        m = s.size // 2
        tr, te = [s[:m]], [s[m:]]
        rule = "chronological within single sequence"
    tr_s = np.concatenate(tr)
    te_s = np.concatenate(te)
    ratio = tr_s.size / max(1, tr_s.size + te_s.size)
    audit[t] = (tr_s, te_s)
    w(f"{t:<5} {n_seq:>5} {n_all:>8} {len(tr):>5} {len(te):>5} "
      f"{tr_s.size:>8} {te_s.size:>8} {ratio:>6.2f}  {rule}")

w("-" * 78)
w()
w("Per-half distribution (one-way ms) and loss:")
w(f"{'tier':<5} {'half':<5} {'N':>8} {'p50':>8} {'p95':>8} {'p99':>8} "
  f"{'mean':>8}   loss")
w("-" * 78)
for t in TIERS:
    tr_s, te_s = audit[t]
    loss = float(bank[t].overall_loss())
    for half, s in (("train", tr_s), ("test", te_s)):
        w(f"{t:<5} {half:<5} {s.size:>8} "
          f"{np.percentile(s,50):>8.2f} {np.percentile(s,95):>8.2f} "
          f"{np.percentile(s,99):>8.2f} {s.mean():>8.2f}   {loss:.4f}")
    w("")

w("Provenance and independence level:")
w("-" * 78)
for t in TIERS:
    tag = "SESSION-DISJOINT" if SESSION_DISJOINT[t] else "same-marginal resample"
    w(f"  {t:<5} [{tag:<22}] {PROVENANCE[t]}")

with open(os.path.join(HERE, "split-audit.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
w()
w("wrote split-audit.txt")
