"""
bcinet-tracebank.py
====================

Access-tier *trace bank*: converts real wireless-measurement datasets into a
uniform per-tier representation of one-way arrival delays, loss probabilities,
link states, and contiguous replay sequences that the collaborative-BCI network
model samples from.

Tier -> source -> how the real trace is used
--------------------------------------------
WIFI  : KTH wireless-pr3d IEEE 802.11g measurements (Mostafavi et al., arXiv
        2307.10648).  Every 172-byte probe carries an IRTT-measured true uplink
        one-way delay (client-send to server-receive) so the Wi-Fi tier uses a
        real *uplink* one-way delay directly, with no round-trip halving.  Loss
        is derived from the 10 ms probe cadence (missing probes between
        consecutive arrivals).
5G    : KTH wireless-pr3d commercial 5G measurements (same source).  Each probe
        carries an IRTT-measured true uplink one-way delay on a band-n78 private
        5G network; loss from the 10 ms cadence as above.
LTE   : UCC 5G dataset (uccmisl/5Gdataset).  Real measured PINGAVG round-trip
        times (one-way = PINGAVG/2) and PINGLOSS as a single measured access
        state for the 4G-LTE radio-access-technology rows.
LEO   : Casparsen Starlink IRTT measurements (Mendeley 479v4mym7j).  Each probe
        carries an IRTT-measured true uplink one-way delay (``sd`` field) over a
        live low-earth-orbit link; loss from missing sequence numbers.

Three of the four tiers (Wi-Fi, 5G, LEO) therefore use a *true* uplink one-way
delay measured with IRTT; the 4G-LTE tier uses the UCC ping half-RTT.  The bank
is built once and cached to ``traces/tracebank.pkl``.  All delays are stored as
*one-way* milliseconds so every tier lives on the same axis as the cBCI decision
deadline, with no artificial rescaling.
"""

from __future__ import annotations

import csv
import glob
import os
import pickle
import re
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TRACES = os.path.join(HERE, "traces")
BANK_PATH = os.path.join(TRACES, "tracebank.pkl")

# Increment whenever a source filter or the cached representation changes.
# Loading an older bank silently would mix results produced from different
# trace definitions (notably the pre-revision cellular bank, which also folded
# HSPA/UMTS fallback rows into the tier labelled "4G-LTE").
TRACEBANK_SCHEMA_VERSION = 3

TIER_ORDER = ["wifi", "5g", "lte", "leo"]

#: The paper labels this tier 4G-LTE, so retain only rows whose measured radio
#: access technology is explicitly LTE.  Legacy 3G/2G fallback modes are not
#: pooled into this tier.
CELL_LTE_MODES = {"LTE"}

# ---------------------------------------------------------------------------
# IRTT true-uplink-OWD source paths (Wi-Fi / 5G KTH wireless-pr3d parquet, LEO
# Casparsen Starlink irtt.txt zip).  The ``send`` parquet column and the ``sd``
# irtt.txt field are both the true uplink one-way delay (client to server).
# ---------------------------------------------------------------------------
WIFI_IRTT_GLOB = os.path.join(TRACES, "wifi_raw", "extracted",
                              "IEEE80211g", "dataset_*_172.parquet")
G5_IRTT_GLOB = os.path.join(TRACES, "wifi_raw", "extracted",
                            "COTS-5G", "f_dataset*.parquet")
LEO_IRTT_ZIP = os.path.join(TRACES, "leo_raw", "starlink-irtt.zip")

#: Nominal IRTT probe cadence (ns).  Both KTH parquet tiers probe at 100 Hz
#: (10 ms); a timestamp gap spanning several cadences marks missing probes that
#: are counted as packet loss.
IRTT_INTERVAL_NS = 10_000_000.0

#: Per-tier caps that keep the cached bank small while preserving each tier's
#: real one-way-delay marginal and short-term autocorrelation.
SEQ_CHUNK = 1500          # contiguous replay-sequence length (samples)
MAX_SEQS = 600            # max replay sequences kept per tier
MAX_POOL = 40000          # max pooled delay samples kept per tier (for the CDF)
LEO_MAX_LINES = 1_500_000  # max irtt.txt lines parsed per LEO session


# ---------------------------------------------------------------------------
# Trace bank container
# ---------------------------------------------------------------------------

@dataclass
class TierBank:
    """One tier's sampled delays/loss/state and contiguous replay sequences."""
    states: List[str]
    state_probs: np.ndarray                       # P(state), aligned to states
    delay_by_state: Dict[str, np.ndarray]         # one-way delay samples (ms)
    loss_by_state: Dict[str, float]               # loss prob per state
    sequences: List[np.ndarray] = field(default_factory=list)  # per-source ordered one-way delays (ms)
    sequence_losses: List[float] = field(default_factory=list)  # aligned per-sequence loss rates

    def overall_loss(self) -> float:
        return float(np.dot(self.state_probs,
                            [self.loss_by_state[s] for s in self.states]))

    def all_delays(self) -> np.ndarray:
        return np.concatenate([self.delay_by_state[s] for s in self.states])


# ---------------------------------------------------------------------------
# Shared IRTT parquet helper (Wi-Fi / 5G): true uplink one-way delay (``send``)
# with loss derived from the fixed probe cadence.
# ---------------------------------------------------------------------------

def _irtt_parquet_tier(files: Sequence[str], rng: np.random.RandomState,
                       fallback_ms: float) -> TierBank:
    """Build a tier from KTH wireless-pr3d IRTT parquet campaign files.

    Each file is one measurement campaign (a distinct user/session).  The
    ``send`` column is the IRTT-measured true uplink one-way delay in
    nanoseconds; we convert it to milliseconds and keep it directly, with no
    round-trip halving.  Rows are ordered by capture ``timestamp`` so each
    campaign yields contiguous replay sequences that preserve the link's real
    short-term autocorrelation.  Loss is derived from the 10 ms probe cadence:
    a timestamp gap spanning several cadences marks probes that never arrived.
    """
    pool: List[np.ndarray] = []
    seqs: List[np.ndarray] = []
    seq_losses: List[float] = []
    tot_recv = 0
    tot_missed = 0
    for fp in sorted(files):
        try:
            df = pd.read_parquet(fp, columns=["send", "timestamp"])
        except Exception:
            continue
        if "send" not in df.columns or "timestamp" not in df.columns:
            continue
        df = df.sort_values("timestamp")
        ts = df["timestamp"].astype("int64").to_numpy()
        sd = df["send"].astype("float64").to_numpy() / 1e6      # ns -> ms
        ok = np.isfinite(sd) & (sd > 0.0) & (sd < 60000.0)
        ts, sd = ts[ok], sd[ok]
        if sd.size < 8:
            continue
        # Loss from cadence gaps: expected probes over each gap minus the one
        # that actually arrived.
        gaps = np.diff(ts)
        gaps = gaps[gaps > 0]
        expected = np.maximum(np.round(gaps / IRTT_INTERVAL_NS).astype(np.int64), 1)
        missed = int(np.sum(expected - 1))
        file_loss = missed / max(int(sd.size) + missed, 1)
        tot_recv += int(sd.size)
        tot_missed += missed
        # Pooled marginal (for the empirical CDF belief).
        pool.append(sd.astype(np.float32))
        # Contiguous replay sequences: split the ordered campaign into chunks.
        n_full = sd.size // SEQ_CHUNK
        for c in range(n_full):
            seqs.append(sd[c * SEQ_CHUNK:(c + 1) * SEQ_CHUNK].astype(np.float32))
            seq_losses.append(float(file_loss))
        if n_full == 0:
            seqs.append(sd.astype(np.float32))
            seq_losses.append(float(file_loss))
    if not pool:
        d_all = np.asarray([fallback_ms], dtype=np.float32)
        return TierBank(states=["nominal"], state_probs=np.array([1.0]),
                        delay_by_state={"nominal": d_all},
                        loss_by_state={"nominal": 0.0}, sequences=[])
    d_all = np.concatenate(pool)
    if d_all.size > MAX_POOL:
        idx = rng.choice(d_all.size, MAX_POOL, replace=False)
        d_all = d_all[np.sort(idx)]
    loss = tot_missed / max(tot_recv + tot_missed, 1)
    if len(seqs) > MAX_SEQS:
        keep = rng.choice(len(seqs), MAX_SEQS, replace=False)
        seqs = [seqs[i] for i in np.sort(keep)]
        seq_losses = [seq_losses[i] for i in np.sort(keep)]
    return TierBank(
        states=["nominal"],
        state_probs=np.array([1.0]),
        delay_by_state={"nominal": d_all},
        loss_by_state={"nominal": float(loss)},
        sequences=seqs,
        sequence_losses=seq_losses,
    )


# ---------------------------------------------------------------------------
# Wi-Fi: KTH IEEE 802.11g IRTT -> true uplink OWD (send) + cadence-gap loss
# ---------------------------------------------------------------------------

def _parse_wifi(rng: np.random.RandomState) -> TierBank:
    """Build the Wi-Fi tier from the KTH 802.11g 172-byte IRTT campaigns."""
    return _irtt_parquet_tier(glob.glob(WIFI_IRTT_GLOB), rng, fallback_ms=1.0)


# ---------------------------------------------------------------------------
# 5G: KTH commercial-5G IRTT -> true uplink OWD (send) + cadence-gap loss
# ---------------------------------------------------------------------------

def _parse_5g(rng: np.random.RandomState) -> TierBank:
    """Build the 5G tier from the KTH band-n78 commercial-5G IRTT sessions."""
    return _irtt_parquet_tier(glob.glob(G5_IRTT_GLOB), rng, fallback_ms=6.0)


# ---------------------------------------------------------------------------
# 4G-LTE: UCC 5G dataset -> measured PINGAVG/2 + PINGLOSS (single state)
# ---------------------------------------------------------------------------

def _parse_lte(rng: np.random.RandomState) -> TierBank:
    """Build the 4G-LTE tier from the UCC measured ping logs.

    The UCC ``5G`` dataset logs every sample's radio-access technology in
    ``NetworkMode``.  We keep only rows explicitly labelled ``LTE``; older
    HSPA/UMTS/EDGE fallback modes are excluded so the source filter matches
    the tier name used in the paper.  The measured ``PINGAVG/2``
    one-way delays and ``PINGLOSS`` are used directly as a single measured
    access state.  This is the only tier that uses a half-RTT rather than a
    true IRTT uplink one-way delay, because no accessible packet-level LTE
    dataset provides a true uplink one-way delay.
    """
    files = glob.glob(os.path.join(TRACES, "cellular",
                                   "5G-production-dataset", "**", "*.csv"),
                      recursive=True)
    delays: List[float] = []
    loss: List[float] = []
    seqs: List[np.ndarray] = []
    seq_losses: List[float] = []
    for fp in files:
        try:
            with open(fp, encoding="utf-8", errors="replace") as f:
                r = csv.DictReader(f)
                cur: List[float] = []
                cur_loss: List[float] = []
                for row in r:
                    nm = (row.get("NetworkMode") or "").strip()
                    if nm not in CELL_LTE_MODES:
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
                    cur.append(ow)
                    pl = row.get("PINGLOSS", "-")
                    try:
                        v = float(pl)
                        loss_value = v / 100.0 if v > 1.0 else v
                        loss.append(loss_value)
                        cur_loss.append(loss_value)
                    except (TypeError, ValueError):
                        pass
                if len(cur) >= 4:
                    seqs.append(np.asarray(cur, dtype=np.float32))
                    seq_losses.append(float(np.mean(cur_loss))
                                      if cur_loss else np.nan)
        except Exception:
            continue
    d = np.asarray(delays or [45.0], dtype=np.float32)
    loss_mean = float(np.mean(loss)) if loss else 0.02
    seq_losses = [loss_mean if not np.isfinite(x) else float(x)
                  for x in seq_losses]
    return TierBank(
        states=["nominal"],
        state_probs=np.array([1.0]),
        delay_by_state={"nominal": d},
        loss_by_state={"nominal": loss_mean},
        sequences=seqs[:MAX_SEQS],
        sequence_losses=seq_losses[:MAX_SEQS],
    )


# ---------------------------------------------------------------------------
# LEO: Casparsen Starlink IRTT -> true uplink OWD (sd) + sequence-gap loss
# ---------------------------------------------------------------------------

_LEO_LINE = re.compile(rb"seq=(\d+).*?\bsd=([0-9.]+)(m?s)\b")


def _parse_leo(rng: np.random.RandomState) -> TierBank:
    """Build the LEO tier from the Casparsen Starlink IRTT text logs.

    Each measurement session is one large ``irtt.txt`` inside the zip whose
    per-probe lines carry an IRTT-measured true uplink one-way delay in the
    ``sd`` field (client-send to server-receive) alongside a monotonically
    increasing sequence number.  We stream every session line by line, keep
    the ``sd`` value directly as the uplink one-way delay in milliseconds (no
    round-trip halving), and derive loss from missing sequence numbers between
    consecutive received probes.  Each session yields contiguous replay
    sequences that preserve the real short-term autocorrelation of the link.
    """
    pool: List[np.ndarray] = []
    seqs: List[np.ndarray] = []
    seq_losses: List[float] = []
    tot_recv = 0
    tot_missed = 0
    if not os.path.exists(LEO_IRTT_ZIP):
        d = np.asarray([30.0], dtype=np.float32)
        return TierBank(states=["nominal"], state_probs=np.array([1.0]),
                        delay_by_state={"nominal": d},
                        loss_by_state={"nominal": 0.003}, sequences=[])
    with zipfile.ZipFile(LEO_IRTT_ZIP) as z:
        members = [m for m in z.namelist() if m.endswith(".txt")]
        for m in sorted(members):
            sd_list: List[float] = []
            last_seq = -1
            session_missed = 0
            n_lines = 0
            with z.open(m) as f:
                for raw in f:
                    n_lines += 1
                    if n_lines > LEO_MAX_LINES:
                        break
                    mm = _LEO_LINE.search(raw)
                    if mm is None:
                        continue
                    seq = int(mm.group(1))
                    val = float(mm.group(2))
                    if mm.group(3) == b"s":            # seconds -> ms
                        val *= 1000.0
                    if not (0.0 < val < 60000.0):
                        continue
                    if last_seq >= 0 and seq > last_seq:
                        gap_missed = seq - last_seq - 1
                        tot_missed += gap_missed
                        session_missed += gap_missed
                    last_seq = seq
                    sd_list.append(val)
            if len(sd_list) < 8:
                continue
            sd = np.asarray(sd_list, dtype=np.float32)
            session_loss = session_missed / max(sd.size + session_missed, 1)
            tot_recv += sd.size
            pool.append(sd)
            n_full = sd.size // SEQ_CHUNK
            for c in range(n_full):
                seqs.append(sd[c * SEQ_CHUNK:(c + 1) * SEQ_CHUNK])
                seq_losses.append(float(session_loss))
            if n_full == 0:
                seqs.append(sd)
                seq_losses.append(float(session_loss))
    if not pool:
        d = np.asarray([30.0], dtype=np.float32)
        return TierBank(states=["nominal"], state_probs=np.array([1.0]),
                        delay_by_state={"nominal": d},
                        loss_by_state={"nominal": 0.003}, sequences=[])
    d_all = np.concatenate(pool)
    if d_all.size > MAX_POOL:
        idx = rng.choice(d_all.size, MAX_POOL, replace=False)
        d_all = d_all[np.sort(idx)]
    loss = tot_missed / max(tot_recv + tot_missed, 1)
    if len(seqs) > MAX_SEQS:
        keep = rng.choice(len(seqs), MAX_SEQS, replace=False)
        seqs = [seqs[i] for i in np.sort(keep)]
        seq_losses = [seq_losses[i] for i in np.sort(keep)]
    return TierBank(
        states=["nominal"],
        state_probs=np.array([1.0]),
        delay_by_state={"nominal": d_all},
        loss_by_state={"nominal": float(loss)},
        sequences=seqs,
        sequence_losses=seq_losses,
    )


# ---------------------------------------------------------------------------
# Build / load
# ---------------------------------------------------------------------------

def _to_plain(bank: Dict[str, TierBank]) -> dict:
    """Convert TierBank instances to plain dicts/arrays for a portable cache."""
    return {t: {
        "states": list(b.states),
        "state_probs": np.asarray(b.state_probs),
        "delay_by_state": {s: np.asarray(v) for s, v in b.delay_by_state.items()},
        "loss_by_state": {s: float(v) for s, v in b.loss_by_state.items()},
        "sequences": [np.asarray(x) for x in b.sequences],
        "sequence_losses": [float(x) for x in b.sequence_losses],
    } for t, b in bank.items()}


def _from_plain(d: dict) -> Dict[str, TierBank]:
    """Rebuild the tier->TierBank dict from the plain cached representation."""
    return {t: TierBank(
        states=list(v["states"]),
        state_probs=np.asarray(v["state_probs"]),
        delay_by_state={s: np.asarray(a) for s, a in v["delay_by_state"].items()},
        loss_by_state={s: float(x) for s, x in v["loss_by_state"].items()},
        sequences=[np.asarray(x) for x in v["sequences"]],
        sequence_losses=[float(x) for x in v.get("sequence_losses", [])],
    ) for t, v in d.items()}


def build_tracebank(seed: int = 0,
                    verbose: bool = True) -> Dict[str, TierBank]:
    """Parse all raw datasets into a tier->TierBank dict and cache it."""
    rng = np.random.RandomState(seed)
    bank: Dict[str, TierBank] = {}
    if verbose:
        print("[tracebank] parsing Wi-Fi 802.11g IRTT uplink OWD (KTH) ...")
    bank["wifi"] = _parse_wifi(rng)
    if verbose:
        print("[tracebank] parsing 5G IRTT uplink OWD (KTH band-n78) ...")
    bank["5g"] = _parse_5g(rng)
    if verbose:
        print("[tracebank] parsing 4G-LTE ping logs (UCC, PINGAVG/2) ...")
    bank["lte"] = _parse_lte(rng)
    if verbose:
        print("[tracebank] parsing LEO Starlink IRTT uplink OWD (Casparsen) ...")
    bank["leo"] = _parse_leo(rng)
    with open(BANK_PATH, "wb") as f:
        pickle.dump({"schema_version": TRACEBANK_SCHEMA_VERSION,
                     "bank": _to_plain(bank)}, f)
    if verbose:
        _print_summary(bank)
    return bank


_BANK_CACHE: Optional[Dict[str, TierBank]] = None


def load_tracebank() -> Dict[str, TierBank]:
    """Load the cached trace bank, building it on first use."""
    global _BANK_CACHE
    if _BANK_CACHE is not None:
        return _BANK_CACHE
    if not os.path.exists(BANK_PATH):
        _BANK_CACHE = build_tracebank()
        return _BANK_CACHE
    with open(BANK_PATH, "rb") as f:
        cached = pickle.load(f)
    if (not isinstance(cached, dict)
            or cached.get("schema_version") != TRACEBANK_SCHEMA_VERSION
            or "bank" not in cached):
        print("[tracebank] cache definition changed; rebuilding ...")
        _BANK_CACHE = build_tracebank()
        return _BANK_CACHE
    _BANK_CACHE = _from_plain(cached["bank"])
    return _BANK_CACHE


def _print_summary(bank: Dict[str, TierBank]) -> None:
    print("\n=== access-tier trace bank summary (one-way ms) ===")
    for t in TIER_ORDER:
        b = bank[t]
        print(f"\n[{t}]  states={b.states}  overall_loss={b.overall_loss():.3f}  "
              f"sequences={len(b.sequences)}")
        for s in b.states:
            d = b.delay_by_state[s]
            print(f"   {s:9s} p={b.state_probs[b.states.index(s)]:.3f} "
                  f"loss={b.loss_by_state[s]:.3f} "
                  f"delay p10/50/90 = {np.percentile(d,10):.1f}/"
                  f"{np.percentile(d,50):.1f}/{np.percentile(d,90):.1f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    build_tracebank(seed=args.seed)
