"""
bcinet-network.py
=================

Network model for the distributed collaborative BCI pipeline.

Each user is assigned a *network tier* (BLE local, Wi-Fi, Broadband,
Cellular, or Satellite). Each tier has a typical delay range, a typical
jitter range, and a typical loss probability. For every trial we draw,
per user, a stochastic arrival time and a Bernoulli loss event. The
fusion strategies in ``bcinet-fusion.py`` consume these arrival traces.

The module also exposes :func:`arrival_probability` which computes the
probability that a still-missing user response will arrive before the
deadline. AWF uses this quantity (denoted ``rho_i(t)``) to decide
whether to keep waiting for a missing user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import math
import os

import numpy as np


# ---------------------------------------------------------------------------
# Network tier specifications (delay/jitter/loss in milliseconds and percent)
# ---------------------------------------------------------------------------

NETWORK_TIERS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "ble":        {"delay": (3.0, 15.0),   "jitter": (1.0, 5.0),    "loss": (0.0, 0.01)},
    "wifi":       {"delay": (15.0, 60.0),  "jitter": (5.0, 20.0),   "loss": (0.0, 0.03)},
    "broadband":  {"delay": (50.0, 200.0), "jitter": (10.0, 40.0),  "loss": (0.0, 0.05)},
    "cellular":   {"delay": (80.0, 350.0), "jitter": (20.0, 80.0),  "loss": (0.02, 0.10)},
    "satellite":  {"delay": (200.0, 600.0),"jitter": (50.0, 150.0), "loss": (0.05, 0.20)},
}

TIER_ORDER = ["ble", "wifi", "broadband", "cellular", "satellite"]


# ---------------------------------------------------------------------------
# Network presets - probability mass over tiers
# ---------------------------------------------------------------------------

NETWORK_PRESETS: Dict[str, Dict[str, float]] = {
    "uniform_ble": {"ble": 1.0, "wifi": 0.0, "broadband": 0.0,
                    "cellular": 0.0, "satellite": 0.0},
    "balanced":    {"ble": 0.20, "wifi": 0.25, "broadband": 0.25,
                    "cellular": 0.20, "satellite": 0.10},
    "cellular_heavy": {"ble": 0.10, "wifi": 0.10, "broadband": 0.20,
                       "cellular": 0.50, "satellite": 0.10},
    "satellite_heavy": {"ble": 0.10, "wifi": 0.10, "broadband": 0.15,
                        "cellular": 0.25, "satellite": 0.40},
    # Each user independently draws a tier with equal probability from all 5
    # tiers.  This is the most realistic "wild-card" scenario: a high-accuracy
    # user may end up on satellite while a low-accuracy user is on BLE, or any
    # other combination, with no correlation between user quality and network.
    "uniform_all": {"ble": 0.20, "wifi": 0.20, "broadband": 0.20,
                    "cellular": 0.20, "satellite": 0.20},
    # Sentinel preset: arrival times are bootstrapped from a *real* measured
    # latency trace (WS-DREAM dataset#1) instead of the synthetic
    # uniform+Gaussian model.  Handled specially in ``assign_network_tiers``;
    # the weights below are unused.
    "realtrace": {"ble": 1.0, "wifi": 0.0, "broadband": 0.0,
                  "cellular": 0.0, "satellite": 0.0},
    # Sentinel preset: hybrid per-tier real trace.  Each user draws a tier
    # uniformly (same marginal as ``uniform_all``) and is bound to a real
    # WS-DREAM link from that tier's pool; tiers with no measured link (BLE)
    # fall back to the synthetic model.  Handled specially in
    # ``assign_network_tiers``; the (uniform) weights below are unused.
    "realtier": {"ble": 0.20, "wifi": 0.20, "broadband": 0.20,
                 "cellular": 0.20, "satellite": 0.20},
    # Sentinel preset: tiers evolve as a Markov chain across trials (temporal
    # correlation / mobility) with a uniform stationary distribution, so the
    # marginal matches ``uniform_all`` while adding inter-trial correlation.
    # Handled specially in the runner via ``assign_markov_tiers``; the weights
    # below are the (uniform) stationary marginal and are otherwise unused.
    "markov": {"ble": 0.20, "wifi": 0.20, "broadband": 0.20,
               "cellular": 0.20, "satellite": 0.20},
}

DEFAULT_DEADLINE_MS = 100.0


# ---------------------------------------------------------------------------
# Access-tier trace-bank presets
# ---------------------------------------------------------------------------
# These presets ground every user in a real per-access-tier measurement trace
# (Wi-Fi / 5G / 4G-LTE / LEO) built by ``bcinet-tracebank.py``.  Unlike the
# WS-DREAM ``realtier`` hybrid (wide-area web-service RT), each access tier
# keeps the character of its own trace: the Wi-Fi, 5G and LEO tiers use an
# IRTT-measured true uplink one-way delay directly, while the 4G-LTE tier uses
# the UCC ping half-RTT.  Realised per-trial arrivals are bootstrapped from the
# tier+state delay samples; the arrival probability ``rho`` is estimated from
# the same empirical CDF (see ``est_samples``).
ACCESS_TIERS = ["wifi", "5g", "lte", "leo"]

#: Tier marginal mixes for the access-tier presets.  The default
#: ``accesstier`` is a *uniform* mix over the present tiers -- deliberately
#: assumption-free, since the traces fix each tier's delay/loss but not how
#: many collaborators use each tier.  The ``leo*`` variants instead sweep the
#: non-terrestrial (LEO) fraction, a clearly-motivated NTN-penetration
#: sensitivity, redistributing the remainder uniformly over the three
#: terrestrial tiers.
ACCESS_TIER_MIXES: Dict[str, Dict[str, float]] = {
    "accesstier":       {"wifi": 0.25,   "5g": 0.25,   "lte": 0.25,   "leo": 0.25},
    "accesstier_leo00": {"wifi": 1/3,    "5g": 1/3,    "lte": 1/3,    "leo": 0.00},
    "accesstier_leo10": {"wifi": 0.30,   "5g": 0.30,   "lte": 0.30,   "leo": 0.10},
    "accesstier_leo30": {"wifi": 0.7/3,  "5g": 0.7/3,  "lte": 0.7/3,  "leo": 0.30},
    # Deployment use-case personas (realistic access compositions rather than
    # synthetic parameter sweeps).  Each maps a plausible networked-cBCI
    # deployment to the access tiers its collaborators would actually use.
    #   clinic : co-located rehab/lab cohort on local Wi-Fi to an edge server
    #   metro  : metropolitan users on Wi-Fi/5G to a mobile-edge node
    #   wide   : wide-area mobile users on 5G/4G cellular
    #   field  : remote/disaster/maritime users on LEO satellite with cellular fallback
    "accesstier_clinic": {"wifi": 0.70, "5g": 0.30, "lte": 0.00,  "leo": 0.00},
    "accesstier_metro":  {"wifi": 0.50, "5g": 0.50, "lte": 0.00,  "leo": 0.00},
    "accesstier_wide":   {"wifi": 0.00, "5g": 0.40, "lte": 0.60,  "leo": 0.00},
    "accesstier_field":  {"wifi": 0.00, "5g": 0.20, "lte": 0.20,  "leo": 0.60},
}

#: Online belief used to estimate ``rho`` for the arrival-probability ablation.
#: ``linear`` falls back to the parametric ramp; ``tier`` pools all of a tier's
#: delays; ``state`` (default) uses the current link state's empirical CDF;
#: ``oracle`` knows the true generating distribution (= state samples + loss).
ACCESS_BELIEFS = ("linear", "tier", "state", "oracle")

#: Register access-tier presets, including arrival-belief ablation variants
#: (``<mix>__belief-<b>``).  The mix dict doubles as the tier marginal.
for _mix_name, _mix_w in ACCESS_TIER_MIXES.items():
    NETWORK_PRESETS[_mix_name] = dict(_mix_w)
    for _belief in ACCESS_BELIEFS:
        NETWORK_PRESETS[f"{_mix_name}__belief-{_belief}"] = dict(_mix_w)


# ---------------------------------------------------------------------------
# Per-user network profile
# ---------------------------------------------------------------------------

@dataclass
class NetworkProfile:
    """Per-user network properties drawn at experiment configuration time."""
    tier: str
    delay_range: Tuple[float, float]   # (d_min, d_max) ms (after sampling within tier range)
    jitter_range: Tuple[float, float]  # ms
    loss_prob: float                   # in [0, 1]
    # When populated (``realtrace`` preset), per-trial arrival delays are
    # bootstrapped from these real measured samples (ms) and the link is
    # considered lost with probability ``trace_loss``.  AWF still reasons
    # about arrivals through the nominal ``delay_range``/``loss_prob`` above,
    # so it remains blind to the true (real) arrival distribution.
    trace_samples: Optional[np.ndarray] = None
    trace_loss: float = 0.0
    # Access-tier trace-bank fields (``accesstier`` presets).  ``state`` is the
    # tier's discrete link state (e.g. Wi-Fi good/weak/handover).  ``est_*`` is
    # the *belief* an online node holds about its own arrival distribution and
    # drives the empirical-CDF arrival probability: when ``est_samples`` is set,
    # AWF estimates ``rho`` empirically from it instead of the linear ramp.
    state: Optional[str] = None
    est_samples: Optional[np.ndarray] = None
    est_loss: float = 0.0


def _sample_in_range(rng: np.random.RandomState,
                     low: float, high: float) -> float:
    """Uniformly sample a scalar in [low, high]."""
    return float(rng.uniform(low, high))


# ---------------------------------------------------------------------------
# Real measured latency trace (WS-DREAM dataset#1)
# ---------------------------------------------------------------------------

# Path to the extracted WS-DREAM response-time matrix (339 users x 5825
# services, in seconds; -1 marks an invalid/timed-out measurement).
_WSDREAM_RT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "realtrace", "dataset1", "rtMatrix.txt")

# Lazily-populated cache: list of (samples_ms, loss_prob, median_ms, p10, p90)
# per user.
_WSDREAM_CACHE: Optional[List[Tuple[np.ndarray, float, float, float, float]]] = None


def _tier_for_median(median_ms: float) -> str:
    """Bin a link by its median latency into one of the five nominal tiers."""
    if median_ms < NETWORK_TIERS["ble"]["delay"][1]:          # < 15 ms
        return "ble"
    if median_ms < NETWORK_TIERS["wifi"]["delay"][1]:         # < 60 ms
        return "wifi"
    if median_ms < NETWORK_TIERS["broadband"]["delay"][1]:    # < 200 ms
        return "broadband"
    if median_ms < NETWORK_TIERS["cellular"]["delay"][1]:     # < 350 ms
        return "cellular"
    return "satellite"


def load_wsdream_traces(deadline_ms: float = DEFAULT_DEADLINE_MS
                        ) -> List[Tuple[np.ndarray, float, float, float, float]]:
    """Load and cache per-user real latency samples from WS-DREAM dataset#1.

    The raw response times (seconds) are linearly rescaled by a single
    global constant so that the population median latency equals the
    deadline ``deadline_ms``.  This places the real trace in the same
    operating regime as the synthetic benchmark (about half the responses
    at risk of missing the deadline) for a like-for-like comparison, while
    preserving every user's empirical shape, heterogeneity, and heavy tail.
    Invalid measurements (-1) are retained as real loss events.

    Returns one ``(samples_ms, loss_prob, median_ms, p10, p90)`` tuple per
    real user; ``p10``/``p90`` are the link's own latency percentiles used
    as a realistic online estimate of its arrival distribution.
    """
    global _WSDREAM_CACHE
    if _WSDREAM_CACHE is not None:
        return _WSDREAM_CACHE
    if not os.path.exists(_WSDREAM_RT_PATH):
        raise FileNotFoundError(
            f"WS-DREAM trace not found at {_WSDREAM_RT_PATH!r}. "
            "Download dataset#1 (wsdream_dataset1.zip) and extract it under "
            "realtrace/dataset1/.")
    M = np.loadtxt(_WSDREAM_RT_PATH)            # (339, 5825) seconds; -1 invalid
    valid_all = M[M > 0] * 1000.0               # to ms
    scale = float(deadline_ms) / float(np.median(valid_all))
    cache: List[Tuple[np.ndarray, float, float, float, float]] = []
    for row in M:
        valid = row[row > 0] * 1000.0 * scale
        if valid.size == 0:
            continue
        loss_prob = float(np.mean(row <= 0))
        p10 = float(np.percentile(valid, 10))
        p90 = float(np.percentile(valid, 90))
        if p90 <= p10:
            p90 = p10 + 1.0
        cache.append((valid.astype(np.float64), loss_prob,
                      float(np.median(valid)), p10, p90))
    _WSDREAM_CACHE = cache
    return cache


def _assign_realtrace_profiles(n_users: int, rng: np.random.RandomState,
                               deadline_ms: float = DEFAULT_DEADLINE_MS
                               ) -> List[NetworkProfile]:
    """Assign each cBCI user to a random real WS-DREAM link.

    The tier label is derived from the link's median latency (for
    reporting and jitter), but AWF's arrival-probability estimate is made
    realistic by setting each link's ``delay_range`` to its *own measured*
    10th--90th percentile latencies, i.e. the statistics a deployed node
    would obtain by monitoring its own round-trip times online. This lets
    rho_i(t) reflect the real (heavy-tailed) arrival distribution instead
    of a nominal tier guess, so AWF stops waiting for links whose tail
    makes a timely arrival unlikely. The actual per-trial arrival delays
    are still bootstrapped from the link's real samples in
    :func:`sample_arrivals`.
    """
    traces = load_wsdream_traces(deadline_ms)
    profiles: List[NetworkProfile] = []
    for _ in range(n_users):
        samples, loss_prob, median_ms, p10, p90 = traces[int(rng.randint(len(traces)))]
        tier = _tier_for_median(median_ms)
        spec = NETWORK_TIERS[tier]
        # Online link estimate: bracket rho's linear ramp by this link's own
        # empirical p10/p90 latency so the arrival probability tracks the
        # real (heavy-tailed) arrival distribution.
        profiles.append(NetworkProfile(
            tier=tier, delay_range=(p10, p90), jitter_range=spec["jitter"],
            loss_prob=loss_prob,
            trace_samples=samples, trace_loss=loss_prob,
        ))
    return profiles


# Lazily-populated per-tier pools of WS-DREAM links (one list of
# ``(samples_ms, loss_prob, median_ms, p10, p90)`` tuples per tier name).
_WSDREAM_TIER_POOLS: Optional[Dict[str, List[Tuple[np.ndarray, float, float, float, float]]]] = None


def _build_wsdream_tier_pools(deadline_ms: float = DEFAULT_DEADLINE_MS
                              ) -> Dict[str, List[Tuple[np.ndarray, float, float, float, float]]]:
    """Group every real WS-DREAM link into its nominal tier by median latency.

    Returns a dict mapping each tier name to the list of real links whose
    rescaled median falls in that tier's range.  Empty tiers (e.g. BLE,
    whose sub-15 ms regime no measured link reaches after the global median
    rescale) map to an empty list and are filled synthetically by the
    caller, so the full five-tier span is preserved while every tier that
    *can* be grounded in real data is.
    """
    global _WSDREAM_TIER_POOLS
    if _WSDREAM_TIER_POOLS is not None:
        return _WSDREAM_TIER_POOLS
    pools: Dict[str, List[Tuple[np.ndarray, float, float, float, float]]] = {
        t: [] for t in TIER_ORDER}
    for entry in load_wsdream_traces(deadline_ms):
        pools[_tier_for_median(entry[2])].append(entry)
    _WSDREAM_TIER_POOLS = pools
    return pools


def _assign_realtier_profiles(n_users: int, rng: np.random.RandomState,
                              deadline_ms: float = DEFAULT_DEADLINE_MS
                              ) -> List[NetworkProfile]:
    """Hybrid per-tier real-trace assignment.

    Each user first draws a tier uniformly over the five tiers (the same
    marginal as ``uniform_all``).  If that tier has real WS-DREAM links, the
    user is bound to a random measured link from the tier's pool -- its
    per-trial arrivals are bootstrapped from the link's real samples and its
    online ``delay_range`` is the link's own p10/p90 latency.  If the tier
    has *no* measured link (BLE, after the global-median rescale), the user
    falls back to a synthetic :func:`_make_tier_profile`, so the extreme
    low-latency regime that real measurements never reach is still spanned.
    This grounds every populated tier in real data while preserving the full
    five-tier heterogeneity the synthetic skeleton provides.
    """
    pools = _build_wsdream_tier_pools(deadline_ms)
    profiles: List[NetworkProfile] = []
    for _ in range(n_users):
        tier = TIER_ORDER[int(rng.randint(len(TIER_ORDER)))]
        pool = pools[tier]
        if not pool:                       # no real link in this tier -> synthetic
            profiles.append(_make_tier_profile(tier, rng))
            continue
        samples, loss_prob, _median, p10, p90 = pool[int(rng.randint(len(pool)))]
        spec = NETWORK_TIERS[tier]
        profiles.append(NetworkProfile(
            tier=tier, delay_range=(p10, p90), jitter_range=spec["jitter"],
            loss_prob=loss_prob,
            trace_samples=samples, trace_loss=loss_prob,
        ))
    return profiles


def assign_network_tiers(n_users: int, preset: str,
                         rng: np.random.RandomState) -> List[NetworkProfile]:
    """Assign one :class:`NetworkProfile` per user according to a preset.
    For each user we (1) draw a tier from the categorical distribution
    defined by the preset, then (2) draw a *user-specific* delay range,
    jitter range, and loss probability from inside the tier-level ranges.
    This way two users on the same tier can still differ in their
    typical delay, which is more realistic than collapsing every tier to
    a point estimate.
    """
    if preset not in NETWORK_PRESETS:
        raise ValueError(
            f"Unknown preset {preset!r}. Choose from {list(NETWORK_PRESETS)}."
        )
    if preset == "realtrace":
        return _assign_realtrace_profiles(n_users, rng)
    if preset == "realtier":
        return _assign_realtier_profiles(n_users, rng)
    if is_accesstier_preset(preset):
        return assign_accesstier_tiers(n_users, preset, rng)
    weights = NETWORK_PRESETS[preset]
    tiers = list(weights.keys())
    p = np.array([weights[t] for t in tiers], dtype=float)
    p = p / p.sum()

    profiles: List[NetworkProfile] = []
    for _ in range(n_users):
        tier = tiers[int(rng.choice(len(tiers), p=p))]
        profiles.append(_make_tier_profile(tier, rng))
    return profiles


def _make_tier_profile(tier: str,
                       rng: np.random.RandomState) -> NetworkProfile:
    """Build one user-specific :class:`NetworkProfile` inside ``tier``.

    Draws a user-specific delay window, jitter range, and loss probability
    from inside the tier-level ranges, so two users on the same tier still
    differ in their typical delay.
    """
    spec = NETWORK_TIERS[tier]
    d_lo, d_hi = spec["delay"]
    # Pick a user-specific delay window that lives inside the tier's range.
    a = _sample_in_range(rng, d_lo, (d_lo + d_hi) / 2)
    b = _sample_in_range(rng, (d_lo + d_hi) / 2, d_hi)
    d_user = (a, b)
    j_user = spec["jitter"]
    loss_user = _sample_in_range(rng, *spec["loss"])
    return NetworkProfile(
        tier=tier, delay_range=d_user, jitter_range=j_user,
        loss_prob=loss_user,
    )


# ---------------------------------------------------------------------------
# Access-tier trace bank (BLE / Wi-Fi / cellular / LEO)
# ---------------------------------------------------------------------------

# Lazily-loaded ``{tier -> TierBank}`` mapping from ``bcinet-tracebank.py``.
_TRACEBANK: Optional[dict] = None
# Per-(tier, belief) cached sorted delay arrays used by the empirical CDF.
_BELIEF_CACHE: Dict[Tuple[str, str, Optional[str]], Tuple[np.ndarray, float]] = {}


def _load_tracebank() -> dict:
    """Load and cache the per-access-tier trace bank built offline.

    Imports ``bcinet-tracebank.py`` as a sibling module (hyphenated filename)
    and returns its ``{tier -> TierBank}`` dictionary.  Raises a clear error if
    the bank has not been built yet.
    """
    global _TRACEBANK
    if _TRACEBANK is not None:
        return _TRACEBANK
    import importlib.util
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "bcinet-tracebank.py")
    spec = importlib.util.spec_from_file_location("bcinet_tracebank", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bcinet_tracebank"] = mod
    spec.loader.exec_module(mod)
    _TRACEBANK = mod.load_tracebank()
    return _TRACEBANK


def _belief_distribution(tier: str, belief: str, state: Optional[str]
                         ) -> Tuple[Optional[np.ndarray], float]:
    """Return ``(sorted_delay_samples, est_loss)`` for an online arrival belief.

    ``linear`` returns ``(None, ...)`` so AWF falls back to the parametric
    ramp; ``tier`` pools every delay sample of the tier with the tier-wide
    loss; ``state``/``oracle`` use the current link state's empirical delay CDF
    and per-state loss (the true generating distribution given the state).
    """
    if belief == "linear":
        bank = _load_tracebank()[tier]
        return None, float(bank.overall_loss())
    key = (tier, belief, state if belief in ("state", "oracle") else None)
    if key in _BELIEF_CACHE:
        return _BELIEF_CACHE[key]
    bank = _load_tracebank()[tier]
    if belief == "tier":
        samples = np.sort(np.asarray(bank.all_delays(), dtype=np.float64))
        loss = float(bank.overall_loss())
    else:                                   # state / oracle
        samples = np.sort(np.asarray(bank.delay_by_state[state], dtype=np.float64))
        loss = float(bank.loss_by_state[state])
    _BELIEF_CACHE[key] = (samples, loss)
    return samples, loss


def _make_accesstier_profile(tier: str, rng: np.random.RandomState,
                             belief: str = "state") -> NetworkProfile:
    """Build one access-tier :class:`NetworkProfile` from the trace bank.

    A link *state* is drawn from the tier's empirical state distribution; the
    realised per-trial arrivals are bootstrapped from that state's measured
    one-way delay samples (``trace_samples``) and lost with the state's
    measured probability.  The online arrival *belief* (``est_samples``) is set
    per ``belief`` and drives the empirical-CDF ``rho`` estimate.
    """
    bank = _load_tracebank()[tier]
    probs = np.asarray(bank.state_probs, dtype=float)
    probs = probs / probs.sum()
    si = int(rng.choice(len(bank.states), p=probs))
    state = bank.states[si]
    samples = np.asarray(bank.delay_by_state[state], dtype=np.float64)
    loss = float(bank.loss_by_state[state])
    if samples.size:
        p10 = float(np.percentile(samples, 10))
        p90 = float(np.percentile(samples, 90))
        if p90 <= p10:
            p90 = p10 + 1.0
    else:
        p10, p90 = 0.0, 1.0
    est_samples, est_loss = _belief_distribution(tier, belief, state)
    return NetworkProfile(
        tier=tier, delay_range=(p10, p90), jitter_range=(0.0, 0.0),
        loss_prob=loss, trace_samples=samples, trace_loss=loss,
        state=state, est_samples=est_samples, est_loss=est_loss,
    )


def _parse_accesstier_preset(preset: str) -> Tuple[str, str]:
    """Split an access-tier preset into ``(mix_name, belief)``.

    ``"accesstier"`` -> ``("accesstier", "state")``;
    ``"accesstier_leo30__belief-tier"`` -> ``("accesstier_leo30", "tier")``.
    """
    belief = "state"
    mix = preset
    if "__belief-" in preset:
        mix, belief = preset.split("__belief-", 1)
    return mix, belief


def assign_accesstier_tiers(n_users: int, preset: str,
                            rng: np.random.RandomState) -> List[NetworkProfile]:
    """Assign one trace-bank-backed profile per user for an access-tier preset.

    Each user draws an access tier from the preset's marginal mix, then a link
    state and a real delay sample bootstrap from the trace bank.
    """
    mix, belief = _parse_accesstier_preset(preset)
    weights = ACCESS_TIER_MIXES[mix]
    tiers = [t for t in ACCESS_TIERS if weights.get(t, 0.0) > 0.0]
    p = np.array([weights[t] for t in tiers], dtype=float)
    p = p / p.sum()
    profiles: List[NetworkProfile] = []
    for _ in range(n_users):
        tier = tiers[int(rng.choice(len(tiers), p=p))]
        profiles.append(_make_accesstier_profile(tier, rng, belief))
    return profiles


def is_accesstier_preset(preset: str) -> bool:
    """True if ``preset`` is an access-tier trace-bank preset."""
    return _parse_accesstier_preset(preset)[0] in ACCESS_TIER_MIXES


def _accesstier_present(mix: str) -> Tuple[List[str], np.ndarray]:
    """Return the tiers present in an access-tier mix and their stationary pmf."""
    weights = ACCESS_TIER_MIXES[mix]
    tiers = [t for t in ACCESS_TIERS if weights.get(t, 0.0) > 0.0]
    pi = np.array([weights[t] for t in tiers], dtype=float)
    pi = pi / pi.sum()
    return tiers, pi


def assign_accesstier_markov(n_users: int, n_trials: int,
                             rng: np.random.RandomState, preset: str,
                             stay_prob: Optional[float] = None
                             ) -> List[List[NetworkProfile]]:
    """Access-tier profiles whose tier follows a Markov chain across trials.

    Each user stays on its current access tier with probability ``stay_prob``
    and otherwise re-draws a tier from the preset's marginal mix (a lazy
    independence sampler whose stationary distribution is exactly that mix, so
    the marginal tier frequency matches the i.i.d. ``accesstier`` preset while
    adding inter-trial temporal correlation / mobility).  Within the chosen
    tier a fresh link state and delay bootstrap are drawn every trial.
    """
    if stay_prob is None:
        stay_prob = MARKOV_STAY_PROB
    mix, belief = _parse_accesstier_preset(preset)
    tiers, pi = _accesstier_present(mix)
    cum = np.cumsum(pi)
    state = np.array([int(np.searchsorted(cum, rng.uniform())) for _ in range(n_users)])
    state = np.clip(state, 0, len(tiers) - 1)
    out: List[List[NetworkProfile]] = []
    for _ in range(n_trials):
        out.append([_make_accesstier_profile(tiers[state[u]], rng, belief)
                    for u in range(n_users)])
        stay = rng.uniform(size=n_users) < stay_prob
        redraw = np.array([int(np.searchsorted(cum, rng.uniform()))
                           for _ in range(n_users)])
        redraw = np.clip(redraw, 0, len(tiers) - 1)
        state = np.where(stay, state, redraw)
    return out


def sample_accesstier_replay(n_users: int, n_trials: int,
                             rng: np.random.RandomState, preset: str
                             ) -> Tuple[List[NetworkProfile], np.ndarray, np.ndarray]:
    """Contiguous trace replay for the access-tier bank (mobility experiment).

    Each user is bound to one real ordered delay *sequence* of a tier (drawn
    from the preset mix) at a random offset; consecutive trials read contiguous
    samples from that sequence, preserving the real short-term autocorrelation
    of the link instead of i.i.d. bootstrapping.  Per-trial loss is applied
    with the tier's overall loss probability (``a = inf``).  Returns a single
    per-user :class:`NetworkProfile` list (fixed tier, tier-CDF belief) plus the
    realised ``arrival_ms``/``lost`` cubes of shape ``(n_users, n_trials)``.
    """
    mix, _belief = _parse_accesstier_preset(preset)
    tiers, pi = _accesstier_present(mix)
    cum = np.cumsum(pi)
    profiles: List[NetworkProfile] = []
    arrival_ms = np.zeros((n_users, n_trials), dtype=np.float64)
    lost = np.zeros((n_users, n_trials), dtype=bool)
    for u in range(n_users):
        ti = int(np.searchsorted(cum, rng.uniform()))
        ti = min(ti, len(tiers) - 1)
        tier = tiers[ti]
        bank = _load_tracebank()[tier]
        seqs = [s for s in bank.sequences if s.size > 0]
        if seqs:
            seq = np.asarray(seqs[int(rng.randint(len(seqs)))], dtype=np.float64)
        else:                                # fallback: pooled tier delays
            seq = np.asarray(bank.all_delays(), dtype=np.float64)
        start = int(rng.randint(seq.size))
        idx = (start + np.arange(n_trials)) % seq.size
        delay = seq[idx]
        loss = float(bank.overall_loss())
        loss_mask = rng.uniform(size=n_trials) < loss
        delay = delay.copy()
        delay[loss_mask] = np.inf
        arrival_ms[u] = delay
        lost[u] = loss_mask
        est_samples, est_loss = _belief_distribution(tier, "tier", None)
        p10 = float(np.percentile(seq, 10)) if seq.size else 0.0
        p90 = float(np.percentile(seq, 90)) if seq.size else 1.0
        if p90 <= p10:
            p90 = p10 + 1.0
        profiles.append(NetworkProfile(
            tier=tier, delay_range=(p10, p90), jitter_range=(0.0, 0.0),
            loss_prob=loss, trace_samples=seq, trace_loss=loss,
            state=None, est_samples=est_samples, est_loss=est_loss,
        ))
    return profiles, arrival_ms, lost


# ---------------------------------------------------------------------------
# Markov tier-transition (mobility / temporal-correlation) channel model
# ---------------------------------------------------------------------------

#: Default per-trial self-transition probability of the tier Markov chain.
#: ``stay_prob -> 1`` freezes each user's tier; ``stay_prob`` small approaches
#: the i.i.d. re-draw used by the standard presets.
MARKOV_STAY_PROB = 0.7


def build_tier_transition(stay_prob: float) -> np.ndarray:
    """Transition matrix of a symmetric nearest-neighbour walk on the tiers.

    The five tiers are ordered by quality (``TIER_ORDER``:
    ble-wifi-broadband-cellular-satellite).  At each step a user stays in
    the current tier with probability ``stay_prob`` and otherwise moves to an
    *adjacent* tier, modelling gradual connectivity change (mobility,
    handover) rather than instantaneous teleporting across tiers.

    Off-diagonal entries are symmetric and the boundaries reflect the missing
    move-mass into the self-loop, so the stationary distribution is uniform
    over the five tiers.  The marginal tier frequency therefore matches the
    ``uniform_all`` preset, isolating *temporal correlation* as the only
    difference between the i.i.d. and Markov channels.
    """
    n = len(TIER_ORDER)
    move = 1.0 - float(stay_prob)
    half = move / 2.0
    P = np.zeros((n, n), dtype=float)
    for i in range(n):
        if i > 0:
            P[i, i - 1] = half
        if i < n - 1:
            P[i, i + 1] = half
        P[i, i] = 1.0 - P[i].sum()
    return P


def assign_markov_tiers(n_users: int, n_trials: int,
                        rng: np.random.RandomState,
                        stay_prob: float = MARKOV_STAY_PROB
                        ) -> List[List[NetworkProfile]]:
    """Generate per-trial profiles whose tiers follow a Markov chain.

    Returns a list of length ``n_trials``; element ``t`` is a list of
    ``n_users`` :class:`NetworkProfile` objects for trial ``t``.  Each user's
    tier index evolves across trials according to :func:`build_tier_transition`
    (temporal correlation), while the user-specific delay window inside the
    tier is resampled every trial.  The initial tier is drawn from the uniform
    stationary distribution.
    """
    P = build_tier_transition(stay_prob)
    cdf = np.cumsum(P, axis=1)              # (n_tiers, n_tiers)
    n_t = len(TIER_ORDER)
    state = rng.randint(0, n_t, size=n_users)

    profiles_per_trial: List[List[NetworkProfile]] = []
    for _ in range(n_trials):
        prof_t = [_make_tier_profile(TIER_ORDER[state[u]], rng)
                  for u in range(n_users)]
        profiles_per_trial.append(prof_t)
        # Advance the chain one step (vectorised inverse-CDF sampling).
        u_rand = rng.uniform(size=n_users)
        nxt = (u_rand[:, None] > cdf[state]).sum(axis=1)
        state = np.clip(nxt, 0, n_t - 1)
    return profiles_per_trial


# ---------------------------------------------------------------------------
# Per-trial arrival generation
# ---------------------------------------------------------------------------

def sample_arrivals(profiles: Sequence[NetworkProfile], n_trials: int,
                    rng: np.random.RandomState
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Sample arrival times and loss indicators for every (user, trial).

    Returns
    -------
    arrival_ms : ndarray, shape (n_users, n_trials)
        Arrival time in milliseconds. ``np.inf`` if the response was lost.
    lost : ndarray, shape (n_users, n_trials), dtype=bool
        ``True`` if the user's response for that trial was lost.
    """
    n_users = len(profiles)
    arrival_ms = np.zeros((n_users, n_trials), dtype=np.float64)
    lost = np.zeros((n_users, n_trials), dtype=bool)

    for ui, p in enumerate(profiles):
        if p.trace_samples is not None:
            # Real-trace mode: bootstrap each trial's delay from this link's
            # measured latency samples; mark a real loss with prob trace_loss.
            idx = rng.randint(0, p.trace_samples.size, size=n_trials)
            delay = p.trace_samples[idx].astype(np.float64)
            loss_mask = rng.uniform(size=n_trials) < p.trace_loss
            delay[loss_mask] = np.inf
            arrival_ms[ui] = delay
            lost[ui] = loss_mask
            continue
        d_lo, d_hi = p.delay_range
        # Base delay is uniform inside [d_lo, d_hi]; add Gaussian jitter
        # whose standard deviation is half the jitter range mean.
        base = rng.uniform(d_lo, d_hi, size=n_trials)
        jitter_std = 0.5 * float(np.mean(p.jitter_range))
        jitter = rng.normal(loc=0.0, scale=jitter_std, size=n_trials)
        delay = np.clip(base + jitter, 0.0, None)

        loss_mask = rng.uniform(size=n_trials) < p.loss_prob
        delay[loss_mask] = np.inf
        arrival_ms[ui] = delay
        lost[ui] = loss_mask
    return arrival_ms, lost


# ---------------------------------------------------------------------------
# Probability of arriving before deadline (rho_i(t))
# ---------------------------------------------------------------------------

def arrival_probability(profile: NetworkProfile, t_now_ms: float,
                        deadline_ms: float) -> float:
    """Compute ``rho_i(t)`` = P(user response arrives within ``[t_now, D]``).

    Implements the closed-form approximation from the AWF design:

        remaining = D - t_now
        if remaining >= d_max:           rho = 1 - loss
        elif remaining < d_min:          rho = 0
        else:                            rho = (1 - loss) *
                                              (remaining - d_min) /
                                              (d_max - d_min)

    This is *not* the exact CDF of the (delay + jitter) sum; it is a
    deliberately simple, monotonically-increasing approximation that
    matches the experiment specification.
    """
    d_min, d_max = profile.delay_range
    remaining = float(deadline_ms - t_now_ms)
    p_link = 1.0 - float(profile.loss_prob)
    if remaining >= d_max:
        return p_link
    if remaining < d_min:
        return 0.0
    if d_max <= d_min:
        return p_link
    return p_link * (remaining - d_min) / (d_max - d_min)


def _norm_cdf(z: float) -> float:
    """Standard-normal CDF Phi(z) via the error function (no SciPy dep)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_pdf(z: float) -> float:
    """Standard-normal PDF phi(z)."""
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _delay_cdf(x: float, a: float, b: float, sigma: float) -> float:
    """CDF F_S(x) of S = U + N, U~Uniform(a,b), N~Normal(0, sigma^2).

    This is the *exact* distribution from which :func:`sample_arrivals`
    draws (base delay uniform in the user's delay window, plus Gaussian
    jitter), so it lets AWF reason about arrivals with the true model
    instead of the linear approximation in :func:`arrival_probability`.

    Closed form (verified by differentiation back to the convolution
    PDF ``(Phi((x-a)/s) - Phi((x-b)/s)) / (b-a)``):

        F_S(x) = 1/(b-a) [ (x-a)Phi((x-a)/s) - (x-b)Phi((x-b)/s)
                           + s( phi((x-a)/s) - phi((x-b)/s) ) ].
    """
    if sigma <= 0.0:
        # Degenerate to the uniform CDF (no jitter).
        if b <= a:
            return 1.0 if x >= a else 0.0
        return float(np.clip((x - a) / (b - a), 0.0, 1.0))
    if b <= a:
        # Degenerate base -> pure Gaussian around a.
        return _norm_cdf((x - a) / sigma)
    za = (x - a) / sigma
    zb = (x - b) / sigma
    val = ((x - a) * _norm_cdf(za) - (x - b) * _norm_cdf(zb)
           + sigma * (_norm_pdf(za) - _norm_pdf(zb))) / (b - a)
    return float(np.clip(val, 0.0, 1.0))


def arrival_probability_exact(profile: NetworkProfile, t_now_ms: float,
                              deadline_ms: float) -> float:
    """Exact conditional arrival probability with Bayes update.

    Unlike :func:`arrival_probability` (a deliberately cheap linear
    approximation), this computes the *proper* posterior probability that
    the response arrives in ``(t_now, D]`` **given that it has not arrived
    by ``t_now``**, using the true sampling distribution of
    :func:`sample_arrivals`::

        a = inf                       w.p. loss
        a = clip(U + N, 0, inf)       w.p. (1 - loss)

    with ``U ~ Uniform(d_min, d_max)`` and ``N ~ Normal(0, sigma^2)``,
    ``sigma = 0.5 * mean(jitter_range)``. Then

        rho_exact(t) = P(t < a <= D | a > t)
                     = (1-loss)(F_S(D) - F_S(t))
                       / [ loss + (1-loss)(1 - F_S(t)) ].

    The conditioning on ``a > t`` is exactly the "already not arrived"
    information that the linear approximation ignores.
    """
    d_min, d_max = profile.delay_range
    sigma = 0.5 * float(np.mean(profile.jitter_range))
    loss = float(profile.loss_prob)
    p_link = 1.0 - loss
    t = float(t_now_ms)
    D = float(deadline_ms)
    if D <= t:
        return 0.0

    F_t = _delay_cdf(t, d_min, d_max, sigma)
    F_D = _delay_cdf(D, d_min, d_max, sigma)

    numer = p_link * max(F_D - F_t, 0.0)
    denom = loss + p_link * (1.0 - F_t)
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(numer / denom, 0.0, 1.0))


def arrival_probability_empirical(profile: NetworkProfile, t_now_ms: float,
                                  deadline_ms: float) -> float:
    """Empirical-CDF arrival probability for the access-tier trace bank.

    Mirrors :func:`arrival_probability_exact` but replaces the parametric
    uniform+Gaussian CDF with the *empirical* CDF of the link's believed delay
    samples ``profile.est_samples`` (sorted) and its believed loss
    ``profile.est_loss``::

        rho_emp(t) = P(t < a <= D | a > t)
                   = (1-loss) (F_hat(D) - F_hat(t))
                     / [ loss + (1-loss)(1 - F_hat(t)) ],

    with ``F_hat`` the empirical CDF.  If ``est_samples`` is unset the call
    degrades to the linear approximation, so a ``belief-linear`` ablation arm
    transparently reuses :func:`arrival_probability`.
    """
    samples = profile.est_samples
    if samples is None or samples.size == 0:
        return arrival_probability(profile, t_now_ms, deadline_ms)
    t = float(t_now_ms)
    D = float(deadline_ms)
    if D <= t:
        return 0.0
    n = samples.size
    # samples are pre-sorted at construction time (np.searchsorted ECDF).
    F_t = float(np.searchsorted(samples, t, side="right")) / n
    F_D = float(np.searchsorted(samples, D, side="right")) / n
    loss = float(profile.est_loss)
    p_link = 1.0 - loss
    numer = p_link * max(F_D - F_t, 0.0)
    denom = loss + p_link * (1.0 - F_t)
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(numer / denom, 0.0, 1.0))



def expected_remaining_delay(profile: NetworkProfile,
                             t_now_ms: float, deadline_ms: float) -> float:
    """Return the deadline-truncated conditional residual waiting time.

    For trace-backed profiles this implements the quantity used in the paper,

        E[min(a - t, D - t) | a > t],

    from the empirical delay belief and its loss mass.  A lost response has
    ``a = inf`` and therefore contributes the full remaining deadline.  The
    conditioning discards empirical responses that would already have arrived
    by ``t`` and renormalizes over the surviving delay and loss mass.

    Profiles without an empirical belief retain the legacy range-based
    approximation, which is used only by the older synthetic presets.
    """
    t = float(t_now_ms)
    remaining = max(0.0, float(deadline_ms) - t)
    if remaining <= 0.0:
        return 0.0

    samples = profile.est_samples
    if samples is not None and samples.size:
        samples = np.asarray(samples, dtype=float)
        surviving = samples[samples > t]
        loss = float(np.clip(profile.est_loss, 0.0, 1.0))
        p_link = 1.0 - loss
        survival_fraction = float(surviving.size) / float(samples.size)
        denominator = loss + p_link * survival_fraction
        if denominator <= 1e-12:
            return 0.0

        observed_mass = 0.0
        if surviving.size:
            observed_mass = p_link * survival_fraction * float(
                np.mean(np.minimum(surviving - t, remaining)))
        return float((observed_mass + loss * remaining) / denominator)

    d_lo, d_hi = profile.delay_range
    mean_delay = 0.5 * (d_lo + d_hi)
    return min(mean_delay, remaining)
