"""
bcinet-fusion.py
================

All 18 fusion / waiting-policy methods used by the cBCI experiments.

The methods are grouped exactly as in the project specification:

* Group 1 - Conventional cBCI fusion baselines
    1. mean_fusion
    2. weighted_sum_fusion
    3. logpool_fusion
    4. majority_voting
    5. borda_count_fusion

* Group 2 - Simple network policy baselines
    6. fastest_only
    7. earliest_k_fusion
    8. deadline_partial_fusion
    9. latency_only_selection

* Group 3 - Reliability-based baselines
    10. best_user_only
    11. reliability_top_k_fusion
    12. reliability_weighted_deadline_fusion
    13. reliability_latency_score_selection

* Group 4 - Upper bound and ablation
    14. oracle_best_stopping
    15-18. Adaptive Waiting Fusion (AWF) and ablation variants

AWF (Adaptive Waiting Fusion) is the proposed method. It simultaneously
considers four signals at every event time:

    1. Decision sufficiency of currently-arrived responses (log-margin).
    2. Expected misclassification-risk reduction from each missing user.
    3. Probability that a missing user's response arrives before D.
    4. Cost of additional waiting (lambda_delay * expected remaining delay).

AWF is *not* claimed to be the optimal Bayes-stop policy; it is a fast,
interpretable adaptive waiting policy.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Hyphenated-module loader so that this file can ``import`` siblings.
# ---------------------------------------------------------------------------

def _load_sibling(name: str, filename: str):
    """Load a sibling .py file whose name contains hyphens."""
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(here, filename)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {filename}")
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec_module: needed by @dataclass on Python 3.13.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_NET = _load_sibling("bcinet_network", "bcinet-network.py")
NetworkProfile = _NET.NetworkProfile
arrival_probability = _NET.arrival_probability
arrival_probability_exact = _NET.arrival_probability_exact
arrival_probability_empirical = _NET.arrival_probability_empirical
expected_remaining_delay = _NET.expected_remaining_delay


# ---------------------------------------------------------------------------
# Container for one trial's fusion outcome
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    """Outcome of running a single fusion/waiting method on one trial."""
    pred: int
    decision_time_ms: float
    used_mask: np.ndarray              # (n_users,) bool
    stopping_reason: str               # "margin", "neg_value", "deadline", "fixed_set", ...
    extras: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

EPS = 1e-6  # default clipping epsilon for log-pool / log-score


def _ensure_simplex(p: np.ndarray) -> np.ndarray:
    """Project a non-negative vector onto the probability simplex."""
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if s <= 0.0:
        return np.full_like(p, 1.0 / len(p))
    return p / s


def _logpool(P_used: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Log-opinion-pool fusion: softmax over sum_i log clipped p_{i,c}."""
    if P_used.shape[0] == 0:
        return np.full(P_used.shape[1], 1.0 / P_used.shape[1])
    L = np.sum(np.log(np.clip(P_used, eps, 1.0)), axis=0)
    L -= L.max()
    q = np.exp(L)
    return q / q.sum()


def _mean(P_used: np.ndarray) -> np.ndarray:
    if P_used.shape[0] == 0:
        return np.full(P_used.shape[1], 1.0 / P_used.shape[1])
    return _ensure_simplex(P_used.mean(axis=0))


def _weighted_mean(P_used: np.ndarray, w_used: np.ndarray) -> np.ndarray:
    if P_used.shape[0] == 0:
        return np.full(P_used.shape[1], 1.0 / P_used.shape[1])
    w = np.asarray(w_used, dtype=float)
    if w.sum() <= 0:
        return _mean(P_used)
    w = w / w.sum()
    return _ensure_simplex((P_used * w[:, None]).sum(axis=0))


def _arrived_mask(arrivals_ms: np.ndarray, deadline_ms: float) -> np.ndarray:
    """Boolean mask of users whose response arrived <= deadline."""
    return np.isfinite(arrivals_ms) & (arrivals_ms <= deadline_ms)


def _empty_decision(n_classes: int, deadline_ms: float, n_users: int,
                    reason: str = "no_response") -> TrialResult:
    """Return a fallback decision for trials where no user contributes."""
    return TrialResult(
        pred=0,
        decision_time_ms=deadline_ms,
        used_mask=np.zeros(n_users, dtype=bool),
        stopping_reason=reason,
    )


# ===========================================================================
# Group 1 - Conventional cBCI fusion baselines (use deadline-arrived users)
# ===========================================================================

def mean_fusion(P: np.ndarray, arrivals_ms: np.ndarray,
                deadline_ms: float, **_) -> TrialResult:
    """Mean of arrived posteriors at the deadline."""
    n_users, n_classes = P.shape
    mask = _arrived_mask(arrivals_ms, deadline_ms)
    if not mask.any():
        return _empty_decision(n_classes, deadline_ms, n_users)
    q = _mean(P[mask])
    return TrialResult(pred=int(q.argmax()), decision_time_ms=deadline_ms,
                       used_mask=mask, stopping_reason="deadline")


def weighted_sum_fusion(P: np.ndarray, arrivals_ms: np.ndarray,
                        deadline_ms: float, val_acc: np.ndarray,
                        **_) -> TrialResult:
    """Weighted average of arrived posteriors using validation accuracy."""
    n_users, n_classes = P.shape
    mask = _arrived_mask(arrivals_ms, deadline_ms)
    if not mask.any():
        return _empty_decision(n_classes, deadline_ms, n_users)
    q = _weighted_mean(P[mask], val_acc[mask])
    return TrialResult(pred=int(q.argmax()), decision_time_ms=deadline_ms,
                       used_mask=mask, stopping_reason="deadline")


def logpool_fusion(P: np.ndarray, arrivals_ms: np.ndarray,
                   deadline_ms: float, eps: float = EPS, **_) -> TrialResult:
    """Sum log clipped probabilities of arrived users, then take argmax."""
    n_users, n_classes = P.shape
    mask = _arrived_mask(arrivals_ms, deadline_ms)
    if not mask.any():
        return _empty_decision(n_classes, deadline_ms, n_users)
    q = _logpool(P[mask], eps=eps)
    return TrialResult(pred=int(q.argmax()), decision_time_ms=deadline_ms,
                       used_mask=mask, stopping_reason="deadline")


def majority_voting(P: np.ndarray, arrivals_ms: np.ndarray,
                    deadline_ms: float, **_) -> TrialResult:
    """Plurality vote of argmax classes among arrived users."""
    n_users, n_classes = P.shape
    mask = _arrived_mask(arrivals_ms, deadline_ms)
    if not mask.any():
        return _empty_decision(n_classes, deadline_ms, n_users)
    votes = np.bincount(P[mask].argmax(axis=1), minlength=n_classes)
    return TrialResult(pred=int(votes.argmax()), decision_time_ms=deadline_ms,
                       used_mask=mask, stopping_reason="deadline")


def borda_count_fusion(P: np.ndarray, arrivals_ms: np.ndarray,
                       deadline_ms: float, **_) -> TrialResult:
    """Each user contributes Borda points equal to its rank of every class."""
    n_users, n_classes = P.shape
    mask = _arrived_mask(arrivals_ms, deadline_ms)
    if not mask.any():
        return _empty_decision(n_classes, deadline_ms, n_users)
    P_used = P[mask]
    # ranks: 0 = worst, n_classes-1 = best.  Using argsort of argsort.
    ranks = np.argsort(np.argsort(P_used, axis=1), axis=1)
    score = ranks.sum(axis=0)
    return TrialResult(pred=int(score.argmax()), decision_time_ms=deadline_ms,
                       used_mask=mask, stopping_reason="deadline")


# ===========================================================================
# Group 2 - Simple network policy baselines
# ===========================================================================

def fastest_only(P: np.ndarray, arrivals_ms: np.ndarray,
                 deadline_ms: float, **_) -> TrialResult:
    """Use only the very first arrived user's posterior."""
    n_users, n_classes = P.shape
    mask = _arrived_mask(arrivals_ms, deadline_ms)
    if not mask.any():
        return _empty_decision(n_classes, deadline_ms, n_users)
    order = np.argsort(arrivals_ms)
    first = order[0]
    if not mask[first]:  # safety
        first = int(np.where(mask)[0][np.argmin(arrivals_ms[mask])])
    used = np.zeros(n_users, dtype=bool)
    used[first] = True
    decision_time = float(arrivals_ms[first])
    return TrialResult(pred=int(P[first].argmax()),
                       decision_time_ms=decision_time,
                       used_mask=used, stopping_reason="fixed_set")


def earliest_k_fusion(P: np.ndarray, arrivals_ms: np.ndarray,
                      deadline_ms: float, k: int = 3, **_) -> TrialResult:
    """Mean fusion of the first ``k`` arrivals (using fewer if needed)."""
    n_users, n_classes = P.shape
    mask = _arrived_mask(arrivals_ms, deadline_ms)
    if not mask.any():
        return _empty_decision(n_classes, deadline_ms, n_users)
    arrived_idx = np.where(mask)[0]
    arrived_idx = arrived_idx[np.argsort(arrivals_ms[arrived_idx])]
    chosen = arrived_idx[:k]
    used = np.zeros(n_users, dtype=bool)
    used[chosen] = True
    decision_time = float(arrivals_ms[chosen].max())
    q = _mean(P[chosen])
    return TrialResult(pred=int(q.argmax()),
                       decision_time_ms=decision_time,
                       used_mask=used, stopping_reason="fixed_set")


def deadline_partial_fusion(P: np.ndarray, arrivals_ms: np.ndarray,
                            deadline_ms: float, **_) -> TrialResult:
    """Mean fusion of arrived users at the deadline (canonical real-time policy)."""
    return mean_fusion(P, arrivals_ms, deadline_ms)


def anytime_cbci(P: np.ndarray, arrivals_ms: np.ndarray,
                 deadline_ms: float,
                 theta: float = 0.80,
                 eps: float = EPS,
                 **_) -> TrialResult:
    """Anytime cBCI-style early stopping.

    After each arrival the fused log-pool posterior is computed. If the
    top-class probability exceeds *theta* the decision is committed
    immediately; otherwise we wait for the next arrival or the deadline.
    """
    n_users, n_classes = P.shape
    finite_mask = np.isfinite(arrivals_ms) & (arrivals_ms <= deadline_ms)
    if not finite_mask.any():
        return _empty_decision(n_classes, deadline_ms, n_users)

    order = [i for i in np.argsort(arrivals_ms) if finite_mask[i]]

    arrived: List[int] = []
    for i in order:
        arrived.append(i)
        L = np.sum(np.log(np.clip(P[arrived], eps, 1.0)), axis=0)
        L -= L.max()
        q = np.exp(L)
        q /= q.sum()
        if float(q.max()) >= theta:
            used = np.zeros(n_users, dtype=bool)
            used[arrived] = True
            return TrialResult(
                pred=int(q.argmax()),
                decision_time_ms=float(arrivals_ms[i]),
                used_mask=used,
                stopping_reason="margin",
            )

    # Deadline reached without confidence threshold being met
    L = np.sum(np.log(np.clip(P[arrived], eps, 1.0)), axis=0)
    L -= L.max()
    q = np.exp(L)
    q /= q.sum()
    used = np.zeros(n_users, dtype=bool)
    used[arrived] = True
    return TrialResult(
        pred=int(q.argmax()),
        decision_time_ms=deadline_ms,
        used_mask=used,
        stopping_reason="deadline",
    )


def anytime_cbci_cw(P: np.ndarray, arrivals_ms: np.ndarray,
                    deadline_ms: float, val_acc: np.ndarray,
                    theta: float = 0.80, eps: float = EPS, **_) -> TrialResult:
    """Anytime cBCI (Bhattacharyya et al.) confidence-weighted fusion.

    Faithful reproduction of the anytime collaborative-BCI fusion rule rather
    than a log-pool with a threshold.  As each user response arrives, the
    group decision is the *confidence-weighted* combination of the responses
    received so far, and an updated decision is available at every arrival
    (the anytime property).  User ``i`` contributes with weight

        w_i = val_acc_i * max_c p_{i,c},

    the product of a stable per-user reliability (the analog of the neural and
    behavioural confidence calibration of the original study) and the
    per-response decision confidence (the posterior peak).  The confidence-
    weighted group posterior is

        q(c)  proportional to  sum_{i in arrived} w_i * p_{i,c}.

    The original method has no intrinsic stopping rule, so to obtain a single
    operating point we commit as soon as the group confidence ``max_c q(c)``
    reaches a threshold ``theta`` calibrated on validation, and otherwise fuse
    whatever has arrived at the deadline.  The stopping is a calibrated add-on
    (labelled as such); the fusion itself is the original confidence-weighted
    rule.
    """
    n_users, n_classes = P.shape
    finite_mask = _arrived_mask(arrivals_ms, deadline_ms)
    if not finite_mask.any():
        return _empty_decision(n_classes, deadline_ms, n_users)
    rel = np.clip(np.asarray(val_acc, dtype=float), 0.0, None)
    order = [i for i in np.argsort(arrivals_ms) if finite_mask[i]]

    def _cw(idx: List[int]) -> np.ndarray:
        w = rel[idx] * P[idx].max(axis=1)          # per-response confidence
        if w.sum() <= 0.0:
            w = np.ones(len(idx))
        q = (P[idx] * w[:, None]).sum(axis=0)
        s = q.sum()
        return q / s if s > 0 else np.full(n_classes, 1.0 / n_classes)

    arrived: List[int] = []
    for i in order:
        arrived.append(i)
        q = _cw(arrived)
        if float(q.max()) >= theta:
            used = np.zeros(n_users, dtype=bool)
            used[arrived] = True
            return TrialResult(pred=int(q.argmax()),
                               decision_time_ms=float(arrivals_ms[i]),
                               used_mask=used, stopping_reason="margin")
    q = _cw(arrived)
    used = np.zeros(n_users, dtype=bool)
    used[arrived] = True
    return TrialResult(pred=int(q.argmax()), decision_time_ms=deadline_ms,
                       used_mask=used, stopping_reason="deadline")


def latency_only_selection(P: np.ndarray, arrivals_ms: np.ndarray,
                           deadline_ms: float,
                           profiles: Sequence[NetworkProfile],
                           k: int = 3, **_) -> TrialResult:
    """Pick the k users with the smallest expected delay; use those that arrive.

    The user ranking is by *expected* delay (mean of (d_min, d_max)),
    not by realized arrival, so this represents an a-priori scheduler
    that knows network tiers but not realized jitter.
    """
    n_users, n_classes = P.shape
    expected = np.array([0.5 * (p.delay_range[0] + p.delay_range[1])
                         for p in profiles])
    candidate = np.argsort(expected)[:k]
    arrived = _arrived_mask(arrivals_ms, deadline_ms)
    chosen = [i for i in candidate if arrived[i]]
    if not chosen:
        return _empty_decision(n_classes, deadline_ms, n_users)
    used = np.zeros(n_users, dtype=bool)
    used[chosen] = True
    decision_time = float(arrivals_ms[chosen].max())
    q = _mean(P[chosen])
    return TrialResult(pred=int(q.argmax()),
                       decision_time_ms=decision_time,
                       used_mask=used, stopping_reason="fixed_set")


# ===========================================================================
# Group 3 - Reliability-based baselines
# ===========================================================================

def best_user_only(P: np.ndarray, arrivals_ms: np.ndarray,
                   deadline_ms: float, val_acc: np.ndarray,
                   **_) -> TrialResult:
    """Use only the single highest-accuracy user (waits for them)."""
    n_users, n_classes = P.shape
    best = int(np.argmax(val_acc))
    arrived = np.isfinite(arrivals_ms[best]) and arrivals_ms[best] <= deadline_ms
    used = np.zeros(n_users, dtype=bool)
    if arrived:
        used[best] = True
        return TrialResult(pred=int(P[best].argmax()),
                           decision_time_ms=float(arrivals_ms[best]),
                           used_mask=used, stopping_reason="fixed_set")
    return _empty_decision(n_classes, deadline_ms, n_users, reason="late_lost")


def reliability_top_k_fusion(P: np.ndarray, arrivals_ms: np.ndarray,
                             deadline_ms: float, val_acc: np.ndarray,
                             k: int = 3, **_) -> TrialResult:
    """Mean fusion of the top-k accuracy users that arrive in time."""
    n_users, n_classes = P.shape
    top = np.argsort(-val_acc)[:k]
    arrived = _arrived_mask(arrivals_ms, deadline_ms)
    chosen = [i for i in top if arrived[i]]
    if not chosen:
        return _empty_decision(n_classes, deadline_ms, n_users)
    used = np.zeros(n_users, dtype=bool)
    used[chosen] = True
    decision_time = float(arrivals_ms[chosen].max())
    q = _mean(P[chosen])
    return TrialResult(pred=int(q.argmax()),
                       decision_time_ms=decision_time,
                       used_mask=used, stopping_reason="fixed_set")


def reliability_weighted_deadline_fusion(P: np.ndarray, arrivals_ms: np.ndarray,
                                         deadline_ms: float,
                                         val_acc: np.ndarray, **_) -> TrialResult:
    """Validation-accuracy-weighted average of arrived users at the deadline."""
    return weighted_sum_fusion(P, arrivals_ms, deadline_ms, val_acc=val_acc)


def reliability_latency_score_selection(P: np.ndarray, arrivals_ms: np.ndarray,
                                        deadline_ms: float,
                                        val_acc: np.ndarray,
                                        profiles: Sequence[NetworkProfile],
                                        k: Optional[int] = None,
                                        **_) -> TrialResult:
    """Select users by score = val_acc * P(arrive in time)."""
    n_users, n_classes = P.shape
    rho = np.array([arrival_probability(p, t_now_ms=0.0,
                                        deadline_ms=deadline_ms)
                    for p in profiles])
    score = val_acc * rho
    if k is None:
        # Choose all users with positive score, ranked by score.
        order = np.argsort(-score)
        cutoff = max(1, int(np.sum(score > 0.5 * score.max())))
        candidate = order[:cutoff]
    else:
        candidate = np.argsort(-score)[:k]
    arrived = _arrived_mask(arrivals_ms, deadline_ms)
    chosen = [i for i in candidate if arrived[i]]
    if not chosen:
        return _empty_decision(n_classes, deadline_ms, n_users)
    used = np.zeros(n_users, dtype=bool)
    used[chosen] = True
    q = _weighted_mean(P[chosen], val_acc[chosen])
    # Decision time is the deadline: this method selects users up-front but
    # still waits until D before committing (it is not an early-stopping policy).
    return TrialResult(pred=int(q.argmax()),
                       decision_time_ms=deadline_ms,
                       used_mask=used, stopping_reason="fixed_set")


# ===========================================================================
# Group 3b - Network / value-of-information waiting baselines
# ===========================================================================

def deadline_aware_waiting(P: np.ndarray, arrivals_ms: np.ndarray,
                           deadline_ms: float,
                           profiles: Sequence[NetworkProfile],
                           tau: float = 0.05, eps: float = EPS,
                           **_) -> TrialResult:
    """Network-only (decision-blind) deadline-aware / min-delay waiting.

    The fuser keeps waiting as long as at least one *missing* user is still
    likely to arrive before the deadline (arrival probability ``>= tau``),
    and commits the log-pool of the arrived users as soon as every missing
    user is unlikely to make it in time. The stopping time therefore depends
    only on the network arrival statistics -- never on the BCI belief -- so
    this isolates whether looking at network delay alone is enough.
    """
    n_users, n_classes = P.shape
    finite_mask = _arrived_mask(arrivals_ms, deadline_ms)
    if not finite_mask.any():
        return _empty_decision(n_classes, deadline_ms, n_users)
    order = [i for i in np.argsort(arrivals_ms) if finite_mask[i]]

    arrived: List[int] = []
    for i in order:
        arrived.append(i)
        t = float(arrivals_ms[i])
        arrived_set = set(arrived)
        missing = [j for j in range(n_users) if j not in arrived_set]
        still_likely = any(
            arrival_probability(profiles[j], t, deadline_ms) >= tau
            for j in missing)
        if not still_likely:
            q = _logpool(P[arrived], eps=eps)
            used = np.zeros(n_users, dtype=bool)
            used[arrived] = True
            return TrialResult(pred=int(q.argmax()), decision_time_ms=t,
                               used_mask=used, stopping_reason="neg_value")

    q = _logpool(P[arrived], eps=eps)
    used = np.zeros(n_users, dtype=bool)
    used[arrived] = True
    return TrialResult(pred=int(q.argmax()), decision_time_ms=deadline_ms,
                       used_mask=used, stopping_reason="deadline")


def da_os_stopping(P: np.ndarray, arrivals_ms: np.ndarray,
                   deadline_ms: float,
                   profiles: Sequence[NetworkProfile],
                   beta: Optional[float] = None,
                   lambda_delay: float = 1e-5, u: float = 1.0,
                   eps: float = EPS, **_) -> TrialResult:
    """Deadline-aware optimal stopping (adapted from He and Zhang).

    Reproduces the infinitesimal (one-stage) look-ahead stopping rule of He
    and Zhang's deadline-aware synchrophasor-data concentration, transferred
    to collaborative-BCI response collection.  The fusion server treats each
    arrived response as one unit of collected *throughput* and, using the
    known per-link arrival-delay distributions, keeps waiting while the
    expected marginal throughput gain of continuing still offsets the marginal
    deadline (latency) cost::

        continue  iff  u * sum_{j in missing} rho_j(t)
                        >  beta * min_{j in missing} E_delay_j(t),

    where ``rho_j(t)`` is the conditional probability that missing user ``j``
    arrives before the deadline and ``E_delay_j(t)`` is its expected remaining
    delay.  Because ``rho_j`` decreases as ``t`` approaches the deadline, the
    continuation value is monotone, which is exactly the case where the
    one-stage look-ahead rule is optimal in He and Zhang.

    The rule is decision-blind: its value is the *number* of additional
    responses, not their effect on class-decision risk, which is the defining
    difference from AWF.  At the stopping time the arrived posteriors are fused
    with the same log-opinion pool used by AWF, so any accuracy or latency
    difference is attributable solely to the stopping value (throughput vs
    decision risk).

    Reward mapping (stated for transparency): one response equals one unit of
    He and Zhang's collected-data throughput (``u``), and ``beta`` is the
    throughput-per-millisecond exchange rate that plays the role of their
    deadline penalty.  We default ``beta = lambda_delay`` so DA-OS and AWF
    price latency on the same scale.
    """
    n_users, n_classes = P.shape
    finite_mask = _arrived_mask(arrivals_ms, deadline_ms)
    if not finite_mask.any():
        return _empty_decision(n_classes, deadline_ms, n_users)
    if beta is None:
        beta = lambda_delay
    order = [i for i in np.argsort(arrivals_ms) if finite_mask[i]]

    arrived: List[int] = []
    for i in order:
        arrived.append(i)
        t = float(arrivals_ms[i])
        arrived_set = set(arrived)
        missing = [j for j in range(n_users) if j not in arrived_set]
        if not missing:
            break
        exp_gain = u * sum(
            arrival_probability_empirical(profiles[j], t, deadline_ms)
            for j in missing)
        exp_cost = beta * min(
            expected_remaining_delay(profiles[j], t, deadline_ms)
            for j in missing)
        if exp_gain <= exp_cost:      # one-stage look-ahead: stop now
            q = _logpool(P[arrived], eps=eps)
            used = np.zeros(n_users, dtype=bool)
            used[arrived] = True
            return TrialResult(pred=int(q.argmax()), decision_time_ms=t,
                               used_mask=used, stopping_reason="neg_value")

    q = _logpool(P[arrived], eps=eps)
    used = np.zeros(n_users, dtype=bool)
    used[arrived] = True
    return TrialResult(pred=int(q.argmax()), decision_time_ms=deadline_ms,
                       used_mask=used, stopping_reason="deadline")


def _expected_entropy_after_user(q_t: np.ndarray, conf_i: np.ndarray,
                                 eps: float = EPS) -> float:
    """Expected (unnormalized) Shannon entropy of the belief after user i,
    under the one-step confusion-matrix lookahead.

    Returns Sum_z P(z|q) * H(q(.|z)), the analog of ``_expected_risk_after_user``
    with entropy replacing zero-one risk.
    """
    n_classes = len(q_t)
    pz = q_t @ conf_i
    exp_H = 0.0
    for z in range(n_classes):
        if pz[z] <= 0.0:
            continue
        post = q_t * conf_i[:, z]
        s = post.sum()
        if s <= 0:
            continue
        post = post / s
        exp_H += pz[z] * (-float(np.sum(post * np.log(np.clip(post, eps, 1.0)))))
    return float(exp_H)


def generic_voi_waiting(P: np.ndarray, arrivals_ms: np.ndarray,
                        deadline_ms: float,
                        profiles: Sequence[NetworkProfile],
                        confusions: Sequence[np.ndarray],
                        alpha: float = 1e-5, lambda_delay: float = 1e-5,
                        eps: float = EPS, **_) -> TrialResult:
    """Entropy-gain waiting: the task-agnostic uncertainty-reduction analog of
    AWF.

    Identical to AWF except that the per-user decision value is the expected
    reduction of the (normalized) Shannon entropy of the belief rather than the
    Bayes decision-risk reduction:

        V_j(t) = rho_j(t) * dH_j(t) - lambda_delay * E_delay_j(t),
        dH_j(t) = [ H(q_t) - E_z H(q_t^j(.|z)) ] / log C  in [0, 1].

    The one-step lookahead uses the same confusion matrices F_j and the same
    margin early-stop as AWF, so the only difference from AWF is entropy vs.
    decision risk. This isolates whether generic uncertainty reduction is
    enough, or whether the class- and user-specific decision-risk reduction of
    AWF is needed. Because dH_j is normalized to [0, 1] like AWF's Delta R_j,
    both policies use the same delay-cost weight lambda on a comparable scale.
    """
    n_users, n_classes = P.shape
    if not np.isfinite(arrivals_ms).any():
        return _empty_decision(n_classes, deadline_ms, n_users)
    A_thresh = np.log(max(n_classes - 1, 1) / alpha)
    logC = np.log(n_classes) if n_classes > 1 else 1.0
    conf_list = list(confusions)
    finite_mask = _arrived_mask(arrivals_ms, deadline_ms)
    order = [i for i in np.argsort(arrivals_ms) if finite_mask[i]]

    arrived: List[int] = []
    for i in order:
        arrived.append(i)
        t = float(arrivals_ms[i])
        L = np.sum(np.log(np.clip(P[arrived], eps, 1.0)), axis=0)
        Lc = L - L.max()
        q = np.exp(Lc)
        q /= q.sum()
        c_star = int(q.argmax())

        L_sorted = np.sort(L)[::-1]
        margin = float(L_sorted[0] - L_sorted[1]) if n_classes >= 2 else np.inf
        if margin >= A_thresh:
            used = np.zeros(n_users, dtype=bool)
            used[arrived] = True
            return TrialResult(pred=c_star, decision_time_ms=t, used_mask=used,
                               stopping_reason="margin")

        H_now = -float(np.sum(q * np.log(np.clip(q, eps, 1.0))))
        arrived_set = set(arrived)
        missing = [j for j in range(n_users) if j not in arrived_set]
        if not missing:
            used = np.zeros(n_users, dtype=bool)
            used[arrived] = True
            return TrialResult(pred=c_star, decision_time_ms=t, used_mask=used,
                               stopping_reason="all_arrived")

        v_max = -np.inf
        for j in missing:
            rho = arrival_probability(profiles[j], t, deadline_ms)
            if rho <= 0.0:
                continue
            exp_H_after = _expected_entropy_after_user(q, conf_list[j], eps=eps)
            dH = max(H_now - exp_H_after, 0.0) / logC
            delay_cost = lambda_delay * expected_remaining_delay(
                profiles[j], t, deadline_ms)
            v_max = max(v_max, rho * dH - delay_cost)

        if v_max > 0:
            continue
        used = np.zeros(n_users, dtype=bool)
        used[arrived] = True
        return TrialResult(pred=c_star, decision_time_ms=t, used_mask=used,
                           stopping_reason="neg_value")

    if not arrived:
        return _empty_decision(n_classes, deadline_ms, n_users)
    L = np.sum(np.log(np.clip(P[arrived], eps, 1.0)), axis=0)
    Lc = L - L.max()
    q = np.exp(Lc)
    q /= q.sum()
    used = np.zeros(n_users, dtype=bool)
    used[arrived] = True
    return TrialResult(pred=int(q.argmax()), decision_time_ms=deadline_ms,
                       used_mask=used, stopping_reason="deadline")


def neurobci_sc_fusion(P: np.ndarray, arrivals_ms: np.ndarray,
                       deadline_ms: float, val_acc: np.ndarray,
                       eps: float = EPS, **_) -> TrialResult:
    """NeuroBCI-style semantic-communication fusion (full-wait, no stopping).

    Represents a recent networked / semantic-BCI system (e.g. NeuroBCI /
    BRIEDGE) in which each user transmits a semantic posterior and the edge
    server fuses every response that arrives by the deadline via a
    reliability-weighted log-opinion pool:

        c_hat = argmax_c  sum_{i in A_D} w_i * log p_{i,c},   w_i = val_acc_i.

    Crucially it performs *no* trial-level online waiting decision (no V_j):
    it always waits until the deadline and fuses whatever arrived. This is the
    system-level comparison point, not a direct online-stopping competitor.
    """
    n_users, n_classes = P.shape
    mask = _arrived_mask(arrivals_ms, deadline_ms)
    if not mask.any():
        return _empty_decision(n_classes, deadline_ms, n_users)
    w = np.clip(np.asarray(val_acc, dtype=float)[mask], 0.0, None)
    if w.sum() <= 0.0:
        w = np.ones(int(mask.sum()))
    L = np.sum(w[:, None] * np.log(np.clip(P[mask], eps, 1.0)), axis=0)
    L -= L.max()
    q = np.exp(L)
    q /= q.sum()
    return TrialResult(pred=int(q.argmax()), decision_time_ms=deadline_ms,
                       used_mask=mask, stopping_reason="deadline")


# ===========================================================================
# Group 4 - Oracle and AWF
# ===========================================================================

def oracle_best_stopping(P: np.ndarray, arrivals_ms: np.ndarray,
                         deadline_ms: float, y_true: int,
                         lambda_delay: float = 1e-3, **_) -> TrialResult:
    """Pick the stopping point that maximises (1 if correct else 0) - lambda*t.

    This is an *upper bound* policy that uses the trial's true label and
    is therefore not realisable in practice. We evaluate every candidate
    stopping time (each arrival of an arrived user) and choose the one
    that maximises the accuracy-latency objective.
    """
    n_users, n_classes = P.shape
    arrived_idx = np.where(_arrived_mask(arrivals_ms, deadline_ms))[0]
    if len(arrived_idx) == 0:
        return _empty_decision(n_classes, deadline_ms, n_users)
    arrived_idx = arrived_idx[np.argsort(arrivals_ms[arrived_idx])]

    best_obj = -np.inf
    best_pred = 0
    best_time = deadline_ms
    best_used = np.zeros(n_users, dtype=bool)
    cumulative = []
    for i in arrived_idx:
        cumulative.append(i)
        q = _logpool(P[cumulative])
        pred = int(q.argmax())
        t = float(arrivals_ms[i])
        obj = (1.0 if pred == int(y_true) else 0.0) - lambda_delay * t
        if obj > best_obj:
            best_obj = obj
            best_pred = pred
            best_time = t
            best_used = np.zeros(n_users, dtype=bool)
            best_used[cumulative] = True
    return TrialResult(pred=best_pred, decision_time_ms=best_time,
                       used_mask=best_used, stopping_reason="oracle")


# ---------------------------------------------------------------------------
# AWF and ablations
# ---------------------------------------------------------------------------

def _expected_risk_after_user(q_t: np.ndarray, conf_i: np.ndarray
                              ) -> Tuple[float, float]:
    """Compute Expected_R_i and the risk-reduction from a missing user i.

    Parameters
    ----------
    q_t : (n_classes,) belief BEFORE seeing user i.
    conf_i : (n_classes, n_classes) row-normalized P_i(z = col | c = row).

    Returns
    -------
    expected_R_i : float
        Sum_z P(z|q) * (1 - max_c q(c|z))
    rstop_now : float
        Convenience: 1 - max(q_t)
    """
    n_classes = len(q_t)
    rstop_now = 1.0 - float(q_t.max())
    # P(z | q_t) = sum_c q(c) * P_i(z|c) -> shape (n_classes,)
    pz = q_t @ conf_i
    expected_R = 0.0
    for z in range(n_classes):
        if pz[z] <= 0.0:
            continue
        # q(c|z) ∝ q_t(c) * P_i(z|c)
        post = q_t * conf_i[:, z]
        s = post.sum()
        if s <= 0:
            continue
        post = post / s
        expected_R += pz[z] * (1.0 - float(post.max()))
    return float(expected_R), rstop_now


def awf_fusion(P: np.ndarray, arrivals_ms: np.ndarray,
               deadline_ms: float,
               profiles: Sequence[NetworkProfile],
               confusions: Sequence[np.ndarray],
               alpha: float = 1e-3,
               lambda_delay: float = 1e-3,
               eps: float = EPS,
               use_delay_cost: bool = True,
               use_reliability: bool = True,
               use_arrival_prob: bool = True,
               exact_arrival: bool = False,
               pred_pool: str = "log",
               **_) -> TrialResult:
    """Adaptive Waiting Fusion (AWF).

    At every event time (= arrival of a user, or the deadline) the
    central fuser checks two things:

    1. Is the current decision sufficiently confident? Concretely, is
       the log-score margin between the top-1 and top-2 candidates at
       least ``A = log((C-1)/alpha)``? If yes, stop with margin reason.

    2. Otherwise, for each missing user i still potentially arriving in
       time, compute

           V_i(t) = rho_i(t) * (R_stop(t) - Expected_R_i(t))
                    - lambda_delay * E_delay_i(t)

       If ``max_i V_i(t) > 0`` we keep waiting; else stop with reason
       ``neg_value``. At the deadline we always stop.

    Ablation flags
    --------------
    use_delay_cost : if False, drop the lambda_delay term (AWF w/o delay cost).
    use_reliability : if False, replace each P_i with a uniform-on-correct
        confusion matrix (= mean accuracy on diagonal) -> AWF w/o reliability.
    use_arrival_prob : if False, set rho_i(t) = 1 -> AWF w/o arrival prob.
    exact_arrival : if True, replace the linear rho approximation with the
        exact Bayes-updated conditional arrival probability
        ``P(t < a_i <= D | a_i > t)`` derived from the true sampling
        distribution -> AWF (exact rho).
    """
    n_users, n_classes = P.shape
    if not np.isfinite(arrivals_ms).any():
        return _empty_decision(n_classes, deadline_ms, n_users)

    A_thresh = np.log(max(n_classes - 1, 1) / alpha)

    # rho estimator: exact parametric (uniform+Gaussian Bayes update) when
    # requested, otherwise the empirical-CDF estimator -- which transparently
    # degrades to the linear ramp for profiles without a trace-bank belief
    # (``est_samples is None``), so non-access-tier presets are unchanged.
    rho_fn = arrival_probability_exact if exact_arrival else arrival_probability_empirical

    # Order of arrivals (only those that beat the deadline; lost = inf)
    finite_mask = np.isfinite(arrivals_ms) & (arrivals_ms <= deadline_ms)
    order = np.argsort(arrivals_ms)
    order = [i for i in order if finite_mask[i]]

    # Optional reliability-ablation: uniform confusion matrices --------------
    if use_reliability:
        conf_list = list(confusions)
    else:
        diag = 1.0 / n_classes
        # Construct an "uninformative-but-still-row-normalized" matrix.
        uniform_conf = np.full((n_classes, n_classes), (1.0 - diag) / max(n_classes - 1, 1))
        np.fill_diagonal(uniform_conf, diag)
        conf_list = [uniform_conf for _ in confusions]

    arrived: List[int] = []
    waiting_values: List[float] = []
    risk_reductions: List[float] = []
    waited_rhos: List[float] = []

    def _belief(arrived_idx: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute log-score L and softmax belief q from arrived users."""
        if not arrived_idx:
            L = np.zeros(n_classes)
        else:
            L = np.sum(np.log(np.clip(P[arrived_idx], eps, 1.0)), axis=0)
        Lc = L - L.max()
        q = np.exp(Lc)
        q = q / q.sum()
        return L, q

    def _pred(arrived_idx: List[int], q_log: np.ndarray) -> int:
        """Committed class. ``pred_pool='linear'`` reports the linear-opinion
        pool of the same arrived posteriors, which isolates the fusion rule
        from the (log-belief-driven) stopping decisions."""
        if pred_pool == "linear" and arrived_idx:
            return int(np.mean(P[arrived_idx], axis=0).argmax())
        return int(q_log.argmax())

    # Walk through events one at a time --------------------------------------
    for ev_idx, i in enumerate(order):
        arrived.append(i)
        t = float(arrivals_ms[i])
        L, q = _belief(arrived)
        c_star = _pred(arrived, q)
        rstop = 1.0 - float(q.max())

        # Margin in log-score space (top-1 minus top-2) ----------------------
        L_sorted = np.sort(L)[::-1]
        margin = float(L_sorted[0] - L_sorted[1]) if n_classes >= 2 else np.inf

        if margin >= A_thresh:
            used = np.zeros(n_users, dtype=bool)
            used[arrived] = True
            return TrialResult(
                pred=c_star, decision_time_ms=t, used_mask=used,
                stopping_reason="margin",
                extras={"mean_waiting_value": float(np.mean(waiting_values)) if waiting_values else 0.0,
                        "mean_risk_reduction": float(np.mean(risk_reductions)) if risk_reductions else 0.0,
                        "mean_waited_rho": float(np.mean(waited_rhos)) if waited_rhos else 0.0,
                        "arrival_count": float(len(arrived))},
            )

        # Compute V_i(t) over still-missing users ---------------------------
        # AWF must be strictly causal: at decision time t we do NOT know
        # whether a missing user will eventually arrive. We therefore
        # consider every user not yet arrived and let rho_i(t) handle the
        # probability of arriving before the deadline.
        arrived_set = set(arrived)
        missing = [j for j in range(n_users) if j not in arrived_set]

        if not missing:
            used = np.zeros(n_users, dtype=bool)
            used[arrived] = True
            return TrialResult(
                pred=c_star, decision_time_ms=t, used_mask=used,
                stopping_reason="all_arrived",
                extras={"mean_waiting_value": float(np.mean(waiting_values)) if waiting_values else 0.0,
                        "mean_risk_reduction": float(np.mean(risk_reductions)) if risk_reductions else 0.0,
                        "mean_waited_rho": float(np.mean(waited_rhos)) if waited_rhos else 0.0,
                        "arrival_count": float(len(arrived))},
            )

        v_per_user = []
        for j in missing:
            rho = (1.0 if not use_arrival_prob
                   else rho_fn(profiles[j], t, deadline_ms))
            if rho <= 0.0 and use_arrival_prob:
                v_per_user.append(-np.inf)
                continue
            exp_R, _ = _expected_risk_after_user(q, conf_list[j])
            risk_red = rstop - exp_R   # >= 0 by data-processing arguments
            delay_cost = (lambda_delay * expected_remaining_delay(profiles[j], t, deadline_ms)
                          if use_delay_cost else 0.0)
            v = rho * risk_red - delay_cost
            v_per_user.append(v)

        v_max = float(np.max(v_per_user)) if v_per_user else -np.inf
        waiting_values.append(v_max)

        if v_max > 0:
            # Track which user we'd be waiting for and remember its rho.
            j_star = missing[int(np.argmax(v_per_user))]
            rho_star = (1.0 if not use_arrival_prob
                        else rho_fn(profiles[j_star], t, deadline_ms))
            waited_rhos.append(rho_star)
            exp_R_star, _ = _expected_risk_after_user(q, conf_list[j_star])
            risk_reductions.append(rstop - exp_R_star)
            continue  # keep waiting -> next arrival event
        else:
            used = np.zeros(n_users, dtype=bool)
            used[arrived] = True
            return TrialResult(
                pred=c_star, decision_time_ms=t, used_mask=used,
                stopping_reason="neg_value",
                extras={"mean_waiting_value": float(np.mean(waiting_values)) if waiting_values else 0.0,
                        "mean_risk_reduction": float(np.mean(risk_reductions)) if risk_reductions else 0.0,
                        "mean_waited_rho": float(np.mean(waited_rhos)) if waited_rhos else 0.0,
                        "arrival_count": float(len(arrived))},
            )

    # Reached the deadline (or ran out of arrivals) ------------------------
    if not arrived:
        return _empty_decision(n_classes, deadline_ms, n_users)
    L, q = _belief(arrived)
    used = np.zeros(n_users, dtype=bool)
    used[arrived] = True
    return TrialResult(
        pred=_pred(arrived, q), decision_time_ms=deadline_ms,
        used_mask=used, stopping_reason="deadline",
        extras={"mean_waiting_value": float(np.mean(waiting_values)) if waiting_values else 0.0,
                "mean_risk_reduction": float(np.mean(risk_reductions)) if risk_reductions else 0.0,
                "mean_waited_rho": float(np.mean(waited_rhos)) if waited_rhos else 0.0,
                "arrival_count": float(len(arrived))},
    )


# ---------------------------------------------------------------------------
# Constrained-uplink active polling (V_j-driven scheduling)
# ---------------------------------------------------------------------------

def _polling_core(P: np.ndarray, arrivals_ms: np.ndarray, deadline_ms: float,
                  profiles: Sequence[NetworkProfile],
                  confusions: Sequence[np.ndarray],
                  priority: Callable[[int, float, Dict[int, float]], float],
                  poll_budget: int, n_rounds: int,
                  alpha: float, lambda_delay: float,
                  eps: float = EPS) -> TrialResult:
    """Shared round-based polling loop for constrained-uplink fusion.

    The uplink admits at most ``poll_budget`` responses per round.  In each
    of ``n_rounds`` equally spaced rounds over ``[0, D]`` the fuser ranks the
    still-missing users by ``priority`` (lower key = polled first), polls the
    top ``poll_budget`` of them, and admits any whose result is already ready
    (``arrivals_ms[j] <= t``).  Both polling policies share the *same* AWF
    stop rules (margin / non-positive waiting value / deadline), so the only
    difference between them is *which* users they choose to poll -- isolating
    the value of using the waiting value ``V_j`` as the scheduling priority.
    """
    n_users, n_classes = P.shape
    if not np.isfinite(arrivals_ms).any():
        return _empty_decision(n_classes, deadline_ms, n_users)
    A_thresh = np.log(max(n_classes - 1, 1) / alpha)
    conf_list = list(confusions)
    dt = deadline_ms / n_rounds
    fused: List[int] = []
    fused_set: set = set()
    waiting_values: List[float] = []

    def _belief(idx: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        if not idx:
            L = np.zeros(n_classes)
        else:
            L = np.sum(np.log(np.clip(P[idx], eps, 1.0)), axis=0)
        Lc = L - L.max()
        q = np.exp(Lc)
        q = q / q.sum()
        return L, q

    def _decide(t: float, reason: str) -> TrialResult:
        _, q = _belief(fused)
        used = np.zeros(n_users, dtype=bool)
        used[fused] = True
        return TrialResult(
            pred=int(q.argmax()), decision_time_ms=t, used_mask=used,
            stopping_reason=reason,
            extras={"arrival_count": float(len(fused)),
                    "mean_waiting_value": (float(np.mean(waiting_values))
                                           if waiting_values else 0.0),
                    "poll_budget": float(poll_budget)},
        )

    for r in range(1, n_rounds + 1):
        t = r * dt
        # ---- waiting values over still-missing users (for ranking + stop) --
        _, q = _belief(fused)
        rstop = 1.0 - float(q.max())
        missing = [j for j in range(n_users) if j not in fused_set]
        v_map: Dict[int, float] = {}
        for j in missing:
            rho = arrival_probability(profiles[j], t, deadline_ms)
            if rho <= 0.0:
                v_map[j] = -np.inf
                continue
            exp_R, _ = _expected_risk_after_user(q, conf_list[j])
            delay_cost = lambda_delay * expected_remaining_delay(
                profiles[j], t, deadline_ms)
            v_map[j] = rho * (rstop - exp_R) - delay_cost

        # ---- poll the top-`poll_budget` missing users by priority ----------
        if missing:
            order = sorted(missing, key=lambda j: priority(j, t, v_map))
            for j in order[:max(1, poll_budget)]:
                if np.isfinite(arrivals_ms[j]) and arrivals_ms[j] <= t:
                    fused.append(j)
                    fused_set.add(j)

        # ---- stop rule 1: log-score margin ---------------------------------
        if fused:
            L, _ = _belief(fused)
            L_sorted = np.sort(L)[::-1]
            margin = (float(L_sorted[0] - L_sorted[1])
                      if n_classes >= 2 else np.inf)
            if margin >= A_thresh:
                return _decide(t, "margin")

        # ---- stop rule 2: non-positive waiting value over missing ----------
        missing2 = [j for j in range(n_users) if j not in fused_set]
        if not missing2:
            return _decide(t, "all_arrived")
        _, q2 = _belief(fused)
        rstop2 = 1.0 - float(q2.max())
        v_max = -np.inf
        for j in missing2:
            rho = arrival_probability(profiles[j], t, deadline_ms)
            if rho <= 0.0:
                continue
            exp_R, _ = _expected_risk_after_user(q2, conf_list[j])
            delay_cost = lambda_delay * expected_remaining_delay(
                profiles[j], t, deadline_ms)
            v_max = max(v_max, rho * (rstop2 - exp_R) - delay_cost)
        waiting_values.append(v_max)
        if v_max <= 0.0:
            return _decide(t, "neg_value")

    # ---- deadline ----------------------------------------------------------
    if not fused:
        return _empty_decision(n_classes, deadline_ms, n_users)
    return _decide(deadline_ms, "deadline")


def awf_poll_fusion(P: np.ndarray, arrivals_ms: np.ndarray, deadline_ms: float,
                    profiles: Sequence[NetworkProfile],
                    confusions: Sequence[np.ndarray],
                    alpha: float = 1e-5, lambda_delay: float = 1e-5,
                    poll_budget: int = 2, n_rounds: int = 20,
                    eps: float = EPS, **_) -> TrialResult:
    """AWF as a value-driven scheduler: poll the highest-``V_j`` missing users.

    Under a constrained uplink that admits only ``poll_budget`` responses per
    round, AWF ranks missing users by their waiting value ``V_j(t)`` -- the
    product of arrival probability and risk reduction minus delay cost -- and
    polls the most valuable ones first.  This couples the network schedule to
    the decision objective (network<->decision co-design).
    """
    def _prio(j: int, t: float, v_map: Dict[int, float]) -> float:
        v = v_map.get(j, -np.inf)
        return -v if np.isfinite(v) else np.inf      # highest V_j first
    return _polling_core(P, arrivals_ms, deadline_ms, profiles, confusions,
                         _prio, poll_budget, n_rounds, alpha, lambda_delay, eps)


def poll_fastest_fusion(P: np.ndarray, arrivals_ms: np.ndarray,
                        deadline_ms: float,
                        profiles: Sequence[NetworkProfile],
                        confusions: Sequence[np.ndarray],
                        alpha: float = 1e-5, lambda_delay: float = 1e-5,
                        poll_budget: int = 2, n_rounds: int = 20,
                        eps: float = EPS, **_) -> TrialResult:
    """Baseline scheduler: poll the soonest-to-respond missing users first.

    Identical to :func:`awf_poll_fusion` except that the polling priority is
    the expected remaining delay (network-only, decision-blind), so any
    accuracy/latency gap to ``awf_poll`` is attributable solely to using the
    decision value ``V_j`` as the scheduling signal.
    """
    def _prio(j: int, t: float, v_map: Dict[int, float]) -> float:
        return expected_remaining_delay(profiles[j], t, deadline_ms)
    return _polling_core(P, arrivals_ms, deadline_ms, profiles, confusions,
                         _prio, poll_budget, n_rounds, alpha, lambda_delay, eps)


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

def get_method(name: str) -> Callable[..., TrialResult]:
    """Return the fusion function for a textual method name.

    Names that include a parameter use ``method:param`` syntax, e.g.
    ``"earliest_k:3"`` or ``"reliability_top_k:5"``. AWF ablations are
    accessed as ``"awf"`` (full), ``"awf_no_delay"``, ``"awf_no_reliability"``,
    ``"awf_no_arrival"``.
    """
    base = name.split(":")[0]
    parts = name.split(":")
    param = parts[1] if len(parts) > 1 else None

    if base == "mean":
        return mean_fusion
    if base == "weighted":
        return weighted_sum_fusion
    if base == "logpool":
        return logpool_fusion
    if base == "majority":
        return majority_voting
    if base == "borda":
        return borda_count_fusion

    if base == "fastest_only":
        return fastest_only
    if base == "earliest_k":
        k = int(param) if param else 3
        return lambda *a, **kw: earliest_k_fusion(*a, k=k, **kw)
    if base == "deadline_partial":
        return deadline_partial_fusion
    if base == "anytime_cbci":
        theta = float(param) if param else 0.80
        return lambda *a, **kw: anytime_cbci(*a, theta=theta, **kw)
    if base == "anytime_cbci_cw":
        theta = float(param) if param else 0.80
        return lambda *a, **kw: anytime_cbci_cw(*a, theta=theta, **kw)
    if base == "latency_only":
        k = int(param) if param else 3
        return lambda *a, **kw: latency_only_selection(*a, k=k, **kw)

    if base == "best_user":
        return best_user_only
    if base == "reliability_top_k":
        k = int(param) if param else 3
        return lambda *a, **kw: reliability_top_k_fusion(*a, k=k, **kw)
    if base == "reliability_weighted":
        return reliability_weighted_deadline_fusion
    if base == "reliability_latency":
        k = int(param) if param else None
        return lambda *a, **kw: reliability_latency_score_selection(*a, k=k, **kw)

    if base == "deadline_aware":
        tau = float(param) if param else 0.05
        return lambda *a, **kw: deadline_aware_waiting(*a, tau=tau, **kw)
    if base == "da_os":
        b = float(param) if param else None
        return lambda *a, **kw: da_os_stopping(*a, beta=b, **kw)
    if base == "generic_voi":
        return generic_voi_waiting
    if base == "neurobci_sc":
        return neurobci_sc_fusion

    if base == "oracle":
        return oracle_best_stopping

    if base == "awf":
        return awf_fusion
    if base == "awf_no_delay":
        return lambda *a, **kw: awf_fusion(*a, use_delay_cost=False, **kw)
    if base == "awf_no_reliability":
        return lambda *a, **kw: awf_fusion(*a, use_reliability=False, **kw)
    if base == "awf_no_arrival":
        return lambda *a, **kw: awf_fusion(*a, use_arrival_prob=False, **kw)
    if base == "awf_linear":
        return lambda *a, **kw: awf_fusion(*a, pred_pool="linear", **kw)
    if base == "awf_exact":
        return lambda *a, **kw: awf_fusion(*a, exact_arrival=True, **kw)

    if base == "awf_poll":
        m = int(param) if param else 2
        return lambda *a, **kw: awf_poll_fusion(*a, poll_budget=m, **kw)
    if base == "poll_fastest":
        m = int(param) if param else 2
        return lambda *a, **kw: poll_fastest_fusion(*a, poll_budget=m, **kw)

    raise ValueError(f"Unknown fusion method {name!r}")
