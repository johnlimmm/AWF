"""
bcinet-metrics.py
=================

Aggregate metrics for the cBCI experiment.

Each metric takes a list of :class:`TrialResult` objects (defined in
``bcinet-fusion.py``) plus the simulator's ground-truth and arrival data
and returns a single scalar (or a small dict).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Per-method aggregates
# ---------------------------------------------------------------------------

def accuracy(results, y_true: np.ndarray) -> float:
    """Fraction of trials whose final prediction matches the truth."""
    correct = sum(1 for r, y in zip(results, y_true) if r.pred == int(y))
    return correct / max(len(results), 1)


def mean_decision_latency(results) -> float:
    """Mean of the per-trial decision time (ms)."""
    if not results:
        return 0.0
    return float(np.mean([r.decision_time_ms for r in results]))


def used_response_ratio(results, n_users: int) -> float:
    """Average (#used responses / #all users) across trials."""
    if not results:
        return 0.0
    return float(np.mean([r.used_mask.sum() / max(n_users, 1) for r in results]))


def late_or_lost_response_ratio(results, arrivals_ms: np.ndarray,
                                deadline_ms: float) -> float:
    """Average ratio of late/lost responses (per trial).

    For each trial, count the number of users whose response was either
    lost or arrived strictly after the decision time used by the method,
    divided by the total number of users.
    """
    n_users, n_trials = arrivals_ms.shape
    if n_trials == 0:
        return 0.0
    ratios = []
    for t in range(n_trials):
        decision_time = results[t].decision_time_ms
        late = (~np.isfinite(arrivals_ms[:, t])) | (arrivals_ms[:, t] > decision_time)
        # We do not count "arrived but unused" - only late/lost (i.e.
        # responses that were never available by the decision time).
        late = late & (~results[t].used_mask)
        ratios.append(late.sum() / n_users)
    return float(np.mean(ratios))


def waiting_reduction(results, baseline_results) -> float:
    """Latency saving fraction relative to a baseline (e.g. deadline-partial).

    Returns ``1 - mean(latency_method) / mean(latency_baseline)``.
    Negative values mean the method is *slower* than the baseline.
    """
    base = mean_decision_latency(baseline_results)
    if base <= 0:
        return 0.0
    return 1.0 - mean_decision_latency(results) / base


def stopping_reason_counts(results) -> Dict[str, int]:
    """Count trials per stopping reason (only meaningful for AWF-like methods)."""
    counts: Dict[str, int] = {}
    for r in results:
        counts[r.stopping_reason] = counts.get(r.stopping_reason, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# AWF-specific metrics
# ---------------------------------------------------------------------------

def awf_specific_metrics(results) -> Dict[str, float]:
    """Aggregate the AWF-specific bookkeeping (mean waiting value, etc.)."""
    keys = ["mean_waiting_value", "mean_risk_reduction", "mean_waited_rho"]
    out: Dict[str, float] = {}
    for k in keys:
        vals = [r.extras.get(k, np.nan) for r in results if r.extras]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        out[k] = float(np.mean(vals)) if vals else 0.0
    return out


# ---------------------------------------------------------------------------
# Convenience: produce a single-row dict for a (method, seed) run
# ---------------------------------------------------------------------------

def summarise_run(method: str, seed: int, dataset: str, preset: str,
                  results, arrivals_ms: np.ndarray, deadline_ms: float,
                  y_true: np.ndarray, n_users: int,
                  baseline_results=None) -> Dict[str, float]:
    """Return one row's worth of aggregate metrics."""
    row: Dict[str, float] = {
        "dataset": dataset,
        "preset": preset,
        "method": method,
        "seed": seed,
        "accuracy": accuracy(results, y_true),
        "decision_latency": mean_decision_latency(results),
        "used_response_ratio": used_response_ratio(results, n_users),
        "late_lost_response_ratio": late_or_lost_response_ratio(
            results, arrivals_ms, deadline_ms),
    }
    if baseline_results is not None:
        row["waiting_reduction"] = waiting_reduction(results, baseline_results)
    else:
        row["waiting_reduction"] = 0.0

    # Most-frequent stopping reason for this run.
    counts = stopping_reason_counts(results)
    if counts:
        row["stopping_reason"] = max(counts.items(), key=lambda kv: kv[1])[0]
    else:
        row["stopping_reason"] = "none"

    # AWF specific (will be 0 for non-AWF runs).
    extras = awf_specific_metrics(results)
    row.update(extras)

    # Raw stopping-reason proportions (helps the AWF stopping-reason plot).
    total = sum(counts.values()) or 1
    for key in ("margin", "neg_value", "deadline", "fixed_set",
                "all_arrived", "oracle", "no_response", "late_lost"):
        row[f"reason_{key}"] = counts.get(key, 0) / total
    return row
