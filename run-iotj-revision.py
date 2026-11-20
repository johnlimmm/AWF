"""Reproducible experiment driver for the IoTJ major revision.

The revision results are isolated from legacy CSVs under
``results/iotj-revision-final`` by default.  Policy knobs are selected on
collaborative validation trials and then frozen for held-out test evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "results", "iotj-revision-final")
OUT = DEFAULT_OUT
OBJECTIVE_LAMBDA = 1e-3
DEADLINE_MS = 50.0
SOURCE_FILES = [
    "run-iotj-revision.py", "run-bcinet-experiment.py",
    "bcinet-data.py", "bcinet-fusion.py", "bcinet-network.py",
    "bcinet-tracebank.py", "bcinet-metrics.py",
    "plot-iotj-revision.py", "test_revision_invariants.py",
]

# Keep MNE's mutable configuration/lock file inside the workspace.  This also
# prevents concurrent or sandboxed revision runs from mutating the user's
# global MNE configuration.
MNE_CONFIG_HOME = os.path.join(HERE, "cache", "mne-home")
os.makedirs(os.path.join(MNE_CONFIG_HOME, ".mne"), exist_ok=True)
# MNE's Windows config resolver uses this test-supported override before
# USERPROFILE; MNE_CONFIG_DIR is not honored by current MNE releases.
os.environ.setdefault("_MNE_FAKE_HOME_DIR", MNE_CONFIG_HOME)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = _load("revision_runner", "run-bcinet-experiment.py")
TB = _load("revision_tracebank", "bcinet-tracebank.py")
metrics = R.bcinet_metrics


def _save(df: pd.DataFrame, filename: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, filename)
    df.to_csv(path, index=False)
    print("wrote", path, "rows=", len(df), flush=True)


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool | None:
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=HERE, text=True).strip())
    except Exception:
        return None


def _source_hashes() -> Dict[str, str]:
    """Hash the experiment implementation used by this campaign."""
    out = {}
    for name in SOURCE_FILES:
        path = os.path.join(HERE, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                out[name] = hashlib.sha256(f.read()).hexdigest()
    return out


def _experiment_sources_dirty() -> bool | None:
    """Report changes only in files that can affect revision results."""
    try:
        unstaged = subprocess.call(
            ["git", "diff", "--quiet", "--", *SOURCE_FILES], cwd=HERE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        staged = subprocess.call(
            ["git", "diff", "--cached", "--quiet", "--", *SOURCE_FILES],
            cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "--",
             *SOURCE_FILES], cwd=HERE, text=True,
            stderr=subprocess.DEVNULL).strip()
        return bool(unstaged or staged or untracked)
    except Exception:
        return None


def write_manifest(args: argparse.Namespace,
                   completed: Sequence[str]) -> None:
    os.makedirs(OUT, exist_ok=True)
    requested = list(args.only)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
        "experiment_sources_dirty": _experiment_sources_dirty(),
        "status": "complete" if list(completed) == requested else "running",
        "requested_experiments": requested,
        "completed_experiments": list(completed),
        "tracebank_schema_version": TB.TRACEBANK_SCHEMA_VERSION,
        "cellular_filter": sorted(TB.CELL_LTE_MODES),
        "deadline_ms": DEADLINE_MS,
        "selection_objective": "1-accuracy + 1e-3*mean_latency_ms",
        "datasets": args.datasets,
        "seeds": args.seeds,
        "n_trials": args.n_trials,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "command": sys.argv,
        "source_sha256": _source_hashes(),
    }
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def experiment_trace() -> None:
    bank = TB.load_tracebank()
    rows = []
    for tier in TB.TIER_ORDER:
        b = bank[tier]
        delay = np.asarray(b.all_delays(), dtype=float)
        row = {
            "tier": tier, "n_pooled": len(delay),
            "n_contiguous_sequences": len(b.sequences),
            "loss_rate": b.overall_loss(),
            "p50_ms": np.percentile(delay, 50),
            "p95_ms": np.percentile(delay, 95),
            "p99_ms": np.percentile(delay, 99),
            "arrival_by_50ms": ((1.0 - b.overall_loss())
                                * float(np.mean(delay <= DEADLINE_MS))),
        }
        rows.append(row)
    _save(pd.DataFrame(rows), "trace-summary.csv")


def _probability_metrics(p: np.ndarray, y: np.ndarray,
                         bins: int = 10) -> Dict[str, float]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    correct = (pred == y).astype(float)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    onehot = np.eye(p.shape[1])[y]
    return {
        "accuracy": float(correct.mean()),
        "nll": -float(np.mean(np.log(np.clip(p[np.arange(len(y)), y],
                                            1e-12, 1.0)))),
        "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "ece10": float(ece),
    }


def experiment_calibration(datasets: Sequence[str]) -> None:
    rows = []
    for dataset in datasets:
        users = R._ensure_users(dataset)
        for u in users:
            raw = _probability_metrics(u.test_posteriors_raw, u.test_labels)
            cal = _probability_metrics(u.test_posteriors, u.test_labels)
            for variant, vals in (("raw", raw), ("temperature", cal)):
                rows.append({
                    "dataset": dataset, "user": u.user_id,
                    "variant": variant, "temperature": u.temperature,
                    **vals,
                })
    _save(pd.DataFrame(rows), "calibration-per-user.csv")


def _safe_wilcoxon(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.allclose(a, b, equal_nan=True):
        return 1.0
    try:
        p = float(wilcoxon(a, b).pvalue)
        return p if np.isfinite(p) else 1.0
    except ValueError:
        return 1.0


def _holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Holm family-wise error correction with monotonic adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def experiment_calibration_policy(datasets: Sequence[str], seeds: int,
                                  n_trials: int | None) -> None:
    """Paired raw-vs-calibrated AWF stopping-behavior comparison.

    The validation-selected AWF delay weight is held fixed.  Each pair uses
    the same collaborative-trial sampling, user-to-tier assignment, and
    realized delay/loss samples; only the posterior calibration is changed.
    """
    selection_path = os.path.join(OUT, "tuned-parameters.csv")
    if not os.path.exists(selection_path):
        raise RuntimeError("run the tuning experiment before calibration_policy")
    selected = pd.read_csv(selection_path)
    seed_rows = []
    trial_rows = []
    reason_keys = ("margin", "neg_value", "deadline", "all_arrived")
    for dataset in datasets:
        users = R._ensure_users(dataset)
        match = selected[(selected.dataset == dataset)
                         & (selected.method == "AWF")]
        if len(match) != 1:
            raise RuntimeError("missing unique tuned AWF lambda for " + dataset)
        tuned_lambda = float(match.iloc[0].control)
        for seed in range(seeds):
            paired = {}
            ref_arrivals = None
            ref_y = None
            for variant, calibrated in (("raw", False),
                                        ("temperature", True)):
                result, arrivals, y, n_users = R.run_one(
                    "awf", seed, dataset, "accesstier", users,
                    deadline_ms=DEADLINE_MS, alpha=1e-3,
                    lambda_delay=tuned_lambda, n_trials=n_trials,
                    calibrated=calibrated, split="test")
                if ref_arrivals is None:
                    ref_arrivals = arrivals
                    ref_y = y
                else:
                    if not np.array_equal(ref_arrivals, arrivals):
                        raise AssertionError(
                            "raw/calibrated runs did not share arrivals")
                    if not np.array_equal(ref_y, y):
                        raise AssertionError(
                            "raw/calibrated runs did not share labels")
                counts = metrics.stopping_reason_counts(result)
                total = len(result) or 1
                row = {
                    "dataset": dataset, "seed": seed, "variant": variant,
                    "tuned_lambda": tuned_lambda,
                    "accuracy": metrics.accuracy(result, y),
                    "latency_mean_ms": metrics.mean_decision_latency(result),
                    "used_response_ratio": metrics.used_response_ratio(
                        result, n_users),
                }
                for key in reason_keys:
                    row[f"reason_{key}"] = counts.get(key, 0) / total
                seed_rows.append(row)
                paired[variant] = result

            raw = paired["raw"]
            cal = paired["temperature"]
            for trial, (r_raw, r_cal, truth) in enumerate(
                    zip(raw, cal, ref_y)):
                trial_rows.append({
                    "dataset": dataset, "seed": seed, "trial": trial,
                    "truth": int(truth),
                    "raw_pred": int(r_raw.pred),
                    "temperature_pred": int(r_cal.pred),
                    "raw_correct": int(r_raw.pred == truth),
                    "temperature_correct": int(r_cal.pred == truth),
                    "raw_latency_ms": r_raw.decision_time_ms,
                    "temperature_latency_ms": r_cal.decision_time_ms,
                    "raw_reason": r_raw.stopping_reason,
                    "temperature_reason": r_cal.stopping_reason,
                    "prediction_agreement": int(r_raw.pred == r_cal.pred),
                    "stop_time_agreement": int(np.isclose(
                        r_raw.decision_time_ms, r_cal.decision_time_ms)),
                    "stop_reason_agreement": int(
                        r_raw.stopping_reason == r_cal.stopping_reason),
                })

    seed_df = pd.DataFrame(seed_rows)
    trial_df = pd.DataFrame(trial_rows)
    _save(seed_df, "calibration-awf-seeds.csv")
    _save(trial_df, "calibration-awf-trials.csv")

    summary_rows = []
    for dataset in datasets:
        d = seed_df[seed_df.dataset == dataset]
        raw = d[d.variant == "raw"].sort_values("seed")
        cal = d[d.variant == "temperature"].sort_values("seed")
        t = trial_df[trial_df.dataset == dataset]
        summary_rows.append({
            "dataset": dataset,
            "raw_accuracy": raw.accuracy.mean(),
            "temperature_accuracy": cal.accuracy.mean(),
            "accuracy_delta_temperature_minus_raw": (
                cal.accuracy.to_numpy() - raw.accuracy.to_numpy()).mean(),
            "accuracy_wilcoxon_p": _safe_wilcoxon(
                cal.accuracy, raw.accuracy),
            "raw_latency_ms": raw.latency_mean_ms.mean(),
            "temperature_latency_ms": cal.latency_mean_ms.mean(),
            "latency_delta_temperature_minus_raw_ms": (
                cal.latency_mean_ms.to_numpy()
                - raw.latency_mean_ms.to_numpy()).mean(),
            "latency_wilcoxon_p": _safe_wilcoxon(
                cal.latency_mean_ms, raw.latency_mean_ms),
            "raw_margin_rate": raw.reason_margin.mean(),
            "temperature_margin_rate": cal.reason_margin.mean(),
            "raw_neg_value_rate": raw.reason_neg_value.mean(),
            "temperature_neg_value_rate": cal.reason_neg_value.mean(),
            "prediction_agreement": t.prediction_agreement.mean(),
            "stop_time_agreement": t.stop_time_agreement.mean(),
            "stop_reason_agreement": t.stop_reason_agreement.mean(),
        })
    summary = pd.DataFrame(summary_rows)
    summary["accuracy_wilcoxon_p_holm"] = _holm_adjust(
        summary.accuracy_wilcoxon_p)
    summary["latency_wilcoxon_p_holm"] = _holm_adjust(
        summary.latency_wilcoxon_p)
    _save(summary, "calibration-awf-summary.csv")


def _candidate_grid(n_users: int) -> Dict[str, List[Tuple[str, float]]]:
    """Display name -> list of (runner method, control value)."""
    return {
        "First-k": [(f"earliest_k:{k}", float(k))
                    for k in range(1, n_users + 1)],
        "Confidence-Stop": [(f"anytime_cbci:{v:g}", v)
                            for v in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95,
                                      0.99, 0.999, 1.0)],
        "Anytime-cBCI": [(f"anytime_cbci_cw:{v:g}", v)
                         for v in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95,
                                   0.99, 0.999, 1.0)],
        "Arrival-Only": [(f"deadline_aware:{v:g}", v)
                         for v in (0.5, 0.3, 0.2, 0.1, 0.05, 0.02,
                                   0.01, 0.0)],
        "Throughput-Stop": [(f"da_os:{v:g}", v)
                            for v in (0.0, 1e-3, 1e-2, 3e-2, 1e-1,
                                      3e-1, 1.0, 3.0, 10.0, 30.0,
                                      100.0, 300.0, 1e3)],
        # These two methods consume the supplied lambda_delay rather than a
        # colon parameter; their runner names remain constant.
        "Entropy-Gain": [("generic_voi", v) for v in
                         (0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3,
                          3e-3, 1e-2, 3e-2, 1e-1)],
        "AWF": [("awf", v) for v in
                (0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3,
                 3e-3, 1e-2, 3e-2, 1e-1)],
    }


def _run_candidate(method_name: str, control: float, display: str,
                   seed: int, dataset: str, users, split: str,
                   n_trials: int | None):
    lam = control if display in ("Entropy-Gain", "AWF") else OBJECTIVE_LAMBDA
    return R.run_one(
        method_name, seed, dataset, "accesstier", users,
        deadline_ms=DEADLINE_MS, alpha=1e-3, lambda_delay=lam,
        n_trials=n_trials, calibrated=True, split=split)


def experiment_tuned(datasets: Sequence[str], seeds: int,
                     n_trials: int | None) -> None:
    selections = []
    test_rows = []
    trial_rows = []
    for dataset in datasets:
        users = R._ensure_users(dataset)
        selected: Dict[str, Tuple[str, float]] = {}
        for display, candidates in _candidate_grid(len(users)).items():
            candidate_rows = []
            for method_name, control in candidates:
                accs, lats = [], []
                for s in range(seeds):
                    # Disjoint network seeds from final test deployment seeds;
                    # EEG samples come only from the validation split here.
                    result, _, y, _ = _run_candidate(
                        method_name, control, display, 10_000 + s,
                        dataset, users, "validation", n_trials)
                    accs.append(metrics.accuracy(result, y))
                    lats.extend(r.decision_time_ms for r in result)
                acc = float(np.mean(accs))
                latency = float(np.mean(lats))
                candidate_rows.append((1.0 - acc + OBJECTIVE_LAMBDA * latency,
                                       method_name, control, acc, latency))
            best = min(candidate_rows, key=lambda x: (x[0], x[4]))
            selected[display] = (best[1], best[2])
            selections.append({
                "dataset": dataset, "method": display,
                "runner_method": best[1], "control": best[2],
                "validation_objective": best[0],
                "validation_accuracy": best[3],
                "validation_latency_ms": best[4],
            })

        # Parameter-free Full-Wait is retained as a fixed reference.
        selected["Full-Wait"] = ("logpool", np.nan)
        for s in range(seeds):
            for display, (method_name, control) in selected.items():
                actual_control = (OBJECTIVE_LAMBDA if np.isnan(control)
                                  else control)
                result, _, y, _ = _run_candidate(
                    method_name, actual_control, display, s, dataset, users,
                    "test", n_trials)
                test_rows.append({
                    "dataset": dataset, "seed": s, "method": display,
                    "accuracy": metrics.accuracy(result, y),
                    "latency_mean_ms": float(np.mean(
                        [r.decision_time_ms for r in result])),
                    "latency_p95_ms": float(np.percentile(
                        [r.decision_time_ms for r in result], 95)),
                })
                for t, (r, truth) in enumerate(zip(result, y)):
                    trial_rows.append({
                        "dataset": dataset, "seed": s, "trial": t,
                        "method": display, "correct": int(r.pred == truth),
                        "latency_ms": r.decision_time_ms,
                    })
    _save(pd.DataFrame(selections), "tuned-parameters.csv")
    _save(pd.DataFrame(test_rows), "tuned-test-summary.csv")
    trials = pd.DataFrame(trial_rows)
    _save(trials, "tuned-test-trials.csv")
    _save(_paired_statistics(trials), "tuned-paired-tests.csv")


def _paired_statistics(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in sorted(trials.dataset.unique()):
        d = trials[trials.dataset == dataset]
        awf = d[d.method == "AWF"].sort_values(["seed", "trial"])
        for method in sorted(set(d.method) - {"AWF"}):
            other = d[d.method == method].sort_values(["seed", "trial"])
            if len(awf) != len(other):
                continue
            a = awf.correct.to_numpy()
            b = other.correct.to_numpy()
            awf_only = int(np.sum((a == 1) & (b == 0)))
            other_only = int(np.sum((a == 0) & (b == 1)))
            discordant = awf_only + other_only
            p_mcnemar = (binomtest(awf_only, discordant, 0.5).pvalue
                          if discordant else 1.0)
            lat_a = awf.latency_ms.to_numpy()
            lat_b = other.latency_ms.to_numpy()
            p_latency = _safe_wilcoxon(lat_a, lat_b)
            seed_a = (awf.groupby("seed")
                      .agg(accuracy=("correct", "mean"),
                           latency=("latency_ms", "mean")))
            seed_b = (other.groupby("seed")
                      .agg(accuracy=("correct", "mean"),
                           latency=("latency_ms", "mean")))
            p_seed_acc = _safe_wilcoxon(seed_a.accuracy, seed_b.accuracy)
            p_seed_lat = _safe_wilcoxon(seed_a.latency, seed_b.latency)
            rows.append({
                "dataset": dataset, "comparison": f"AWF vs {method}",
                "awf_only_correct": awf_only,
                "other_only_correct": other_only,
                "mcnemar_exact_p": p_mcnemar,
                "mean_paired_latency_delta_ms": float(np.mean(lat_a - lat_b)),
                "wilcoxon_latency_p": p_latency,
                "mean_seed_accuracy_delta": float(np.mean(
                    seed_a.accuracy.to_numpy() - seed_b.accuracy.to_numpy())),
                "wilcoxon_seed_accuracy_p": p_seed_acc,
                "wilcoxon_seed_latency_p": p_seed_lat,
            })
    out = pd.DataFrame(rows)
    for dataset in out.dataset.unique():
        mask = out.dataset == dataset
        out.loc[mask, "mcnemar_exact_p_holm"] = _holm_adjust(
            out.loc[mask, "mcnemar_exact_p"])
        out.loc[mask, "wilcoxon_seed_accuracy_p_holm"] = _holm_adjust(
            out.loc[mask, "wilcoxon_seed_accuracy_p"])
        out.loc[mask, "wilcoxon_seed_latency_p_holm"] = _holm_adjust(
            out.loc[mask, "wilcoxon_seed_latency_p"])
    return out


def experiment_ablation(datasets: Sequence[str], seeds: int,
                        n_trials: int | None) -> None:
    methods = ["awf", "awf_no_delay", "awf_no_reliability",
               "awf_accuracy_only", "awf_no_arrival", "awf_no_margin"]
    selection_path = os.path.join(OUT, "tuned-parameters.csv")
    if not os.path.exists(selection_path):
        raise RuntimeError("run the tuning experiment before ablation")
    selected = pd.read_csv(selection_path)
    frames = []
    for dataset in datasets:
        match = selected[(selected.dataset == dataset)
                         & (selected.method == "AWF")]
        if len(match) != 1:
            raise RuntimeError("missing unique tuned AWF lambda for " + dataset)
        tuned_lambda = float(match.iloc[0].control)
        frame = R.run_main_table(
            datasets=[dataset], presets=["accesstier"], methods=methods,
            seeds=seeds, deadline_ms=DEADLINE_MS, alpha=1e-3,
            lambda_delay=tuned_lambda, n_trials=n_trials,
            max_subjects=None)
        frame["tuned_lambda"] = tuned_lambda
        frames.append(frame)
    _save(pd.concat(frames, ignore_index=True), "awf-ablation.csv")


def experiment_dp(seeds: int, n_trials: int | None) -> None:
    dataset = "Zhou2016"
    users = R._ensure_users(dataset)
    rows = []
    for s in range(seeds):
        # The primary 5-ms DP reference uses every seed.  Finer/coarser grid
        # sensitivity is expensive and is repeated on the first three seeds.
        methods = ["awf", "dp_optimal:5"]
        if s < min(seeds, 3):
            methods.extend(["dp_optimal:2.5", "dp_optimal:10"])
        for method in methods:
            result, _, y, _ = R.run_one(
                method, s, dataset, "accesstier", users,
                deadline_ms=DEADLINE_MS, alpha=1e-3,
                lambda_delay=OBJECTIVE_LAMBDA, n_trials=n_trials,
                calibrated=True, split="test")
            rows.append({
                "dataset": dataset, "seed": s, "method": method,
                "accuracy": metrics.accuracy(result, y),
                "latency_mean_ms": float(np.mean(
                    [r.decision_time_ms for r in result])),
                "objective": (1.0 - metrics.accuracy(result, y)
                              + OBJECTIVE_LAMBDA * float(np.mean(
                                  [r.decision_time_ms for r in result]))),
            })
    _save(pd.DataFrame(rows), "dp-gap-zhou2016.csv")


def _heldout_trace_split():
    """Chronologically split each tier's preserved sequences and loss rates."""
    bank = TB.load_tracebank()
    split = {}
    for tier in TB.TIER_ORDER:
        pairs = [(np.asarray(seq, dtype=float), float(loss))
                 for seq, loss in zip(bank[tier].sequences,
                                      bank[tier].sequence_losses)
                 if len(seq) >= 4 and np.isfinite(seq).all()]
        if len(pairs) < 2:
            raise RuntimeError("held-out replay needs >=2 sequences for " + tier)
        cut = len(pairs) // 2
        train, test = pairs[:cut], pairs[cut:]
        split[tier] = {
            "train_pool": np.sort(np.concatenate([x[0] for x in train])),
            "test_pool": np.sort(np.concatenate([x[0] for x in test])),
            "test_sequences": [x[0] for x in test],
            "test_sequence_losses": [x[1] for x in test],
            "train_loss": float(np.mean([x[1] for x in train])),
            "test_loss": float(np.mean([x[1] for x in test])),
        }
    return split


def _trace_profile(tier: str, samples: np.ndarray, loss: float):
    samples = np.sort(np.asarray(samples, dtype=float))
    p10, p90 = np.percentile(samples, [10, 90])
    if p90 <= p10:
        p90 = p10 + 1.0
    if len(samples) > TB.MAX_POOL:
        # Deterministic quantile-preserving cap, matching the main trace-bank
        # empirical-CDF budget while leaving ordered replay sequences intact.
        keep = np.linspace(0, len(samples) - 1, TB.MAX_POOL, dtype=int)
        samples = samples[keep]
    return R.bcinet_network.NetworkProfile(
        tier=tier, delay_range=(float(p10), float(p90)),
        jitter_range=(0.0, 0.0), loss_prob=float(loss),
        est_samples=samples,
        est_loss=float(loss))


def _heldout_replay(tiers: Sequence[str], n_trials: int,
                    rng: np.random.RandomState, split,
                    shared_link: bool) -> np.ndarray:
    """Replay test-half sequences, optionally sharing one path per tier."""
    arrivals = np.empty((len(tiers), n_trials), dtype=float)
    shared = {}
    for user, tier in enumerate(tiers):
        source = split[tier]
        if shared_link and tier in shared:
            arrivals[user] = shared[tier]
            continue
        idx = int(rng.randint(len(source["test_sequences"])))
        seq = source["test_sequences"][idx]
        start = int(rng.randint(len(seq)))
        replay = seq[(start + np.arange(n_trials)) % len(seq)].astype(float)
        loss_mask = rng.random(n_trials) < source["test_sequence_losses"][idx]
        replay[loss_mask] = np.inf
        arrivals[user] = replay
        if shared_link:
            # Strong shared-bottleneck stress: users on the same access tier
            # experience the same delay trajectory and loss events.
            shared[tier] = replay.copy()
    return arrivals


def _evaluate_custom(method: str, P_cube: np.ndarray, y: np.ndarray,
                     arrivals: np.ndarray, profiles, users,
                     lambda_delay: float) -> Tuple[float, float]:
    fn = R.bcinet_fusion.get_method(method)
    val_acc = np.asarray([u.val_accuracy for u in users])
    confusions = [u.confusion for u in users]
    correct, latency = [], []
    for t in range(P_cube.shape[1]):
        result = fn(
            P=P_cube[:, t, :], arrivals_ms=arrivals[:, t],
            deadline_ms=DEADLINE_MS, profiles=profiles,
            confusions=confusions, val_acc=val_acc, alpha=1e-3,
            lambda_delay=lambda_delay, y_true=int(y[t]))
        correct.append(int(result.pred == int(y[t])))
        latency.append(result.decision_time_ms)
    return float(np.mean(correct)), float(np.mean(latency))


def experiment_heldout(datasets: Sequence[str], seeds: int,
                       n_trials: int | None) -> None:
    """Out-of-sample delay/loss belief plus independent/shared ordered replay."""
    split = _heldout_trace_split()
    selected = pd.read_csv(os.path.join(OUT, "tuned-parameters.csv"))
    rows = []
    for dataset in datasets:
        print("[heldout]", dataset, flush=True)
        users = R._ensure_users(dataset)
        lam = float(selected[(selected.dataset == dataset)
                             & (selected.method == "AWF")].iloc[0].control)
        for seed in range(seeds):
            rng = np.random.RandomState(seed)
            P_cube, y = R.bcinet_data.align_test_trials(
                users, rng, n_trials=n_trials, calibrated=True, split="test")
            tiers = [TB.TIER_ORDER[int(rng.randint(len(TB.TIER_ORDER)))]
                     for _ in users]
            nominal = [_trace_profile(t, split[t]["train_pool"],
                                      split[t]["train_loss"]) for t in tiers]
            oracle = [_trace_profile(t, split[t]["test_pool"],
                                     split[t]["test_loss"]) for t in tiers]
            for replay_mode, shared in (("independent", False),
                                        ("shared_link", True)):
                # Reset the replay RNG so nominal/oracle see identical arrivals.
                replay_rng = np.random.RandomState(50_000 + seed)
                arrivals = _heldout_replay(tiers, len(y), replay_rng,
                                           split, shared)
                for policy, profiles in (("AWF-heldout", nominal),
                                         ("AWF-oracle", oracle)):
                    acc, latency = _evaluate_custom(
                        "awf", P_cube, y, arrivals, profiles, users, lam)
                    rows.append({
                        "dataset": dataset, "seed": seed,
                        "replay_mode": replay_mode, "policy": policy,
                        "accuracy": acc, "latency_mean_ms": latency,
                        "objective": 1.0 - acc + OBJECTIVE_LAMBDA * latency,
                    })
                acc, latency = _evaluate_custom(
                    "logpool", P_cube, y, arrivals, nominal, users, lam)
                rows.append({
                    "dataset": dataset, "seed": seed,
                    "replay_mode": replay_mode, "policy": "Full-Wait",
                    "accuracy": acc, "latency_mean_ms": latency,
                    "objective": 1.0 - acc + OBJECTIVE_LAMBDA * latency,
                })
    _save(pd.DataFrame(rows), "heldout-replay.csv")
    trace_rows = []
    for tier, values in split.items():
        trace_rows.append({
            "tier": tier,
            "train_n": len(values["train_pool"]),
            "test_n": len(values["test_pool"]),
            "train_loss": values["train_loss"],
            "test_loss": values["test_loss"],
            "train_p50_ms": np.median(values["train_pool"]),
            "test_p50_ms": np.median(values["test_pool"]),
        })
    _save(pd.DataFrame(trace_rows), "heldout-trace-split.csv")


def experiment_homogeneous(datasets: Sequence[str], seeds: int,
                           n_trials: int | None) -> None:
    """All committee members use the same access technology."""
    split = _heldout_trace_split()
    selected = pd.read_csv(os.path.join(OUT, "tuned-parameters.csv"))
    rows = []
    for dataset in datasets:
        print("[homogeneous]", dataset, flush=True)
        users = R._ensure_users(dataset)
        p = selected[selected.dataset == dataset]
        lam = float(p[p.method == "AWF"].iloc[0].control)
        confidence_method = str(p[p.method == "Confidence-Stop"]
                                .iloc[0].runner_method)
        for seed in range(seeds):
            rng = np.random.RandomState(seed)
            P_cube, y = R.bcinet_data.align_test_trials(
                users, rng, n_trials=n_trials, calibrated=True, split="test")
            for tier in TB.TIER_ORDER:
                tiers = [tier] * len(users)
                profiles = [_trace_profile(tier, split[tier]["train_pool"],
                                           split[tier]["train_loss"])
                            for _ in users]
                replay_rng = np.random.RandomState(70_000 + seed)
                arrivals = _heldout_replay(tiers, len(y), replay_rng,
                                           split, shared_link=False)
                for policy, method in (("AWF", "awf"),
                                       ("Confidence-Stop", confidence_method),
                                       ("Full-Wait", "logpool")):
                    acc, latency = _evaluate_custom(
                        method, P_cube, y, arrivals, profiles, users, lam)
                    rows.append({
                        "dataset": dataset, "seed": seed, "tier": tier,
                        "policy": policy, "accuracy": acc,
                        "latency_mean_ms": latency,
                    })
    _save(pd.DataFrame(rows), "homogeneous-access.csv")


def experiment_runtime() -> None:
    """Measure worst-case AWF policy time as committee size grows."""
    rows = []
    n_classes = 7
    for n_users in (4, 8, 10, 16, 32, 64):
        P = np.full((n_users, n_classes), 1.0 / n_classes)
        arrivals = np.linspace(1.0, 49.0, n_users)
        profile = _trace_profile(
            "runtime", np.linspace(1.0, 50.0, TB.MAX_POOL), 0.01)
        profiles = [profile] * n_users
        conf = np.full((n_classes, n_classes), 0.3 / (n_classes - 1))
        np.fill_diagonal(conf, 0.7)
        confusions = [conf] * n_users
        repeats = 100 if n_users <= 16 else 30
        samples = []
        call = dict(
            P=P, arrivals_ms=arrivals, deadline_ms=DEADLINE_MS,
            profiles=profiles, confusions=confusions,
            val_acc=np.full(n_users, 0.7), alpha=1e-3,
            lambda_delay=0.0, use_margin=False)
        for _ in range(3):
            R.bcinet_fusion.awf_fusion(**call)
        for _ in range(repeats):
            start = time.perf_counter_ns()
            R.bcinet_fusion.awf_fusion(**call)
            samples.append((time.perf_counter_ns() - start) / 1e3)
        belief_bytes = n_users * (
            profile.est_samples.nbytes + profile.est_prefix_sum.nbytes)
        rows.append({
            "n_users": n_users, "n_classes": n_classes,
            "repeats": repeats, "median_runtime_us": np.median(samples),
            "p95_runtime_us": np.percentile(samples, 95),
            "model_state_bytes": (P.nbytes + arrivals.nbytes
                                  + n_users * conf.nbytes + belief_bytes),
        })
    _save(pd.DataFrame(rows), "runtime-scaling.csv")


def _selected_policy_specs(dataset: str, selected: pd.DataFrame):
    rows = selected[selected.dataset == dataset]
    specs = []
    for display in ("First-k", "Confidence-Stop", "Anytime-cBCI",
                    "Arrival-Only", "Throughput-Stop", "Entropy-Gain",
                    "AWF"):
        row = rows[rows.method == display].iloc[0]
        specs.append((display, str(row.runner_method), float(row.control)))
    specs.append(("Full-Wait", "logpool", OBJECTIVE_LAMBDA))
    return specs


def experiment_error_correlation(datasets: Sequence[str], seeds: int,
                                 n_trials: int | None) -> None:
    """Preserve marginal user accuracy while correlating committee errors."""
    selected = pd.read_csv(os.path.join(OUT, "tuned-parameters.csv"))
    rows = []
    for dataset in datasets:
        print("[error-correlation]", dataset, flush=True)
        users = R._ensure_users(dataset)
        specs = {name: (method, control) for name, method, control
                 in _selected_policy_specs(dataset, selected)
                 if name in ("AWF", "Confidence-Stop", "Full-Wait")}
        for rho in (0.0, 0.2, 0.5):
            for seed in range(seeds):
                for display, (method, control) in specs.items():
                    lam = control if display == "AWF" else OBJECTIVE_LAMBDA
                    result, _, y, _ = R.run_one(
                        method, seed, dataset, "accesstier", users,
                        deadline_ms=DEADLINE_MS, alpha=1e-3,
                        lambda_delay=lam, n_trials=n_trials,
                        calibrated=True, split="test", rho_err=rho)
                    rows.append({
                        "dataset": dataset, "seed": seed,
                        "rho_error": rho, "policy": display,
                        "accuracy": metrics.accuracy(result, y),
                        "latency_mean_ms": float(np.mean(
                            [r.decision_time_ms for r in result])),
                    })
    _save(pd.DataFrame(rows), "correlated-errors.csv")


def experiment_deadlines(datasets: Sequence[str], seeds: int,
                         n_trials: int | None) -> None:
    """Fixed validation-selected policies across practical fusion budgets."""
    selected = pd.read_csv(os.path.join(OUT, "tuned-parameters.csv"))
    rows = []
    for dataset in datasets:
        print("[deadlines]", dataset, flush=True)
        users = R._ensure_users(dataset)
        specs = _selected_policy_specs(dataset, selected)
        for deadline in (20.0, 50.0, 100.0, 200.0):
            for seed in range(seeds):
                for display, method, control in specs:
                    lam = (control if display in ("Entropy-Gain", "AWF")
                           else OBJECTIVE_LAMBDA)
                    result, _, y, _ = R.run_one(
                        method, seed, dataset, "accesstier", users,
                        deadline_ms=deadline, alpha=1e-3,
                        lambda_delay=lam, n_trials=n_trials,
                        calibrated=True, split="test")
                    rows.append({
                        "dataset": dataset, "seed": seed,
                        "deadline_ms": deadline, "policy": display,
                        "accuracy": metrics.accuracy(result, y),
                        "latency_mean_ms": float(np.mean(
                            [r.decision_time_ms for r in result])),
                    })
    _save(pd.DataFrame(rows), "deadline-all-policies.csv")


EXPERIMENTS = {
    "trace": lambda a: experiment_trace(),
    "calibration": lambda a: experiment_calibration(a.datasets),
    "tuning": lambda a: experiment_tuned(a.datasets, a.seeds, a.n_trials),
    "calibration_policy": lambda a: experiment_calibration_policy(
        a.datasets, a.seeds, a.n_trials),
    "ablation": lambda a: experiment_ablation(a.datasets, a.seeds, a.n_trials),
    "dp": lambda a: experiment_dp(a.seeds, a.n_trials),
    "heldout": lambda a: experiment_heldout(a.datasets, a.seeds, a.n_trials),
    "homogeneous": lambda a: experiment_homogeneous(
        a.datasets, a.seeds, a.n_trials),
    "runtime": lambda a: experiment_runtime(),
    "correlation": lambda a: experiment_error_correlation(
        a.datasets, a.seeds, a.n_trials),
    "deadlines": lambda a: experiment_deadlines(
        a.datasets, a.seeds, a.n_trials),
}


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=EXPERIMENTS,
                        default=list(EXPERIMENTS))
    parser.add_argument("--datasets", nargs="+", choices=R.DATASETS,
                        default=list(R.DATASETS))
    parser.add_argument("--seeds", type=int, default=15)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument(
        "--out-dir", default=os.path.relpath(DEFAULT_OUT, HERE),
        help="result directory (relative paths are resolved from the repo root)")
    args = parser.parse_args()
    OUT = (args.out_dir if os.path.isabs(args.out_dir)
           else os.path.join(HERE, args.out_dir))
    OUT = os.path.abspath(OUT)
    completed = []
    write_manifest(args, completed)
    for name in args.only:
        print("[revision]", name, flush=True)
        EXPERIMENTS[name](args)
        completed.append(name)
        write_manifest(args, completed)


if __name__ == "__main__":
    main()
