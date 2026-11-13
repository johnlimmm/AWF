"""
run-accesstier.py
=================

Full access-tier trace-driven experiment suite for the collaborative-BCI
study.  Every experiment is grounded in the real per-access-tier trace bank
(Wi-Fi / 5G / 4G-LTE / LEO) built by ``bcinet-tracebank.py`` and consumed
through the ``accesstier`` network presets in ``bcinet-network.py``.

Experiments (each checkpoints its own CSV under ``results/accesstier/``):

  E1 main        : datasets x accesstier x methods, deadlines D in {50, 100}.
  E2 frontier    : AWF accuracy-latency frontier (lambda sweep) vs baselines.
  E3 rho-ablate  : AWF arrival-belief ablation (linear / tier / state / oracle).
  E4 ntn         : LEO fraction robustness (leo00 / leo10 / accesstier / leo30).
  E5 deadline    : decision-deadline sensitivity sweep.
  E6 mobility    : access-switching dynamics (fixed / iid / markov / replay).

Run all:   C:\\python\\python.exe run-accesstier.py
Subset:    C:\\python\\python.exe run-accesstier.py --only main frontier
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load("runexp", "run-bcinet-experiment.py")
net = R.bcinet_network
data = R.bcinet_data
fusion = R.bcinet_fusion
metrics = R.bcinet_metrics

OUT = os.path.join(HERE, "results", "accesstier")
os.makedirs(OUT, exist_ok=True)

ALL_DATASETS = list(R.DATASETS)
SEEDS = 15

# Method groups -------------------------------------------------------------
MAIN_METHODS = [
    "mean", "weighted", "logpool", "majority", "borda",
    "fastest_only", "earliest_k:3", "deadline_partial", "latency_only:3",
    "deadline_aware", "generic_voi", "neurobci_sc",
    "anytime_cbci", "anytime_cbci_cw", "da_os",
    "best_user", "reliability_top_k:3", "reliability_weighted",
    "reliability_latency", "oracle", "awf",
]
KEY_METHODS = ["weighted", "logpool", "fastest_only", "earliest_k:3",
               "anytime_cbci", "anytime_cbci_cw", "da_os", "deadline_partial",
               "deadline_aware", "generic_voi", "neurobci_sc", "awf", "oracle"]
AWF_LAMBDAS = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3,
               1e-2, 3e-2, 1e-1]
ANYTIME_THETAS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0]
DEADLINE_TAUS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0]
DA_OS_BETAS = [0.0, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0,
               10.0, 30.0, 100.0, 300.0, 1e3]

# Unified operating point.  Both the margin-scale alpha and the delay-cost
# lambda default to 1e-3.  The sensitivity sweeps below vary one knob at a
# time over the same decade grid while holding the other at this default.
DEFAULT_ALPHA = 1e-3
DEFAULT_LAMBDA = 1e-3


def _log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def _save(df: pd.DataFrame, name: str) -> None:
    path = os.path.join(OUT, name)
    df.to_csv(path, index=False)
    _log("wrote %s (%d rows)" % (path, len(df)))


def _pct_latencies(results) -> Dict[str, float]:
    lat = np.array([d.decision_time_ms for d in results], dtype=float)
    return {"latency_mean": float(lat.mean()),
            "latency_p95": float(np.percentile(lat, 95)),
            "latency_p99": float(np.percentile(lat, 99))}


# ---------------------------------------------------------------------------
# E1 main / E3 rho-ablation / E4 ntn  (all reuse run_main_table)
# ---------------------------------------------------------------------------

def exp_main() -> None:
    for D in (50.0, 100.0):
        _log("E1 main  D=%.0f" % D)
        df = R.run_main_table(
            datasets=ALL_DATASETS, presets=["accesstier"],
            methods=MAIN_METHODS, seeds=SEEDS, deadline_ms=D,
            alpha=DEFAULT_ALPHA, lambda_delay=DEFAULT_LAMBDA,
            n_trials=None, max_subjects=None)
        _save(df, "table-main-d%.0f.csv" % D)


def exp_rho_ablation() -> None:
    presets = ["accesstier__belief-linear", "accesstier__belief-tier",
               "accesstier__belief-state", "accesstier__belief-oracle"]
    _log("E3 rho-ablation D=50")
    df = R.run_main_table(
        datasets=ALL_DATASETS, presets=presets, methods=["awf"],
        seeds=SEEDS, deadline_ms=50.0, alpha=DEFAULT_ALPHA,
        lambda_delay=DEFAULT_LAMBDA, n_trials=None, max_subjects=None)
    _save(df, "table-rho-ablation.csv")


def exp_ntn() -> None:
    presets = ["accesstier_leo00", "accesstier_leo10", "accesstier",
               "accesstier_leo30"]
    _log("E4 NTN robustness D=50")
    df = R.run_main_table(
        datasets=ALL_DATASETS, presets=presets, methods=KEY_METHODS,
        seeds=SEEDS, deadline_ms=50.0, alpha=DEFAULT_ALPHA,
        lambda_delay=DEFAULT_LAMBDA, n_trials=None, max_subjects=None)
    _save(df, "table-ntn.csv")


# ---------------------------------------------------------------------------
# E2 frontier (AWF lambda sweep)
# ---------------------------------------------------------------------------

def exp_frontier() -> None:
    _log("E2 frontier (multi-method knob sweep) D=50")
    rows: List[Dict] = []
    preset = "accesstier"
    for dataset in ALL_DATASETS:
        try:
            users = R._ensure_users(dataset)
        except Exception as exc:
            _log("  skip %s: %s" % (dataset, exc)); continue
        for seed in range(SEEDS):
            # fixed single-point reference baselines
            for meth in ["weighted", "fastest_only", "earliest_k:3",
                         "neurobci_sc"]:
                res, arr, y, nu = R.run_one(meth, seed, dataset, preset, users,
                                            deadline_ms=50.0,
                                            alpha=DEFAULT_ALPHA,
                                            lambda_delay=DEFAULT_LAMBDA)
                row = {"dataset": dataset, "seed": seed, "method": meth,
                       "sweep": np.nan,
                       "accuracy": metrics.accuracy(res, y)}
                row.update(_pct_latencies(res))
                rows.append(row)
            # First-k frontier.  Sweep every feasible committee size rather
            # than reporting only the favorable k=3 operating point.
            for k in range(1, len(users) + 1):
                res, arr, y, nu = R.run_one("earliest_k:%d" % k, seed,
                                            dataset, preset, users,
                                            deadline_ms=50.0,
                                            alpha=DEFAULT_ALPHA,
                                            lambda_delay=DEFAULT_LAMBDA)
                row = {"dataset": dataset, "seed": seed,
                       "method": "earliest_k", "sweep": k,
                       "accuracy": metrics.accuracy(res, y)}
                row.update(_pct_latencies(res))
                rows.append(row)
            # AWF lambda frontier
            for lam in AWF_LAMBDAS:
                res, arr, y, nu = R.run_one("awf", seed, dataset, preset, users,
                                            deadline_ms=50.0,
                                            alpha=DEFAULT_ALPHA,
                                            lambda_delay=lam)
                row = {"dataset": dataset, "seed": seed, "method": "awf",
                       "sweep": lam,
                       "accuracy": metrics.accuracy(res, y)}
                row.update(_pct_latencies(res))
                rows.append(row)
            # Generic-VoI lambda frontier (BCI-agnostic VoI, same knob as AWF)
            for lam in AWF_LAMBDAS:
                res, arr, y, nu = R.run_one("generic_voi", seed, dataset,
                                            preset, users, deadline_ms=50.0,
                                            alpha=DEFAULT_ALPHA,
                                            lambda_delay=lam)
                row = {"dataset": dataset, "seed": seed,
                       "method": "generic_voi", "sweep": lam,
                       "accuracy": metrics.accuracy(res, y)}
                row.update(_pct_latencies(res))
                rows.append(row)
            # Anytime cBCI confidence-threshold frontier
            for th in ANYTIME_THETAS:
                res, arr, y, nu = R.run_one("anytime_cbci:%g" % th, seed,
                                            dataset, preset, users,
                                            deadline_ms=50.0, alpha=1e-5,
                                            lambda_delay=1e-5)
                row = {"dataset": dataset, "seed": seed,
                       "method": "anytime_cbci", "sweep": th,
                       "accuracy": metrics.accuracy(res, y)}
                row.update(_pct_latencies(res))
                rows.append(row)
            # Anytime cBCI (confidence-weighted fusion) threshold frontier
            for th in ANYTIME_THETAS:
                res, arr, y, nu = R.run_one("anytime_cbci_cw:%g" % th, seed,
                                            dataset, preset, users,
                                            deadline_ms=50.0, alpha=1e-5,
                                            lambda_delay=1e-5)
                row = {"dataset": dataset, "seed": seed,
                       "method": "anytime_cbci_cw", "sweep": th,
                       "accuracy": metrics.accuracy(res, y)}
                row.update(_pct_latencies(res))
                rows.append(row)
            # DA-OS (He-Zhang) throughput/deadline exchange-rate frontier
            for bt in DA_OS_BETAS:
                res, arr, y, nu = R.run_one("da_os:%g" % bt, seed,
                                            dataset, preset, users,
                                            deadline_ms=50.0, alpha=1e-5,
                                            lambda_delay=1e-5)
                row = {"dataset": dataset, "seed": seed,
                       "method": "da_os", "sweep": bt,
                       "accuracy": metrics.accuracy(res, y)}
                row.update(_pct_latencies(res))
                rows.append(row)
            # Deadline-aware arrival-probability-threshold frontier
            for ta in DEADLINE_TAUS:
                res, arr, y, nu = R.run_one("deadline_aware:%g" % ta, seed,
                                            dataset, preset, users,
                                            deadline_ms=50.0, alpha=1e-5,
                                            lambda_delay=1e-5)
                row = {"dataset": dataset, "seed": seed,
                       "method": "deadline_aware", "sweep": ta,
                       "accuracy": metrics.accuracy(res, y)}
                row.update(_pct_latencies(res))
                rows.append(row)
        _log("  frontier done: %s" % dataset)
    _save(pd.DataFrame(rows), "table-frontier.csv")


# ---------------------------------------------------------------------------
# E7 alpha / gamma sensitivity (margin-threshold scale)
# ---------------------------------------------------------------------------

#: margin-stop scale-parameter sweep.  The margin threshold is a deterministic
#: function of alpha and the class count C: gamma = log((C-1)/alpha).  Sweeping
#: alpha therefore sweeps gamma; a smaller alpha means a larger (more
#: conservative) gamma so the margin safety-valve fires less often.  The delay
#: cost lambda is held at its default while alpha is varied.
ALPHA_SWEEP = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def exp_alpha() -> None:
    _log("E7 alpha/gamma sensitivity (AWF margin-scale sweep, "
         "lambda=%g) D=50" % DEFAULT_LAMBDA)
    rows: List[Dict] = []
    preset = "accesstier"
    for dataset in ALL_DATASETS:
        try:
            users = R._ensure_users(dataset)
        except Exception as exc:
            _log("  skip %s: %s" % (dataset, exc)); continue
        C = int(np.asarray(users[0].confusion).shape[0])
        for seed in range(SEEDS):
            for a in ALPHA_SWEEP:
                res, arr, y, nu = R.run_one("awf", seed, dataset, preset, users,
                                            deadline_ms=50.0, alpha=a,
                                            lambda_delay=DEFAULT_LAMBDA)
                row = {"dataset": dataset, "seed": seed, "method": "awf",
                       "alpha": a, "n_classes": C,
                       "gamma": float(np.log(max(C - 1, 1) / a)),
                       "accuracy": metrics.accuracy(res, y)}
                row.update(_pct_latencies(res))
                rows.append(row)
        _log("  alpha sweep done: %s" % dataset)
    _save(pd.DataFrame(rows), "table-alpha.csv")


# ---------------------------------------------------------------------------
# E8 lambda sensitivity (delay-cost scale)
# ---------------------------------------------------------------------------

#: delay-cost sweep.  lambda scales the per-millisecond penalty in the AWF
#: stopping rule; a larger lambda makes waiting more expensive so the decision
#: commits earlier.  The margin scale alpha is held at its default while lambda
#: is varied over the same decade grid as ALPHA_SWEEP.
LAMBDA_SWEEP = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def exp_lambda() -> None:
    _log("E8 lambda sensitivity (AWF delay-cost sweep, "
         "alpha=%g) D=50" % DEFAULT_ALPHA)
    rows: List[Dict] = []
    preset = "accesstier"
    for dataset in ALL_DATASETS:
        try:
            users = R._ensure_users(dataset)
        except Exception as exc:
            _log("  skip %s: %s" % (dataset, exc)); continue
        for seed in range(SEEDS):
            for lam in LAMBDA_SWEEP:
                res, arr, y, nu = R.run_one("awf", seed, dataset, preset, users,
                                            deadline_ms=50.0,
                                            alpha=DEFAULT_ALPHA,
                                            lambda_delay=lam)
                row = {"dataset": dataset, "seed": seed, "method": "awf",
                       "lambda": lam,
                       "accuracy": metrics.accuracy(res, y)}
                row.update(_pct_latencies(res))
                rows.append(row)
        _log("  lambda sweep done: %s" % dataset)
    _save(pd.DataFrame(rows), "table-lambda.csv")


# ---------------------------------------------------------------------------
# E5 deadline sensitivity
# ---------------------------------------------------------------------------

def exp_deadline() -> None:
    _log("E5 deadline sensitivity")
    Ds = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    rows: List[Dict] = []
    preset = "accesstier"
    for dataset in ALL_DATASETS:
        try:
            users = R._ensure_users(dataset)
        except Exception as exc:
            _log("  skip %s: %s" % (dataset, exc)); continue
        for D in Ds:
            for seed in range(SEEDS):
                ref, _, _, _ = R.run_one("deadline_partial", seed, dataset,
                                         preset, users, deadline_ms=D,
                                         alpha=DEFAULT_ALPHA,
                                         lambda_delay=DEFAULT_LAMBDA)
                for meth in KEY_METHODS:
                    res, arr, y, nu = R.run_one(meth, seed, dataset, preset,
                                                users, deadline_ms=D,
                                                alpha=DEFAULT_ALPHA,
                                                lambda_delay=DEFAULT_LAMBDA)
                    row = metrics.summarise_run(
                        method=meth, seed=seed, dataset=dataset, preset=preset,
                        results=res, arrivals_ms=arr, deadline_ms=D, y_true=y,
                        n_users=nu, baseline_results=ref)
                    row["deadline_ms"] = D
                    rows.append(row)
        _log("  deadline done: %s" % dataset)
    _save(pd.DataFrame(rows), "table-deadline.csv")


# ---------------------------------------------------------------------------
# E6 mobility / access-switching dynamics
# ---------------------------------------------------------------------------

def _mobility_run(method, seed, dataset, users, mode, preset, D, alpha, lam):
    rng = np.random.RandomState(seed)
    P_cube, y_true = data.align_test_trials(users, rng, n_trials=None)
    n_users = len(users)
    n_trials = P_cube.shape[1]

    if mode == "fixed":
        profiles = net.assign_accesstier_tiers(n_users, preset, rng)
        profiles_per_trial = [profiles] * n_trials
        arrivals_ms, _ = net.sample_arrivals(profiles, n_trials, rng)
    elif mode == "iid":
        profiles_per_trial = [net.assign_accesstier_tiers(n_users, preset, rng)
                              for _ in range(n_trials)]
        arrivals_ms = np.zeros((n_users, n_trials))
        for t, prof in enumerate(profiles_per_trial):
            a, _ = net.sample_arrivals(prof, 1, rng)
            arrivals_ms[:, t] = a[:, 0]
    elif mode == "markov":
        profiles_per_trial = net.assign_accesstier_markov(
            n_users, n_trials, rng, preset)
        arrivals_ms = np.zeros((n_users, n_trials))
        for t, prof in enumerate(profiles_per_trial):
            a, _ = net.sample_arrivals(prof, 1, rng)
            arrivals_ms[:, t] = a[:, 0]
    elif mode == "replay":
        profiles, arrivals_ms, _ = net.sample_accesstier_replay(
            n_users, n_trials, rng, preset)
        profiles_per_trial = [profiles] * n_trials
    else:
        raise ValueError(mode)

    val_acc = np.array([u.val_accuracy for u in users])
    confusions = [u.confusion for u in users]
    fn = fusion.get_method(method)
    results = []
    for t in range(n_trials):
        results.append(fn(P=P_cube[:, t, :], arrivals_ms=arrivals_ms[:, t],
                          deadline_ms=D, val_acc=val_acc,
                          profiles=profiles_per_trial[t], confusions=confusions,
                          alpha=alpha, lambda_delay=lam, y_true=int(y_true[t])))
    return results, arrivals_ms, y_true, n_users


def exp_mobility() -> None:
    _log("E6 mobility (fixed/iid/markov/replay) D=50")
    rows: List[Dict] = []
    preset = "accesstier"
    methods = ["earliest_k:3", "anytime_cbci_cw", "da_os", "anytime_cbci",
               "deadline_aware", "generic_voi", "logpool", "awf"]
    for dataset in ALL_DATASETS:
        try:
            users = R._ensure_users(dataset)
        except Exception as exc:
            _log("  skip %s: %s" % (dataset, exc)); continue
        for mode in ("fixed", "iid", "markov", "replay"):
            for seed in range(SEEDS):
                for meth in methods:
                    res, arr, y, nu = _mobility_run(
                        meth, seed, dataset, users, mode, preset, 50.0,
                        DEFAULT_ALPHA, DEFAULT_LAMBDA)
                    row = metrics.summarise_run(
                        method=meth, seed=seed, dataset=dataset, preset=preset,
                        results=res, arrivals_ms=arr, deadline_ms=50.0,
                        y_true=y, n_users=nu)
                    row["mobility"] = mode
                    rows.append(row)
        _log("  mobility done: %s" % dataset)
    _save(pd.DataFrame(rows), "table-mobility.csv")


EXPERIMENTS = {
    "main": exp_main,
    "frontier": exp_frontier,
    "rho": exp_rho_ablation,
    "ntn": exp_ntn,
    "deadline": exp_deadline,
    "mobility": exp_mobility,
    "alpha": exp_alpha,
    "lambda": exp_lambda,
}


def main(argv: Optional[Sequence[str]] = None) -> None:
    global SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(EXPERIMENTS),
                    default=list(EXPERIMENTS))
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args(argv)
    SEEDS = args.seeds
    t0 = time.time()
    for name in args.only:
        _log("=== START %s ===" % name)
        EXPERIMENTS[name]()
        _log("=== END %s (%.0fs elapsed) ===" % (name, time.time() - t0))
    _log("ALL DONE (%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
