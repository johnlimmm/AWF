"""
run-bcinet-experiment.py
========================

Top-level experiment runner for the distributed cBCI study.

Examples
--------
Run a single configuration::

    python run-bcinet-experiment.py --dataset Cho2017 --preset balanced \\
        --methods mean logpool deadline_partial fastest_only earliest_k \\
                  reliability_latency awf oracle --seeds 30

Run *all* (dataset x preset x default methods)::

    python run-bcinet-experiment.py --all

Run only the user-scale (Cho2017 N sweep)::

    python run-bcinet-experiment.py --user-scale

Run only the AWF ablation table::

    python run-bcinet-experiment.py --ablation

Outputs land in ``results/``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Hyphenated-module loader
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, filename))
    if spec is None or spec.loader is None:
        raise ImportError(filename)
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec_module: needed by @dataclass on Python 3.13.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bcinet_data = _load_sibling("bcinet_data", "bcinet-data.py")
bcinet_network = _load_sibling("bcinet_network", "bcinet-network.py")
bcinet_fusion = _load_sibling("bcinet_fusion", "bcinet-fusion.py")
bcinet_metrics = _load_sibling("bcinet_metrics", "bcinet-metrics.py")


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

DATASETS = bcinet_data.SUPPORTED_DATASETS
PRESETS = list(bcinet_network.NETWORK_PRESETS.keys())


def _preset_arg(value: str) -> str:
    """argparse type for --preset: a known preset or 'markov:<stay_prob>'."""
    if value in PRESETS:
        return value
    if value.startswith("markov:"):
        try:
            sp = float(value.split(":", 1)[1])
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"invalid markov stay_prob in {value!r}")
        if not 0.0 <= sp <= 1.0:
            raise argparse.ArgumentTypeError(
                f"markov stay_prob must be in [0,1], got {sp}")
        return value
    raise argparse.ArgumentTypeError(
        f"invalid preset {value!r}; choose from {PRESETS} or 'markov:<float>'")

# A reasonable default method set. Variants like ``earliest_k`` are
# expanded into ``earliest_k:1, earliest_k:2, ...`` only when explicitly
# requested via --sweep-k.
DEFAULT_METHODS = [
    # Group 1
    "mean", "weighted", "logpool", "majority", "borda",
    # Group 2
    "fastest_only", "earliest_k:3", "deadline_partial", "latency_only:3",
    "anytime_cbci",
    # Group 3
    "best_user", "reliability_top_k:3", "reliability_weighted",
    "reliability_latency",
    # Group 4
    "oracle", "awf",
]

ABLATION_METHODS = ["awf", "awf_no_delay", "awf_no_reliability",
                    "awf_no_arrival"]

USER_SCALE_METHODS = [
    "mean", "logpool", "deadline_partial", "fastest_only",
    "earliest_k:3", "reliability_top_k:3", "reliability_latency",
    "awf", "oracle",
]


# ---------------------------------------------------------------------------
# Single (dataset, preset, method, seed) run
# ---------------------------------------------------------------------------

def run_one(method: str, seed: int, dataset: str, preset: str,
            users, n_trials: Optional[int] = None,
            deadline_ms: float = bcinet_network.DEFAULT_DEADLINE_MS,
            alpha: float = 1e-3, lambda_delay: float = 1e-3,
            user_subset: Optional[Sequence[int]] = None,
            rho_err: float = 0.0,
            arrival_scale: float = 1.0,
            arrival_drift: float = 0.0,
            extra_hops: int = 0,
            hop_mean_ms: float = 6.0,
            hop_spike_p: float = 0.10,
            hop_spike_mult: float = 5.0,
            ) -> Tuple[List, np.ndarray, np.ndarray, int]:
    """Run one method for one (dataset, preset, seed). Returns trial-level data."""
    rng = np.random.RandomState(seed)

    # Optional user subset (used by the user-scale experiment)
    if user_subset is not None:
        sub_users = [users[i] for i in user_subset]
    else:
        sub_users = users
    n_users = len(sub_users)

    # Trial cube (deterministic given seed, dataset, n_users) ---------------
    P_cube, y_true = bcinet_data.align_test_trials(sub_users, rng,
                                                   n_trials=n_trials,
                                                   rho_err=rho_err)
    n_trials_actual = P_cube.shape[1]

    # Per-seed network assignment: each seed defines one fixed deployment
    # scenario.  The user-to-tier mapping is sampled once at the start of the
    # seed and held fixed across all EEG trials of that seed; only the
    # per-trial delay/loss (arrival time) is resampled below.  Across seeds
    # the mapping is redrawn to average over diverse heterogeneous
    # deployments.  This avoids the unrealistic assumption that a user's
    # access tier changes on every trial.
    #
    # ``markov`` presets instead let each user's tier evolve as a Markov
    # chain across trials (temporal correlation / mobility), while keeping
    # the uniform marginal so results stay comparable with ``uniform_all``.
    if preset == "markov" or preset.startswith("markov:"):
        stay_prob = bcinet_network.MARKOV_STAY_PROB
        if ":" in preset:
            stay_prob = float(preset.split(":", 1)[1])
        profiles_per_trial = bcinet_network.assign_markov_tiers(
            n_users, n_trials_actual, rng, stay_prob=stay_prob)
    else:
        profiles = bcinet_network.assign_network_tiers(n_users, preset, rng)
        profiles_per_trial = [profiles] * n_trials_actual
    # Sample one arrival time per (user, trial) from that trial's profile.
    arrivals_ms = np.zeros((n_users, n_trials_actual), dtype=np.float64)
    for t, prof_t in enumerate(profiles_per_trial):
        a_t, _ = bcinet_network.sample_arrivals(prof_t, 1, rng)
        arrivals_ms[:, t] = a_t[:, 0]

    # Arrival-model mismatch / non-stationarity stress.  The realised arrival
    # times are perturbed while the policy's arrival belief (derived from
    # ``profiles``) is left untouched, so AWF acts on a stale/incorrect
    # residual-delay model.  ``arrival_scale`` applies a fixed multiplicative
    # bias (the true delays are larger/smaller than believed); ``arrival_drift``
    # ramps that bias linearly across trials to emulate within-session drift.
    if arrival_scale != 1.0 or arrival_drift != 0.0:
        denom = max(n_trials_actual - 1, 1)
        for t in range(n_trials_actual):
            factor = arrival_scale * (1.0 + arrival_drift * (t / denom))
            arrivals_ms[:, t] *= factor

    # Multi-hop transit stress.  AWF's arrival belief (from ``profiles``) models
    # only the single wireless access hop, but the realised arrival additionally
    # traverses ``extra_hops`` unmodeled network hops.  Each hop adds an
    # exponential transit delay plus an occasional heavy queueing spike, so the
    # true end-to-end delay is shifted and heavier-tailed than AWF believes.
    # This emulates an internet / multi-hop path without AWF modelling it.
    if extra_hops > 0:
        finite = np.isfinite(arrivals_ms)
        for h in range(extra_hops):
            base = rng.exponential(hop_mean_ms, size=arrivals_ms.shape)
            spike = (rng.random(arrivals_ms.shape) < hop_spike_p) * \
                rng.exponential(hop_mean_ms * hop_spike_mult, size=arrivals_ms.shape)
            arrivals_ms[finite] += (base + spike)[finite]

    val_acc = np.array([u.val_accuracy for u in sub_users])
    confusions = [u.confusion for u in sub_users]

    fn = bcinet_fusion.get_method(method)

    results = []
    for t in range(n_trials_actual):
        Pt = P_cube[:, t, :]                  # (n_users, n_classes)
        a_t = arrivals_ms[:, t]
        kwargs = dict(
            P=Pt, arrivals_ms=a_t, deadline_ms=deadline_ms,
            val_acc=val_acc, profiles=profiles_per_trial[t],
            confusions=confusions,
            alpha=alpha, lambda_delay=lambda_delay,
            y_true=int(y_true[t]),
        )
        results.append(fn(**kwargs))

    return results, arrivals_ms, y_true, n_users


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def _ensure_users(dataset: str, cache_dir: str = "cache",
                  max_subjects: Optional[int] = None):
    """Load (or build) the user pool for ``dataset``."""
    return bcinet_data.build_user_pool(
        dataset_name=dataset, cache_dir=cache_dir,
        seed=0, max_subjects=max_subjects)


def run_main_table(datasets: Iterable[str], presets: Iterable[str],
                   methods: Iterable[str], seeds: int,
                   deadline_ms: float, alpha: float,
                   lambda_delay: float, n_trials: Optional[int],
                   max_subjects: Optional[int]) -> pd.DataFrame:
    """Cross product of (dataset, preset, method, seed). Returns DataFrame."""
    rows: List[Dict] = []
    for dataset in datasets:
        try:
            users = _ensure_users(dataset, max_subjects=max_subjects)
        except Exception as exc:
            print(f"[{dataset}] failed to load: {exc}", file=sys.stderr)
            continue

        for preset in presets:
            # First compute the deadline-partial reference latency for
            # waiting_reduction. We use seed 0 only; this baseline is
            # deterministic per (dataset, preset, seed) so we recompute
            # for every seed below.
            for seed in tqdm(range(seeds),
                             desc=f"{dataset}/{preset}", leave=False):
                ref_results, ref_arrivals, ref_y, n_users = run_one(
                    "deadline_partial", seed, dataset, preset, users,
                    n_trials=n_trials, deadline_ms=deadline_ms,
                    alpha=alpha, lambda_delay=lambda_delay)
                for method in methods:
                    try:
                        results, arrivals, y, n_users = run_one(
                            method, seed, dataset, preset, users,
                            n_trials=n_trials, deadline_ms=deadline_ms,
                            alpha=alpha, lambda_delay=lambda_delay)
                    except Exception as exc:
                        print(f"  [{method}] failed: {exc}", file=sys.stderr)
                        continue
                    row = bcinet_metrics.summarise_run(
                        method=method, seed=seed, dataset=dataset,
                        preset=preset, results=results,
                        arrivals_ms=arrivals, deadline_ms=deadline_ms,
                        y_true=y, n_users=n_users,
                        baseline_results=ref_results)
                    rows.append(row)
    return pd.DataFrame(rows)


def run_user_scale(seeds: int, deadline_ms: float, alpha: float,
                   lambda_delay: float, preset: str = "balanced",
                   n_trials: Optional[int] = None,
                   user_counts: Sequence[int] = (4, 8, 16, 24, 32, 40, 52),
                   ) -> pd.DataFrame:
    """Cho2017 user-count sweep."""
    dataset = "Cho2017"
    try:
        users = _ensure_users(dataset)
    except Exception as exc:
        print(f"[{dataset}] failed to load: {exc}", file=sys.stderr)
        return pd.DataFrame()

    rows: List[Dict] = []
    pool_size = len(users)
    for N in user_counts:
        if N > pool_size:
            print(f"[user-scale] skip N={N} (pool has {pool_size})")
            continue
        for seed in tqdm(range(seeds), desc=f"N={N}", leave=False):
            rng = np.random.RandomState(10_000 + seed)
            subset = list(rng.choice(pool_size, size=N, replace=False))

            ref_results, ref_arrivals, ref_y, _ = run_one(
                "deadline_partial", seed, dataset, preset, users,
                user_subset=subset, n_trials=n_trials,
                deadline_ms=deadline_ms, alpha=alpha,
                lambda_delay=lambda_delay)

            for method in USER_SCALE_METHODS:
                try:
                    results, arrivals, y, n_users = run_one(
                        method, seed, dataset, preset, users,
                        user_subset=subset, n_trials=n_trials,
                        deadline_ms=deadline_ms, alpha=alpha,
                        lambda_delay=lambda_delay)
                except Exception as exc:
                    print(f"  [N={N}/{method}] failed: {exc}", file=sys.stderr)
                    continue
                row = bcinet_metrics.summarise_run(
                    method=method, seed=seed, dataset=dataset, preset=preset,
                    results=results, arrivals_ms=arrivals,
                    deadline_ms=deadline_ms, y_true=y,
                    n_users=n_users, baseline_results=ref_results)
                row["n_users"] = N
                rows.append(row)
    return pd.DataFrame(rows)


def run_ablation(seeds: int, deadline_ms: float, alpha: float,
                 lambda_delays: Sequence[float] = (1e-4, 1e-3, 1e-2),
                 datasets: Sequence[str] = ("Cho2017",),
                 presets: Sequence[str] = ("balanced",),
                 n_trials: Optional[int] = None,
                 max_subjects: Optional[int] = None) -> pd.DataFrame:
    """AWF ablation table sweeping lambda_delay and three reduced versions."""
    rows: List[Dict] = []
    for dataset in datasets:
        try:
            users = _ensure_users(dataset, max_subjects=max_subjects)
        except Exception as exc:
            print(f"[{dataset}] failed to load: {exc}", file=sys.stderr)
            continue
        for preset in presets:
            for lam in lambda_delays:
                for seed in tqdm(range(seeds),
                                 desc=f"abl {dataset}/{preset}/lam={lam}",
                                 leave=False):
                    ref_results, ref_arrivals, ref_y, _ = run_one(
                        "deadline_partial", seed, dataset, preset, users,
                        n_trials=n_trials, deadline_ms=deadline_ms,
                        alpha=alpha, lambda_delay=lam)
                    for method in ABLATION_METHODS:
                        results, arrivals, y, n_users = run_one(
                            method, seed, dataset, preset, users,
                            n_trials=n_trials, deadline_ms=deadline_ms,
                            alpha=alpha, lambda_delay=lam)
                        row = bcinet_metrics.summarise_run(
                            method=method, seed=seed, dataset=dataset,
                            preset=preset, results=results,
                            arrivals_ms=arrivals, deadline_ms=deadline_ms,
                            y_true=y, n_users=n_users,
                            baseline_results=ref_results)
                        row["lambda_delay"] = lam
                        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

CORR_ERR_METHODS = [
    "fastest_only", "awf", "anytime_cbci", "earliest_k:3", "weighted",
]


def run_corr_err(datasets: Iterable[str], seeds: int, deadline_ms: float,
                 alpha: float, lambda_delay: float,
                 n_trials: Optional[int], max_subjects: Optional[int],
                 preset: str = "uniform_all",
                 rho_values: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
                 ) -> pd.DataFrame:
    """Committee error-correlation sweep.

    For each ``rho_err`` a shared latent trial-hardness factor correlates
    user errors (see :func:`bcinet_data.align_test_trials`) while preserving
    each user's marginal accuracy. Tests whether AWF's accuracy-latency
    advantage survives as the conditional-independence assumption is relaxed.
    """
    rows: List[Dict] = []
    for dataset in datasets:
        try:
            users = _ensure_users(dataset, max_subjects=max_subjects)
        except Exception as exc:
            print(f"[{dataset}] failed to load: {exc}", file=sys.stderr)
            continue
        for rho in rho_values:
            for seed in tqdm(range(seeds),
                             desc=f"corr {dataset}/rho={rho}", leave=False):
                ref_results, _, _, _ = run_one(
                    "deadline_partial", seed, dataset, preset, users,
                    n_trials=n_trials, deadline_ms=deadline_ms,
                    alpha=alpha, lambda_delay=lambda_delay, rho_err=rho)
                for method in CORR_ERR_METHODS:
                    try:
                        results, arrivals, y, n_users = run_one(
                            method, seed, dataset, preset, users,
                            n_trials=n_trials, deadline_ms=deadline_ms,
                            alpha=alpha, lambda_delay=lambda_delay,
                            rho_err=rho)
                    except Exception as exc:
                        print(f"  [rho={rho}/{method}] failed: {exc}",
                              file=sys.stderr)
                        continue
                    row = bcinet_metrics.summarise_run(
                        method=method, seed=seed, dataset=dataset,
                        preset=preset, results=results,
                        arrivals_ms=arrivals, deadline_ms=deadline_ms,
                        y_true=y, n_users=n_users,
                        baseline_results=ref_results)
                    row["rho_err"] = rho
                    rows.append(row)
    return pd.DataFrame(rows)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Distributed cBCI experiment runner."
    )
    p.add_argument("--dataset", action="append", choices=DATASETS,
                   help="Dataset(s) to use. Can be repeated.")
    p.add_argument("--preset", action="append", type=_preset_arg,
                   help="Network preset(s); one of %s, or a parametrised "
                        "'markov:<stay_prob>'. Can be repeated." % PRESETS)
    p.add_argument("--methods", nargs="+", default=None,
                   help="Method names (see DEFAULT_METHODS).")
    p.add_argument("--seeds", type=int, default=30,
                   help="Number of random seeds per configuration.")
    p.add_argument("--deadline", type=float,
                   default=bcinet_network.DEFAULT_DEADLINE_MS,
                   help="Decision deadline D in milliseconds (default 100).")
    p.add_argument("--alpha", type=float, default=1e-3,
                   help="AWF margin alpha (default 1e-3).")
    p.add_argument("--lambda-delay", type=float, default=1e-3,
                   help="AWF delay-cost coefficient (default 1e-3).")
    p.add_argument("--n-trials", type=int, default=None,
                   help="Number of simulated trials per (dataset, seed). "
                        "Default = min over users.")
    p.add_argument("--max-subjects", type=int, default=None,
                   help="Cap subjects when building user pools (debug).")
    p.add_argument("--out-dir", default="results", help="Output directory.")

    p.add_argument("--all", action="store_true",
                   help="Run main table + user-scale + ablation.")
    p.add_argument("--user-scale", action="store_true",
                   help="Run only the Cho2017 user-count sweep.")
    p.add_argument("--ablation", action="store_true",
                   help="Run only the AWF ablation sweep.")
    p.add_argument("--corr-err", action="store_true",
                   help="Run only the committee error-correlation sweep.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    datasets = args.dataset or [DATASETS[0]]
    presets = args.preset or PRESETS
    methods = args.methods or DEFAULT_METHODS

    if args.all:
        datasets = DATASETS
        presets = PRESETS
        methods = DEFAULT_METHODS

    print(f"[run] datasets={datasets}\n      presets={presets}\n"
          f"      methods={methods}\n      seeds={args.seeds}")

    summary_path = os.path.join(args.out_dir, "summary-results.csv")

    # Main table -----------------------------------------------------------
    if not (args.user_scale or args.ablation or args.corr_err) or args.all:
        t0 = time.time()
        df = run_main_table(datasets, presets, methods,
                            seeds=args.seeds, deadline_ms=args.deadline,
                            alpha=args.alpha, lambda_delay=args.lambda_delay,
                            n_trials=args.n_trials,
                            max_subjects=args.max_subjects)
        if not df.empty:
            df.to_csv(summary_path, index=False)
            print(f"[run] wrote {summary_path} "
                  f"({len(df)} rows, {time.time()-t0:.1f}s)")

            # table-main: average across seeds and (dataset, preset)
            agg = (df.groupby("method")
                     .agg(accuracy=("accuracy", "mean"),
                          decision_latency=("decision_latency", "mean"),
                          used_response_ratio=("used_response_ratio", "mean"),
                          late_lost_response_ratio=("late_lost_response_ratio", "mean"),
                          waiting_reduction=("waiting_reduction", "mean"))
                     .reset_index())
            agg.to_csv(os.path.join(args.out_dir, "table-main.csv"), index=False)

            # table-network-presets
            agg2 = (df.groupby(["preset", "method"])
                      .agg(accuracy=("accuracy", "mean"),
                           decision_latency=("decision_latency", "mean"),
                           used_response_ratio=("used_response_ratio", "mean"))
                      .reset_index())
            agg2.to_csv(os.path.join(args.out_dir,
                                     "table-network-presets.csv"), index=False)

    # User-scale -----------------------------------------------------------
    if args.user_scale or args.all:
        df_us = run_user_scale(seeds=args.seeds, deadline_ms=args.deadline,
                               alpha=args.alpha, lambda_delay=args.lambda_delay,
                               n_trials=args.n_trials)
        if not df_us.empty:
            df_us.to_csv(os.path.join(args.out_dir,
                                      "table-user-scale-raw.csv"), index=False)
            agg_us = (df_us.groupby(["n_users", "method"])
                          .agg(accuracy=("accuracy", "mean"),
                               decision_latency=("decision_latency", "mean"),
                               used_response_ratio=("used_response_ratio", "mean"))
                          .reset_index())
            agg_us.to_csv(os.path.join(args.out_dir,
                                       "table-user-scale.csv"), index=False)

    # Ablation -------------------------------------------------------------
    if args.ablation or args.all:
        df_ab = run_ablation(seeds=args.seeds, deadline_ms=args.deadline,
                             alpha=args.alpha,
                             max_subjects=args.max_subjects,
                             n_trials=args.n_trials)
        if not df_ab.empty:
            df_ab.to_csv(os.path.join(args.out_dir,
                                      "table-ablation-raw.csv"), index=False)
            agg_ab = (df_ab.groupby(["lambda_delay", "method"])
                          .agg(accuracy=("accuracy", "mean"),
                               decision_latency=("decision_latency", "mean"),
                               used_response_ratio=("used_response_ratio", "mean"),
                               late_lost_response_ratio=("late_lost_response_ratio", "mean"))
                          .reset_index())
            agg_ab.to_csv(os.path.join(args.out_dir,
                                       "table-ablation.csv"), index=False)

    # Committee error-correlation sweep ------------------------------------
    if args.corr_err:
        df_ce = run_corr_err(datasets, seeds=args.seeds,
                             deadline_ms=args.deadline, alpha=args.alpha,
                             lambda_delay=args.lambda_delay,
                             n_trials=args.n_trials,
                             max_subjects=args.max_subjects)
        if not df_ce.empty:
            df_ce.to_csv(os.path.join(args.out_dir,
                                      "table-corr-err-raw.csv"), index=False)
            agg_ce = (df_ce.groupby(["rho_err", "method"])
                          .agg(accuracy=("accuracy", "mean"),
                               accuracy_sem=("accuracy", "sem"),
                               decision_latency=("decision_latency", "mean"))
                          .reset_index())
            agg_ce.to_csv(os.path.join(args.out_dir,
                                       "table-corr-err.csv"), index=False)
            print(f"[run] wrote table-corr-err.csv ({len(df_ce)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
