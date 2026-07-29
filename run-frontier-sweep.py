"""
run-frontier-sweep.py
======================
Unified, deterministic operating-point sweeps for the AWF paper.

Everything here is computed through the *same* ``run_one`` used to build the
main comparison table (``run-bcinet-experiment.py``). Because ``run_one`` is
deterministic given (seed, dataset, preset, n_trials, deadline) and the
fusion methods consume no RNG, every sweep shares *identical* per-trial
network draws as the main table. Consequently the endpoints

    * deadline sweep at D = 100 ms,
    * lambda sweep at lambda = 1e-5,
    * alpha  sweep at alpha  = 1e-5,
    * the AWF frontier point at lambda = 1e-5,

all coincide *exactly* with the AWF operating point in ``table-main.csv``,
removing the run-to-run "wobble" between Table 3 and the figures.

Outputs (all under ``results/``):
    frontier/frontier-raw.csv              row-level frontier sweep
    table-awf-sensitivity-lambda.csv       (value, accuracy, accuracy_sem, decision_latency)
    table-awf-sensitivity-alpha.csv        same schema
    deadline_sweep/d{D}/summary-results.csv  row-level, 5 methods per D

Usage:
    python run-frontier-sweep.py            # full run (6 datasets x 30 seeds)
    python run-frontier-sweep.py --smoke    # quick check (2 datasets x 3 seeds)
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, filename))
    if spec is None or spec.loader is None:
        raise ImportError(filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


exp = _load_sibling("run_bcinet_experiment", "run-bcinet-experiment.py")
run_one = exp.run_one
ensure_users = exp._ensure_users
summarise_run = exp.bcinet_metrics.summarise_run

# --- fixed configuration (matches the main table exactly) -------------------
PRESET = "uniform_all"
DEADLINE = 100.0
ALPHA0 = 1e-5
LAMBDA0 = 1e-5

# One-at-a-time sweep grids (same as the original sensitivity analysis).
LAMBDAS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
ALPHAS = [1e-5, 1e-3, 1e-2, 3e-2, 1e-1, 2e-1]

# Frontier knobs. Earliest-k is capped at k=4 = min N over datasets
# (Zhou2016, N=4) so that every operating point averages over the *same*
# six datasets. WeightedSum is the full-wait (k -> infinity) endpoint.
KS = [1, 2, 3, 4]
THETAS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

# Deadline grid for Figure 5.
DEADLINES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
DEADLINE_METHODS = ["awf", "weighted", "fastest_only", "earliest_k:3",
                    "anytime_cbci"]


def _eval(method: str, dataset: str, users, seed: int,
          deadline: float = DEADLINE, alpha: float = ALPHA0,
          lam: float = LAMBDA0) -> dict:
    """One deterministic (method, dataset, seed) evaluation -> summary row."""
    results, arrivals, y, n_users = run_one(
        method, seed, dataset, PRESET, users,
        n_trials=None, deadline_ms=deadline, alpha=alpha, lambda_delay=lam)
    row = summarise_run(
        method=method, seed=seed, dataset=dataset, preset=PRESET,
        results=results, arrivals_ms=arrivals, deadline_ms=deadline,
        y_true=y, n_users=n_users, baseline_results=results)
    return row


def _sem(s: pd.Series) -> float:
    n = len(s)
    return float(s.std() / np.sqrt(n)) if n > 1 else 0.0


def run_frontier(datasets, users_by_ds, seeds, out_dir):
    """AWF(lambda) vs Earliest-k(k) vs Anytime(theta), at D=100."""
    rows = []
    for ds in datasets:
        users = users_by_ds[ds]
        for seed in range(seeds):
            # AWF frontier (lambda knob)
            for lam in LAMBDAS:
                r = _eval("awf", ds, users, seed, lam=lam)
                rows.append({"knob": "awf", "value": lam, "label": "AWF",
                             "dataset": ds, "seed": seed,
                             "accuracy": r["accuracy"],
                             "decision_latency": r["decision_latency"]})
            # Earliest-k frontier (k knob)
            for k in KS:
                r = _eval(f"earliest_k:{k}", ds, users, seed)
                rows.append({"knob": "earliest_k", "value": k,
                             "label": "Earliest-k", "dataset": ds, "seed": seed,
                             "accuracy": r["accuracy"],
                             "decision_latency": r["decision_latency"]})
            # Anytime cBCI frontier (theta knob)
            for th in THETAS:
                r = _eval(f"anytime_cbci:{th}", ds, users, seed)
                rows.append({"knob": "anytime", "value": th,
                             "label": "Anytime cBCI", "dataset": ds,
                             "seed": seed, "accuracy": r["accuracy"],
                             "decision_latency": r["decision_latency"]})
            # Single-point references.
            for m, lab in [("weighted", "WeightedSum"),
                           ("fastest_only", "Fastest-only")]:
                r = _eval(m, ds, users, seed)
                rows.append({"knob": m, "value": np.nan, "label": lab,
                             "dataset": ds, "seed": seed,
                             "accuracy": r["accuracy"],
                             "decision_latency": r["decision_latency"]})
        print(f"  [frontier] {ds} done", flush=True)
    df = pd.DataFrame(rows)
    fdir = os.path.join(out_dir, "frontier")
    os.makedirs(fdir, exist_ok=True)
    df.to_csv(os.path.join(fdir, "frontier-raw.csv"), index=False)
    print(f"[frontier] wrote {len(df)} rows -> frontier/frontier-raw.csv",
          flush=True)
    return df


def run_ablation(datasets, users_by_ds, seeds, out_dir):
    """AWF component ablation at the lambda=1e-5 operating point (D=100).

    Evaluates the four ablation methods through the same deterministic
    ``run_one`` as the frontier, so the full-AWF row coincides exactly with
    the AWF operating point in the main table for the active preset.
    """
    rows = []
    for ds in datasets:
        users = users_by_ds[ds]
        for seed in range(seeds):
            for m in exp.ABLATION_METHODS:
                r = _eval(m, ds, users, seed)
                rows.append({"method": m, "preset": PRESET,
                             "dataset": ds, "seed": seed,
                             "accuracy": r["accuracy"],
                             "decision_latency": r["decision_latency"]})
        print(f"  [ablation] {ds} done", flush=True)
    df = pd.DataFrame(rows)
    adir = os.path.join(out_dir, "ablation")
    os.makedirs(adir, exist_ok=True)
    df.to_csv(os.path.join(adir, "summary-results.csv"), index=False)
    agg = (df.groupby("method")
             .agg(accuracy=("accuracy", "mean"),
                  decision_latency=("decision_latency", "mean"))
             .reset_index())
    print("[ablation] aggregated (method, acc, lat):", flush=True)
    for _, rr in agg.iterrows():
        print(f"    {rr['method']:>20s}  acc={rr['accuracy']:.4f}  "
              f"lat={rr['decision_latency']:.2f}ms", flush=True)
    print(f"[ablation] wrote {len(df)} rows -> ablation/summary-results.csv",
          flush=True)
    return df


def run_sensitivity(datasets, users_by_ds, seeds, out_dir):
    """One-at-a-time lambda and alpha sweeps for AWF (D=100)."""
    def _sweep(param, values):
        rows = []
        for v in values:
            lam = v if param == "lambda" else LAMBDA0
            alpha = v if param == "alpha" else ALPHA0
            for ds in datasets:
                users = users_by_ds[ds]
                for seed in range(seeds):
                    r = _eval("awf", ds, users, seed, alpha=alpha, lam=lam)
                    rows.append({"value": v, "accuracy": r["accuracy"],
                                 "decision_latency": r["decision_latency"]})
            print(f"  [{param}] {v:.0e} done", flush=True)
        df = pd.DataFrame(rows)
        agg = (df.groupby("value")
                 .agg(accuracy=("accuracy", "mean"),
                      accuracy_sem=("accuracy", _sem),
                      decision_latency=("decision_latency", "mean"))
                 .reset_index())
        return agg

    agg_lam = _sweep("lambda", LAMBDAS)
    agg_alp = _sweep("alpha", ALPHAS)
    agg_lam.to_csv(os.path.join(out_dir, "table-awf-sensitivity-lambda.csv"),
                   index=False)
    agg_alp.to_csv(os.path.join(out_dir, "table-awf-sensitivity-alpha.csv"),
                   index=False)
    print("[sensitivity] wrote lambda/alpha CSVs", flush=True)
    return agg_lam, agg_alp


def run_deadline(datasets, users_by_ds, seeds, out_dir):
    """Sweep the deadline D for the five headline methods."""
    for D in DEADLINES:
        rows = []
        for ds in datasets:
            users = users_by_ds[ds]
            for seed in range(seeds):
                for m in DEADLINE_METHODS:
                    r = _eval(m, ds, users, seed, deadline=float(D))
                    rows.append({"method": m, "preset": PRESET,
                                 "dataset": ds, "seed": seed,
                                 "accuracy": r["accuracy"],
                                 "decision_latency": r["decision_latency"]})
        ddir = os.path.join(out_dir, "deadline_sweep", f"d{D}")
        os.makedirs(ddir, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(ddir, "summary-results.csv"), index=False)
        print(f"  [deadline] D={D} done", flush=True)
    print("[deadline] wrote per-D summaries", flush=True)


def main(argv=None) -> int:
    global PRESET
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="Quick check: 2 datasets x 3 seeds.")
    p.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    p.add_argument("--preset", default=PRESET,
                   help="Network preset (e.g. uniform_all, realtier).")
    p.add_argument("--skip-frontier", action="store_true")
    p.add_argument("--skip-sensitivity", action="store_true")
    p.add_argument("--skip-deadline", action="store_true")
    p.add_argument("--ablation", action="store_true",
                   help="Also run the AWF component-ablation sweep.")
    args = p.parse_args(argv)

    PRESET = args.preset

    # Resolve relative cache/ and results/ paths against the repo root so the
    # cached user pools are found regardless of the terminal's CWD.
    os.chdir(HERE)

    datasets = list(exp.DATASETS)
    seeds = 30
    if args.smoke:
        datasets = datasets[:2]
        seeds = 3

    print(f"[run] datasets={datasets} seeds={seeds}", flush=True)
    t0 = time.time()
    users_by_ds = {ds: ensure_users(ds) for ds in datasets}
    print(f"[run] loaded {len(users_by_ds)} user pools "
          f"({time.time()-t0:.1f}s)", flush=True)

    if not args.skip_frontier:
        run_frontier(datasets, users_by_ds, seeds, args.out_dir)
    if args.ablation:
        run_ablation(datasets, users_by_ds, seeds, args.out_dir)
    if not args.skip_sensitivity:
        run_sensitivity(datasets, users_by_ds, seeds, args.out_dir)
    if not args.skip_deadline:
        run_deadline(datasets, users_by_ds, seeds, args.out_dir)

    print(f"[run] all done in {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
