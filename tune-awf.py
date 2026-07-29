"""
tune-awf.py
===========
Quick sweep of AWF (lambda_delay, alpha) to find settings that maximise
accuracy while still achieving meaningful latency reduction.

Uses only cached user pools (no MOABB re-download needed).
Runs on all cached datasets x all 4 presets x 10 seeds per config.

Usage:
    python tune-awf.py
"""
import importlib.util, os, sys, itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))

def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

net  = _load("bcinet_network",  "bcinet-network.py")
fus  = _load("bcinet_fusion",   "bcinet-fusion.py")
met  = _load("bcinet_metrics",  "bcinet-metrics.py")
data = _load("bcinet_data",     "bcinet-data.py")

PRESETS  = list(net.NETWORK_PRESETS.keys())
DEADLINE = net.DEFAULT_DEADLINE_MS  # 100 ms

# ── sweep grid ──────────────────────────────────────────────────────────────
LAMBDAS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]     # delay cost coefficient
ALPHAS  = [1e-5, 1e-4, 1e-3]                  # margin confidence threshold
SEEDS   = 10                                   # seeds per config (fast)
N_TRIALS = 200                                 # simulated trials per seed

# ── load cached user pools ──────────────────────────────────────────────────
cache_dir = os.path.join(HERE, "cache")
users_by_ds = {}
for fn in os.listdir(cache_dir):
    if fn.startswith("users-") and fn.endswith("-seed0.pkl"):
        import pickle
        ds = fn[len("users-"):-len("-seed0.pkl")]
        with open(os.path.join(cache_dir, fn), "rb") as f:
            users_by_ds[ds] = pickle.load(f)
print(f"Loaded {len(users_by_ds)} datasets: {list(users_by_ds.keys())}")

# ── helper: one (dataset, preset, lambda, alpha, seed) run ──────────────────
def _run_one(users, preset, lam, alpha, seed):
    rng = np.random.RandomState(seed)
    n_users = len(users)
    P_cube, y_true = data.align_test_trials(users, rng, n_trials=N_TRIALS)
    profiles  = net.assign_network_tiers(n_users, preset, rng)
    arrivals, _ = net.sample_arrivals(profiles, P_cube.shape[1], rng)
    val_acc   = np.array([u.val_accuracy for u in users])
    confs     = [u.confusion for u in users]

    awf_fn = fus.get_method("awf")
    dp_fn  = fus.get_method("deadline_partial")

    awf_res, dp_res = [], []
    for t in range(P_cube.shape[1]):
        kw = dict(P=P_cube[:,t,:], arrivals_ms=arrivals[:,t],
                  deadline_ms=DEADLINE, val_acc=val_acc,
                  profiles=profiles, confusions=confs,
                  alpha=alpha, lambda_delay=lam)
        awf_res.append(awf_fn(**kw))
        dp_res.append(dp_fn(**kw))

    awf_row = met.summarise_run("awf", seed, "multi", preset,
                                 awf_res, arrivals, DEADLINE, y_true,
                                 n_users, baseline_results=dp_res)
    dp_row  = met.summarise_run("dp",  seed, "multi", preset,
                                 dp_res,  arrivals, DEADLINE, y_true,
                                 n_users, baseline_results=dp_res)
    return awf_row, dp_row

# ── main sweep ───────────────────────────────────────────────────────────────
rows = []
configs = list(itertools.product(LAMBDAS, ALPHAS, PRESETS))
for lam, alpha, preset in tqdm(configs, desc="sweep"):
    for ds, users in users_by_ds.items():
        for seed in range(SEEDS):
            try:
                awf_r, dp_r = _run_one(users, preset, lam, alpha, seed)
                awf_r.update({"lambda_delay": lam, "alpha": alpha,
                               "dataset": ds, "seed": seed})
                rows.append(awf_r)
            except Exception as e:
                pass

df = pd.DataFrame(rows)

# ── aggregate ────────────────────────────────────────────────────────────────
agg = (df.groupby(["lambda_delay", "alpha"])
         .agg(accuracy         = ("accuracy",           "mean"),
              accuracy_std     = ("accuracy",           "std"),
              decision_latency = ("decision_latency",   "mean"),
              used_ratio       = ("used_response_ratio","mean"),
              wait_reduction   = ("waiting_reduction",  "mean"))
         .reset_index()
         .sort_values("accuracy", ascending=False))

print("\n=== AWF tuning results (sorted by accuracy) ===")
print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ── per-preset breakdown for the top 3 configs ───────────────────────────────
top3 = agg.head(3)[["lambda_delay","alpha"]].values
print("\n=== Per-preset breakdown for top-3 configs ===")
for lam, alpha in top3:
    sub = df[(df["lambda_delay"]==lam) & (df["alpha"]==alpha)]
    pr  = (sub.groupby("preset")
              .agg(accuracy=("accuracy","mean"),
                   latency =("decision_latency","mean"))
              .reset_index())
    print(f"\nλ={lam:.0e}, α={alpha:.0e}")
    print(pr.to_string(index=False))

# ── recommend best config ─────────────────────────────────────────────────────
# "best" = highest accuracy among configs with wait_reduction > 0.5
cand = agg[agg["wait_reduction"] > 0.5]
if not cand.empty:
    best = cand.iloc[0]
    print(f"\n★ Recommended: λ={best['lambda_delay']:.0e}, α={best['alpha']:.0e}")
    print(f"  accuracy={best['accuracy']:.4f}  "
          f"latency={best['decision_latency']:.1f}ms  "
          f"wait_reduction={best['wait_reduction']:.3f}")
    # Save recommended config
    with open(os.path.join(HERE, "awf-best-config.txt"), "w") as f:
        f.write(f"lambda_delay={best['lambda_delay']}\nalpha={best['alpha']}\n")
    print("  → saved to awf-best-config.txt")

agg.to_csv(os.path.join(HERE, "results", "table-awf-tuning.csv"), index=False)
print("\nFull tuning table saved to results/table-awf-tuning.csv")
