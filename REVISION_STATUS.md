# IoTJ Major Revision Status

This document records the clean revision campaign. Legacy CSVs generated from
older trace definitions must not be quoted in the revised manuscript.

## Completed code and evaluation changes

- The UCC 4G-LTE source now retains only `NetworkMode == "LTE"` rows. The old
  bank also pooled HSPA-family fallback rows while labeling the tier LTE.
- The trace cache has an explicit schema version. Schema 3 additionally stores
  a loss estimate aligned with each preserved contiguous sequence.
- Current LTE-only statistics are `n=663`, p50/p95/p99 =
  42.0/53.0/162.2 ms, loss = 1.16%, and effective arrival by 50 ms = 88.85%.
  The old 67% statement must be removed.
- Per-user scalar temperature is fitted only on validation NLL and frozen on
  test. Raw and calibrated posteriors are both retained.
- Policy parameters are selected on collaborative validation trials with the
  common objective `1 - accuracy + 1e-3 * mean latency (ms)` and are frozen on
  held-out test trials.
- First-k is swept over every feasible `k`; Confidence-Stop reaches
  `theta=1`; Arrival-Only, Throughput-Stop, Entropy-Gain, and AWF include their
  zero-cost/threshold endpoints.
- AWF ablations now include no margin, no arrival probability, no delay cost,
  no reliability, and scalar-accuracy reliability instead of a full confusion
  matrix.
- A non-clairvoyant Bellman reference exactly solves the discretized finite
  horizon on Zhou2016 (`N=4`, `C=3`).
- Held-out delay and loss beliefs use the first half of each tier's preserved
  sequences; arrivals replay the second half in order. A shared-link condition
  gives same-tier users common spikes and loss events.
- Homogeneous Wi-Fi, 5G, LTE, and LEO committees, cross-user error
  correlation, all-policy deadline sweeps, and worst-case runtime scaling are
  included.
- Accuracy significance is reported both as trial-level exact McNemar and as
  paired seed-level Wilcoxon tests. Latency has paired seed-level Wilcoxon
  tests.

## Key current results (15 seeds, calibrated posterior, D=50 ms)

Validation-selected AWF:

| Dataset | Accuracy | Mean latency | Full-Wait accuracy | Full-Wait latency |
|---|---:|---:|---:|---:|
| BNCI2014_001 | 99.50% | 11.02 ms | 99.69% | 50 ms |
| Zhou2016 | 96.44% | 24.80 ms | 97.20% | 50 ms |
| Weibo2014 | 82.04% | 31.30 ms | 82.31% | 50 ms |

AWF versus Confidence-Stop has no significant accuracy difference on any
dataset. AWF latency is significantly lower on BNCI2014_001 and Weibo2014 but
not on Zhou2016. The revised claims must present Confidence-Stop as a strong
competitor.

Temperature scaling preserves accuracy and improves mean NLL/ECE:

| Dataset | NLL raw -> calibrated | ECE raw -> calibrated |
|---|---:|---:|
| BNCI2014_001 | 0.7961 -> 0.7890 | 0.0774 -> 0.0649 |
| Zhou2016 | 0.5611 -> 0.5493 | 0.0766 -> 0.0710 |
| Weibo2014 | 1.4769 -> 1.4492 | 0.0919 -> 0.0811 |

On Zhou2016, the 5-ms DP reference has mean objective 0.05758 versus AWF
0.06042, an average gap of 0.00284. Grid sensitivity at 2.5/5/10 ms gives the
same accuracy on the first three seeds.

The no-margin ablation produces exactly the same accuracy and latency as full
AWF on all three datasets. The margin test changes the recorded stop reason
but adds no empirical performance here. It must be described as a safety
certificate rather than claimed as an independent source of gain.

Held-out-versus-oracle mean objective differences are small (absolute mean
approximately 0.0002--0.0015 depending on dataset and replay mode), including
the shared-link stress. This supports robustness to the measured train/test
trace shift but does not replace a live deployment.

The current worst-case Python implementation takes about 4 ms median at the
largest evaluated committee (`N=10`, `C=7`) after prefix-sum optimization of
the empirical residual-delay expectation. Runtime scales quadratically in
the number of users in the wait-until-all worst case.

## Manuscript and response-letter work still required

- Rewrite the optimization problem and constraints explicitly, then separate
  the exact Bellman objective from the one-step AWF approximation.
- Replace Fig. 1's neural network with CSP-LDA and fix Fig. 2's class indices
  from `1,...,N` to `1,...,C`.
- Replace legacy tables/figures with the revision CSVs and regenerated plots.
- State that UCC is a 1-Hz application-level KPI log and that `PINGAVG/2` is a
  one-way approximation, not packet-level uplink ground truth.
- Motivate a post-inference edge-fusion budget without implying that a
  20--30 ms saving materially shortens a several-second MI acquisition.
- Add transport security/privacy discussion (authenticated encryption,
  endpoint authentication, key management, metadata leakage, and measured or
  bounded overhead).
- Discuss online reliability updates under fatigue/session drift and broaden
  future work to semantic communication, edge scheduling, and adaptive user
  selection.
- Reframe originality around class- and user-specific decision-risk value
  under a hard deadline, not generic distributed early stopping.

## Reproduction

```bash
python -m unittest -v test_revision_invariants.py
python run-iotj-revision.py --only trace calibration tuning ablation dp heldout homogeneous runtime correlation deadlines
python plot-iotj-revision.py
python run-accesstier.py --only frontier
python run_replot_frontier.py
```
