# AWF: Adaptive Wait-Fuse for Networked cBCI

This repository contains the experimental code for **Adaptive Wait-Fuse
(AWF)**, an online stop-or-wait policy for collaborative BCI over
heterogeneous wireless access.

AWF receives user-level BCI posterior vectors asynchronously and decides
whether the currently available evidence is sufficient or whether an
outstanding response is still worth waiting for before the decision deadline.
The policy combines:

1. the confidence of the current fused decision,
2. the expected reduction in classification risk from a missing response,
3. its conditional probability of arriving before the deadline, and
4. the cost of additional waiting.

## Evaluation scope

### BCI datasets

The experiments use three public multi-class motor-imagery datasets through
MOABB:

| Dataset | Users | Classes | Role |
|---|---:|---:|---|
| Zhou2016 | 4 | 3 | small, relatively uniform committee |
| BNCI2014_001 | 9 | 4 | large differences among users |
| Weibo2014 | 10 | 7 | most difficult multi-class setting |

Each subject is treated as one BCI user. EEG is band-pass filtered at
8-30 Hz, represented with six CSP components, and classified with shrinkage
LDA. Each subject is split into approximately 56% training, 14% validation,
and 30% test data. Validation data estimates user reliability and confusion
matrices and fits one scalar posterior temperature per user. The fitted
temperature is then frozen and applied to held-out test posteriors; raw test
posteriors are retained for the calibration ablation. Held-out test labels are
used only for final evaluation.

Because these datasets are not simultaneous multi-user recordings,
class-matched held-out trials from different subjects are combined to create
simulated collaborative trials.

### Network traces

| Access | Source | Delay used |
|---|---|---|
| Wi-Fi | KTH wireless-pr3d, IEEE 802.11g IRTT | measured uplink one-way delay |
| 5G | KTH wireless-pr3d, band-n78 5G IRTT | measured uplink one-way delay |
| 4G-LTE | UCC 5G dataset | `PINGAVG / 2` approximation |
| LEO | Casparsen Starlink IRTT | measured uplink one-way delay |

Wi-Fi and 5G loss is inferred from missing probes in the nominal 10 ms probe
cadence. LTE uses the source `PINGLOSS` field. LEO loss is inferred from
missing IRTT sequence numbers.

The LTE tier keeps only UCC rows with `NetworkMode == "LTE"`. HSPA/UMTS/EDGE
fallback rows are excluded so the sample definition matches the 4G-LTE label.
The trace-bank cache carries a schema version and is rebuilt automatically if
this source definition changes.

The main experiment independently bootstraps per-response delays from each
access tier's empirical CDF and applies its measured mean loss probability.
The replay experiment instead consumes ordered trace segments to preserve
short-term delay correlation. Raw measurement files and the generated trace
bank are intentionally not stored in Git.

## Core files

| File | Purpose |
|---|---|
| `bcinet-data.py` | MOABB loading, preprocessing, subject-wise CSP-LDA training, trial construction |
| `bcinet-tracebank.py` | Wi-Fi/5G/LTE/LEO trace parsing and empirical trace-bank construction |
| `bcinet-network.py` | access assignment, delay/loss sampling, arrival probability |
| `bcinet-fusion.py` | fusion baselines, AWF, and ablations |
| `bcinet-metrics.py` | accuracy, latency, response-use, and stopping metrics |
| `run-bcinet-experiment.py` | common experiment runner |
| `run-accesstier.py` | main, frontier, arrival-belief, NTN, deadline, and replay experiments |
| `run-iotj-revision.py` | isolated major-revision calibration, tuning, ablation, significance, and DP experiments |
| `plot-accesstier.py` | access-tier result plots |
| `test_revision_invariants.py` | deterministic policy-endpoint and ablation regression tests |
| `analyze_new_network_datasets.py` | network trace statistics and CDF/deadline plots |

Additional `run_*.py` scripts contain robustness checks for held-out trace
beliefs, cross-source shift, temporal correlation, non-stationarity,
multi-hop delay, loss removal, and trace-pool size.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The first MOABB run downloads the EEG datasets and may require several
gigabytes. Trained user pools are cached under `cache/`; this directory is not
version controlled.

## Expected trace locations

`bcinet-tracebank.py` expects the source data under `traces/`:

```text
traces/
  wifi_raw/extracted/IEEE80211g/dataset_*_172.parquet
  wifi_raw/extracted/COTS-5G/f_dataset*.parquet
  cellular/5G-production-dataset/**/*.csv
  leo_raw/starlink-irtt.zip
```

Build or refresh the local trace bank:

```bash
python run_rebuild_tracebank.py
```

The generated `traces/tracebank.pkl` is local cache data and is excluded from
Git.

## Running the experiments

Run the complete access-tier suite with the default 15 seeds:

```bash
python run-accesstier.py
```

Run selected experiments:

```bash
python run-accesstier.py --only main frontier deadline mobility
```

Generate the main figures:

```bash
python plot-accesstier.py
```

Run the isolated IoTJ revision campaign (results are written under
`results/iotj-revision-final/` with a resumable manifest and source hashes):

```bash
python run-iotj-revision.py --only trace calibration tuning calibration_policy ablation dp heldout homogeneous runtime correlation deadlines
python plot-iotj-revision.py --out-dir results/iotj-revision-final
```

For each dataset, First-k, Confidence-Stop, Anytime-cBCI, Arrival-Only,
Throughput-Stop, Entropy-Gain, and AWF control parameters are selected using
collaborative trials constructed only from the validation split. Every method
minimizes the same validation objective
`1 - accuracy + 1e-3 * mean_latency_ms`; the selected value is frozen before
held-out test evaluation. The runner also writes trial-level paired records,
exact McNemar tests for accuracy, paired Wilcoxon tests, and Holm-adjusted
family-wise p-values.

The `calibration_policy` experiment holds the tuned AWF parameter, EEG trial
sampling, user-to-access assignment, and realized delay/loss samples fixed,
then changes only raw versus validation-temperature-scaled posteriors. It
reports AWF accuracy, latency, stopping-reason rates, and trial-level decision
agreement.

The `dp` experiment compares AWF with an exact Bellman solution of a
time-discretized finite-horizon model on Zhou2016 (four users). The DP
integrates every possible future arrival subset and confusion-matrix report;
it does not observe realized future delays, posteriors, or labels. It is an
approximation-gap reference, not a scalable online baseline.

The `heldout` experiment estimates both delay and loss beliefs on the first
half of each tier's preserved sequences and replays the second half in order.
It includes a shared-link stress in which users on the same access experience
the same spikes and loss events. The `homogeneous` experiment assigns every
committee member to Wi-Fi, 5G, LTE, or LEO in turn. `runtime` reports a
worst-case user-scaling benchmark and policy-state memory.

Run the deterministic revision checks before a long campaign:

```bash
python -m unittest -v test_revision_invariants.py
```

Recalculate the network dataset summary:

```bash
python analyze_new_network_datasets.py
```

The default AWF operating parameters are `alpha = 1e-3` and
`lambda_delay = 1e-3`. The main evaluation uses deadlines of 50 ms and
100 ms and 15 random seeds.

## Interpretation and limitations

This is a trace-driven policy evaluation, not a live heterogeneous-network
cBCI testbed. It preserves measured per-access delay distributions and mean
loss rates. The basic bootstrap experiment does not preserve burst loss,
delay-loss dependence, or full protocol, queueing, retransmission, and routing
behavior. Ordered trace replay and distribution-shift experiments are
provided to test sensitivity beyond the matched empirical-CDF setting.

The 4G-LTE one-way delay is an RTT/2 approximation, unlike the directly
measured IRTT uplink delays used for Wi-Fi, 5G, and LEO.

## Included analysis summary

The compact reports and plots under `results/network-dataset-analysis/`
document the trace distributions, per-user BCI results, sampling procedure,
trial counts, loss calculation, and limitations. Large raw tables, runtime
logs, caches, paper sources, and raw measurement datasets are excluded.
