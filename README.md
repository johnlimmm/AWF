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
matrices; held-out test posteriors are used only for final fusion evaluation.

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
| `plot-accesstier.py` | access-tier result plots |
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
