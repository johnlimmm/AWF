"""Analyze the current Wi-Fi/5G/4G/LEO trace bank for a concise report."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "network-dataset-analysis"
OUT.mkdir(parents=True, exist_ok=True)
TIERS = ["wifi", "5g", "lte", "leo"]
LABEL = {"wifi": "Wi-Fi", "5g": "5G", "lte": "4G-LTE", "leo": "LEO"}
SOURCE = {
    "wifi": "KTH wireless-pr3d, IEEE 802.11g IRTT",
    "5g": "KTH wireless-pr3d, band-n78 5G IRTT",
    "lte": "UCC 5G dataset, LTE PINGAVG/2",
    "leo": "Casparsen Starlink IRTT",
}
MEASURE = {
    "wifi": "실측 uplink one-way delay",
    "5g": "실측 uplink one-way delay",
    "lte": "RTT의 절반으로 근사한 one-way delay",
    "leo": "실측 uplink one-way delay",
}


def load_bank():
    path = ROOT / "bcinet-tracebank.py"
    spec = importlib.util.spec_from_file_location("new_tracebank", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_tracebank()


def safe_acf1(x: np.ndarray) -> float:
    if x.size < 3 or np.std(x[:-1]) == 0 or np.std(x[1:]) == 0:
        return np.nan
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def summarize(bank) -> pd.DataFrame:
    rows = []
    for tier in TIERS:
        b = bank[tier]
        d = np.asarray(b.all_delays(), dtype=float)
        seqs = [np.asarray(s, dtype=float) for s in b.sequences if len(s) >= 10]
        acfs = np.asarray([safe_acf1(s) for s in seqs], dtype=float)
        drifts = np.asarray([
            100 * abs(np.mean(s[len(s)//2:]) - np.mean(s[:len(s)//2])) / np.mean(s)
            for s in seqs if np.mean(s) > 0
        ])
        half1 = np.concatenate([s[:len(s)//2] for s in seqs]) if seqs else d[:len(d)//2]
        half2 = np.concatenate([s[len(s)//2:] for s in seqs]) if seqs else d[len(d)//2:]
        ks = stats.ks_2samp(half1, half2).statistic
        p50, p95, p99 = np.percentile(d, [50, 95, 99])
        row = {
            "tier": LABEL[tier], "source": SOURCE[tier], "measurement": MEASURE[tier],
            "samples": len(d), "segments": len(seqs),
            "mean_ms": np.mean(d), "std_ms": np.std(d),
            "cv": np.std(d) / np.mean(d), "p50_ms": p50,
            "p95_ms": p95, "p99_ms": p99, "p99_p50": p99 / p50,
            "skew": stats.skew(d), "kurtosis": stats.kurtosis(d),
            "loss_pct": 100 * b.overall_loss(),
            "acf1_median": np.nanmedian(acfs) if acfs.size else np.nan,
            "drift_pct_median": np.median(drifts) if drifts.size else np.nan,
            "split_half_ks": ks,
        }
        for deadline in (20, 30, 40, 50, 75, 100):
            row[f"arrival_d{deadline}"] = np.mean(d <= deadline) * (1 - b.overall_loss())
        rows.append(row)
    return pd.DataFrame(rows)


def interpretation(row) -> str:
    if row.tier == "Wi-Fi":
        return ("중앙 지연은 매우 짧지만 드물게 큰 지연 spike가 나타나는 강한 "
                "우측 꼬리 환경")
    if row.tier == "5G":
        return ("지연 분포가 좁고 시간 변화도 작아 네 데이터 중 가장 안정적인 "
                "저지연 이동통신 환경")
    if row.tier == "4G-LTE":
        return ("중앙 지연과 분산이 크고 세션 간 차이도 커서 tight deadline에서 "
                "가장 불리한 광역 이동통신 환경")
    return ("평상시 지연은 비교적 집중되어 있지만 위성 링크 특유의 지연 변화와 "
            "드문 spike를 포함하는 비지상망 환경")


def report(df: pd.DataFrame) -> str:
    lines = [
        "# 신규 네트워크 데이터셋 특성 분석",
        "",
        "## 1. 데이터 구성",
        "",
        "| 데이터 | 출처 | 지연 정의 | 표본 수 | 연속 구간 |",
        "|---|---|---|---:|---:|",
    ]
    for r in df.itertuples():
        lines.append(f"| {r.tier} | {r.source} | {r.measurement} | {r.samples:,} | {r.segments} |")
    lines += [
        "",
        "Wi-Fi·5G·LEO는 패킷 단위 IRTT에서 얻은 실제 uplink one-way delay를 "
        "사용한다. 4G-LTE만 데이터 제약 때문에 PINGAVG/2를 one-way delay로 사용하므로, "
        "4G 수치는 경로 비대칭을 직접 관측한 값이 아니라 RTT 기반 근사값이다.",
        "Wi-Fi·5G·LEO 원본은 40,000개보다 크다. 계산량과 cache 크기를 제한하기 위해 "
        "전체 수신 지연 중 최대 40,000개를 무작위 선택하여 CDF 추정에 사용한다. 이 "
        "표본은 시간 순서를 보존하지 않으므로, 시간적 분석에는 순서를 유지한 별도의 "
        "연속 구간(sequence)을 사용한다.",
        "",
        "## 2. 핵심 통계",
        "",
        "| 데이터 | 평균 | p50 | p95 | p99 | CV | p99/p50 | 손실률 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in df.itertuples():
        lines.append(
            f"| {r.tier} | {r.mean_ms:.2f} ms | {r.p50_ms:.2f} | {r.p95_ms:.2f} | "
            f"{r.p99_ms:.2f} | {r.cv:.2f} | {r.p99_p50:.2f} | {r.loss_pct:.2f}% |"
        )
    lines += [
        "",
        "## 3. Deadline 내 도착률",
        "",
        "손실 응답을 deadline 내 미도착으로 포함한 비율이다.",
        "",
        "| 데이터 | D=20 | D=30 | D=40 | D=50 | D=75 | D=100 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in df.itertuples():
        lines.append(
            f"| {r.tier} | {r.arrival_d20:.2f} | {r.arrival_d30:.2f} | "
            f"{r.arrival_d40:.2f} | {r.arrival_d50:.2f} | "
            f"{r.arrival_d75:.2f} | {r.arrival_d100:.2f} |"
        )
    lines += [
        "",
        "## 4. 시간적 구조와 구간 변화",
        "",
        "| 데이터 | lag-1 자기상관 중앙값 | 구간 내 전·후반 평균 변화 | split-half KS |",
        "|---|---:|---:|---:|",
    ]
    for r in df.itertuples():
        lines.append(
            f"| {r.tier} | {r.acf1_median:.3f} | {r.drift_pct_median:.1f}% | "
            f"{r.split_half_ks:.3f} |"
        )
    lines += ["", "## 5. 데이터별 대표 특성", ""]
    for r in df.itertuples():
        lines.append(f"- **{r.tier}:** {interpretation(r)}.")
    lines += [
        "",
        "## 6. 교수님 보고용 요약",
        "",
        "이번 실험은 Wi-Fi, 5G, 4G-LTE, LEO의 서로 다른 실제 네트워크 측정 "
        "데이터를 사용한다. 네 데이터는 지연 중앙값뿐 아니라 분산, tail latency, 손실, "
        "시간적 변화가 서로 달라 heterogeneous end-to-end 경로 환경을 구성한다. Wi-Fi는 "
        "빠르지만 드문 큰 지연값을 포함하고, 5G는 좁고 안정적인 저지연 분포, 4G-LTE는 높은 지연과 큰 "
        "변동성, LEO는 집중된 평상시 지연과 위성 링크의 드문 spike를 대표한다. 따라서 "
        "동일한 평균 지연만으로 응답 도착을 판단하기 어렵고, deadline과 현재 시점에 따른 "
        "조건부 도착 가능성을 고려하는 stop-or-wait 평가에 적합하다.",
    ]
    return "\n".join(lines) + "\n"


def plot_distributions(bank, df: pd.DataFrame) -> None:
    colors = {"wifi": "#2E86AB", "5g": "#2A9D8F", "lte": "#E9A23B", "leo": "#7B4AB5"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for tier in TIERS:
        d = np.sort(np.asarray(bank[tier].all_delays(), dtype=float))
        y = np.arange(1, len(d) + 1) / len(d)
        axes[0].plot(d, y, lw=2, color=colors[tier], label=LABEL[tier])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("One-way delay (ms, log scale)")
    axes[0].set_ylabel("Empirical CDF")
    axes[0].set_title("(a) Delay distribution")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    deadlines = np.array([20, 30, 40, 50, 75, 100])
    for tier in TIERS:
        row = df[df["tier"] == LABEL[tier]].iloc[0]
        vals = [row[f"arrival_d{x}"] for x in deadlines]
        axes[1].plot(deadlines, vals, marker="o", lw=2, color=colors[tier], label=LABEL[tier])
    axes[1].set_xlabel("Deadline D (ms)")
    axes[1].set_ylabel("Fraction arriving within D")
    axes[1].set_ylim(-0.03, 1.04)
    axes[1].set_title("(b) Deadline feasibility (loss included)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "network-delay-distributions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    bank = load_bank()
    df = summarize(bank)
    df.to_csv(OUT / "network-dataset-statistics.csv", index=False, encoding="utf-8-sig")
    (OUT / "professor-network-dataset-report.md").write_text(report(df), encoding="utf-8")
    plot_distributions(bank, df)
    print(df[["tier", "samples", "p50_ms", "p95_ms", "p99_ms", "loss_pct"]].to_string(index=False))
    print(OUT / "professor-network-dataset-report.md")


if __name__ == "__main__":
    main()
