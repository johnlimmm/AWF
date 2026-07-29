"""Build a professor-facing dataset analysis from cached EEG users and results.

The report deliberately labels policy results as preliminary when they were
generated before the empirical residual-delay implementation was corrected.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "accesstier"
DATASETS = ["Zhou2016", "BNCI2014_001", "Weibo2014"]
DATASET_ORDER = {name: i for i, name in enumerate(DATASETS)}
METHOD_LABELS = {
    "fastest_only": "First-Arrival",
    "earliest_k:3": "First-k (k=3)",
    "anytime_cbci_cw": "Anytime-cBCI",
    "anytime_cbci": "Confidence-Stop",
    "deadline_aware": "Arrival-Only",
    "generic_voi": "Entropy-Gain",
    "logpool": "Full-Wait (log)",
    "weighted": "Full-Wait (linear)",
    "awf": "AWF",
}


def load_runner():
    path = ROOT / "run-bcinet-experiment.py"
    spec = importlib.util.spec_from_file_location("dataset_report_runner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def aligned_trial_count(users) -> int:
    """Match the runner's default number of simulated collaborative trials."""
    return int(min(min(len(u.test_labels) for u in users), 400))


def eeg_summary(runner) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        users = runner._ensure_users(dataset)
        val_acc = np.asarray([u.val_accuracy for u in users], dtype=float)
        val_f1 = np.asarray([u.val_macro_f1 for u in users], dtype=float)
        test_counts = np.asarray([len(u.test_labels) for u in users], dtype=int)
        rows.append({
            "dataset": dataset,
            "subjects": len(users),
            "classes": users[0].n_classes,
            "aligned_trials": aligned_trial_count(users),
            "test_trials_subject_mean": test_counts.mean(),
            "test_trials_subject_min": test_counts.min(),
            "test_trials_subject_max": test_counts.max(),
            "val_accuracy_mean": val_acc.mean(),
            "val_accuracy_std_subject": val_acc.std(ddof=1),
            "val_accuracy_min": val_acc.min(),
            "val_accuracy_max": val_acc.max(),
            "val_macro_f1_mean": val_f1.mean(),
            "val_macro_f1_std_subject": val_f1.std(ddof=1),
        })
    return pd.DataFrame(rows)


def policy_summary() -> pd.DataFrame:
    frames = []
    for deadline in (50, 100):
        df = pd.read_csv(OUT / f"table-main-d{deadline}.csv")
        df = df[df["method"].isin(METHOD_LABELS)].copy()
        agg = df.groupby(["dataset", "method"], as_index=False).agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std_seed=("accuracy", "std"),
            latency_mean=("decision_latency", "mean"),
            latency_std_seed=("decision_latency", "std"),
            used_ratio_mean=("used_response_ratio", "mean"),
        )
        agg["deadline_ms"] = deadline
        agg["method_label"] = agg["method"].map(METHOD_LABELS)
        frames.append(agg)
    return pd.concat(frames, ignore_index=True)


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{100.0 * x:.{digits}f}"


def build_report(eeg: pd.DataFrame, policy: pd.DataFrame) -> str:
    lines = [
        "# EEG·네트워크 Trace·Robustness 종합 분석 보고서 (교수님 보고용 초안)",
        "",
        "## 1. 분석 목적과 주의사항",
        "",
        "본 분석은 세 개의 다중 클래스 motor-imagery EEG 데이터셋과 "
        "BLE/Wi-Fi/5G/4G/LEO 네트워크 trace를 결합한 cBCI 실험을 설명한다. "
        "EEG 통계는 캐시된 사용자별 test posterior와 calibration 성능에서 직접 "
        "집계했으며, 네트워크 통계는 trace bank의 원시 통계 산출물을 사용했다.",
        "",
        "> **중요:** 아래 AWF 정책 성능은 경험적 조건부 잔여지연 계산을 수정하기 "
        "전에 생성된 기존 CSV를 분석한 잠정값이다. 데이터셋 자체의 기술통계는 "
        "유효하지만, 최종 성능 수치는 핵심 실험 재실행 후 갱신해야 한다.",
        "",
        "## 2. EEG 데이터셋 구성",
        "",
        "| 데이터셋 | 사용자(피험자) | 클래스 | seed당 정렬 trial | 사용자별 calibration 정확도 | 범위 | macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in eeg.sort_values("dataset", key=lambda s: s.map(DATASET_ORDER)).iterrows():
        lines.append(
            f"| {row.dataset} | {int(row.subjects)} | {int(row.classes)} | "
            f"{int(row.aligned_trials)} | {fmt_pct(row.val_accuracy_mean)}±"
            f"{fmt_pct(row.val_accuracy_std_subject)}% | "
            f"{fmt_pct(row.val_accuracy_min)}–{fmt_pct(row.val_accuracy_max)}% | "
            f"{fmt_pct(row.val_macro_f1_mean)}% |"
        )

    lines += [
        "",
        "seed당 정렬 trial 수는 가장 적은 사용자의 test trial 수를 기준으로 한다. 각 "
        "그룹 trial에서는 모든 사용자에게 존재하는 정답 클래스를 먼저 뽑고, 사용자마다 "
        "그 클래스의 실제 test trial을 복원추출한다. 따라서 데이터셋 간 표본 수 차이는 "
        "사용자별 test set 크기에서 발생한다. 15개 seed를 "
        "기준으로 Zhou2016 1,965건, BNCI2014_001 2,595건, Weibo2014 2,250건으로, "
        "방법·deadline별 총 6,810개의 그룹 결정을 평가한다.",
        "",
        "## 3. 데이터셋별 난이도와 그룹 융합 결과",
        "",
        "### D=50 ms 핵심 비교 (15 deployment seeds)",
        "",
        "| 데이터셋 | AWF 정확도/지연 | Full-Wait 정확도/지연 | 정확도 차이 | 지연 감소 | Confidence-Stop 정확도 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        d = policy[(policy.dataset == dataset) & (policy.deadline_ms == 50)]
        awf = d[d.method == "awf"].iloc[0]
        fw = d[d.method == "logpool"].iloc[0]
        cs = d[d.method == "anytime_cbci"].iloc[0]
        gap_pp = 100 * (fw.accuracy_mean - awf.accuracy_mean)
        saving = 100 * (1 - awf.latency_mean / fw.latency_mean)
        lines.append(
            f"| {dataset} | {fmt_pct(awf.accuracy_mean)}% / {awf.latency_mean:.1f} ms | "
            f"{fmt_pct(fw.accuracy_mean)}% / {fw.latency_mean:.1f} ms | "
            f"-{gap_pp:.1f}%p | {saving:.0f}% | {fmt_pct(cs.accuracy_mean)}% |"
        )

    lines += [
        "",
        "- **BNCI2014_001은 가장 쉬운 그룹 결정 환경**이다. 4-class임에도 Full-Wait가 "
        "약 99%에 도달하며 AWF도 약 13 ms에 97% 이상을 유지한다. 다수 사용자(9명)의 "
        "상호보완적 증거가 빠르게 명확한 log-belief margin을 만드는 것으로 해석된다.",
        "- **Zhou2016은 사용자 수가 4명으로 가장 적다.** 개별 사용자의 영향이 상대적으로 "
        "커서 첫 응답만 쓰는 방법과 다중 응답 융합 간 차이를 설명하기 좋은 데이터셋이다.",
        "- **Weibo2014가 가장 어려운 환경**이다. 클래스가 7개로 가장 많고 Full-Wait도 "
        "약 82% 수준이다. AWF가 다른 데이터셋보다 오래 기다리는 것은 고정 timer가 아니라 "
        "현재 결정 불확실성과 추가 사용자 증거의 가치를 반영한다는 근거가 된다.",
        "",
        "## 4. 네트워크 trace 특성",
        "",
        "| Access | 평균 지연(ms) | CV | p50 | p95 | p99 | p99/p50 | 손실률 | D=50 도착률 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| BLE | 6.96 | 0.42 | 6.93 | 11.48 | 11.90 | 1.72 | 0.20% | 1.00 |",
        "| Wi-Fi | 17.27 | 2.03 | 13.07 | 39.07 | 112.28 | 8.59 | 3.86% | 0.93 |",
        "| 5G | 38.53 | 0.40 | 36.00 | 45.50 | 105.55 | 2.93 | 0.68% | 0.96 |",
        "| 4G | 55.38 | 0.65 | 45.25 | 127.57 | 219.56 | 4.85 | 2.15% | 0.67 |",
        "| LEO | 31.57 | 0.23 | 30.73 | 42.47 | 51.31 | 1.67 | 0.35% | 0.98 |",
        "",
        "평균만 보면 Wi-Fi가 빠르지만 CV=2.03, p99/p50=8.59로 꼬리가 매우 길다. "
        "4G도 D=50 ms 도착률이 0.67에 불과해 deadline이 실질적으로 작동한다. "
        "LEO는 중앙부 분산은 작지만 드문 spike가 존재한다. 따라서 평균 지연 기반 정책보다 "
        "시점별 조건부 도착확률과 잔여지연을 사용하는 것이 타당하다.",
        "",
        "## 5. 기존 robustness 결과 종합",
        "",
        "### 5.1 Trace 내부 비정상성",
        "",
        "| Access | split-half KS | Wasserstein(ms) | 중앙값 변화 95% CI | 해석 |",
        "|---|---:|---:|---:|---|",
        "| BLE | 0.012 | 0.05 | [-0.01, +0.21] | 안정적 |",
        "| Wi-Fi | 0.003 | 0.29 | [-0.16, +0.14] | 안정적이나 heavy tail |",
        "| 5G | 0.051 | 0.88 | [-0.51, +0.50] | 대체로 안정적 |",
        "| 4G | 0.161 | 6.28 | [+0.50, +10.00] | 유의한 drift |",
        "| LEO | 0.184 | 1.43 | [-1.69, -1.28] | 유의한 median shift |",
        "",
        "대부분의 trace는 세션 전·후반 중앙값이 안정적이지만 4G와 LEO에는 측정 가능한 "
        "변화가 있다. 따라서 stationary empirical belief는 합리적인 1차 모델이지만 모든 "
        "tier에 완전히 성립하는 가정은 아니다.",
        "",
        "### 5.2 도착 모델 mismatch와 session drift (D=50 ms, 8 seeds)",
        "",
        "| 조건 | AWF 정확도 | Full-Wait | 차이 | AWF 지연 |",
        "|---|---:|---:|---:|---:|",
        "| nominal scale 1.0 | 90.48% | 92.22% | -1.74%p | 21.22 ms |",
        "| 실제 지연 ×1.3 | 88.15% | 89.47% | -1.32%p | 26.17 ms |",
        "| 실제 지연 ×1.5 | 82.49% | 83.10% | -0.61%p | 28.39 ms |",
        "| session 끝 지연 +50% | 87.87% | 89.25% | -1.38%p | 25.26 ms |",
        "| session 끝 지연 +100% | 83.87% | 84.65% | -0.78%p | 27.75 ms |",
        "",
        "환경이 악화될수록 AWF와 Full-Wait가 함께 하락하므로 절대 정확도 하락의 상당 "
        "부분은 정책 오류보다 deadline 내 이용 가능한 응답 감소에서 발생한다. 시험 범위에서 "
        "AWF–Full-Wait 차이는 1.9%p 이내였다.",
        "",
        "### 5.3 미모델링 multi-hop transit (D=50 ms, 8 seeds)",
        "",
        "| 추가 hop | 실제 p50/p99 | AWF 정확도/지연 | Full-Wait 정확도 | 차이 |",
        "|---:|---:|---:|---:|---:|",
        "| 0 | 27/124 ms | 90.48% / 21.22 ms | 92.22% | -1.73%p |",
        "| 1 | 34/156 ms | 89.33% / 27.57 ms | 90.68% | -1.35%p |",
        "| 2 | 42/175 ms | 86.29% / 33.02 ms | 87.17% | -0.88%p |",
        "| 3 | 49/191 ms | 81.34% / 37.65 ms | 82.10% | -0.76%p |",
        "| 4 | 57/209 ms | 75.31% / 41.44 ms | 75.75% | -0.44%p |",
        "",
        "AWF가 access hop만 믿고 실제 경로에는 추가 heavy-tailed hop이 있어도 Full-Wait와의 "
        "차이는 오히려 축소됐다. 다만 이는 절대 성능이 유지됐다는 뜻이 아니라 두 방법이 "
        "함께 저하되는 graceful degradation 결과다.",
        "",
        "### 5.4 Held-out trace와 cross-source transfer",
        "",
        "Held-out 평가는 belief 추정용 sequence와 arrival replay용 sequence를 분리한다. "
        "D=50에서 AWF-Nominal은 90.21%/23.7 ms, test 분포를 아는 AWF-Oracle은 "
        "90.16%/23.5 ms였다. D=100에서도 정확도 차이는 0.12%p, 지연 차이는 약 "
        "0.1 ms였다. value stop 비율은 각각 73%와 82%이므로 단순 deadline 대기 결과가 아니다.",
        "",
        "Cross-source에서는 5G Static↔Driving(KS=0.408)과 LEO 도시 간(KS=0.151)으로 "
        "belief를 이전했다. D=100에서 Transfer–Oracle 정확도 차이는 모든 방향에서 "
        "±0.11%p 이내이고 최종 결정 일치율은 99.8% 이상이었다. 이는 같은 source 내부 "
        "분할보다 강한 검증이지만, 여전히 동일 공개 measurement campaign 범위의 이전이다.",
        "",
        "### 5.5 실제 배치 archetype별 robustness (D=50 ms, 15 seeds)",
        "",
        "| 배치 | AWF 정확도/지연 | Full-Wait | 정확도 차이 | 지연 감소 |",
        "|---|---:|---:|---:|---:|",
        "| Co-located clinic | 92.26% / 11.0 ms | 93.22% | -0.96%p | 78% |",
        "| Metro edge | 91.40% / 22.7 ms | 93.00% | -1.60%p | 55% |",
        "| Wide-area mobile | 89.16% / 40.4 ms | 90.08% | -0.91%p | 19% |",
        "| Non-terrestrial field | 91.00% / 33.8 ms | 92.45% | -1.45%p | 32% |",
        "| Heterogeneous | 90.57% / 23.1 ms | 92.37% | -1.80%p | 54% |",
        "",
        "빠른 clinic 환경에서는 지연 절감이 크고, 5G/4G 중심 wide-area에서는 deadline "
        "가까이까지 기다려 지연 절감이 작다. 즉 AWF가 deployment 난이도에 따라 동작점을 "
        "바꾸는 패턴이 확인된다.",
        "",
        "## 6. Trade-off와 민감도 분석",
        "",
        "- **λ 민감도:** λ=10⁻⁵에서 92.23%/29.69 ms, 10⁻³에서 "
        "90.57%/23.09 ms, 10⁻²에서 75.76%/11.09 ms다. λ가 실제 정확도–지연 "
        "trade-off를 제어한다.",
        "- **α 민감도:** 10⁻⁵–10⁻²에서 90.57%/23.09 ms로 동일하고, 10⁻¹에서도 "
        "90.47%/22.68 ms다. 기본 λ에서는 margin stop이 주 결정 요인이 아님을 뜻한다.",
        "- **Deadline:** D=10→100 ms에서 Full-Wait 정확도는 62.7→93.1%로 증가하지만 "
        "AWF 지연은 D=50 이후 약 23 ms에서 포화한다. AWF가 deadline 자체를 고정 timer로 "
        "사용하지 않는다는 근거다.",
        "- **Frontier:** 기존 sweep에서 AWF는 약 7–30 ms 범위를 구성하며, 30 ms 부근에서 "
        "92.2%에 도달했다. Anytime-cBCI는 46 ms까지 기다려도 89.6%, Entropy-Gain은 "
        "27 ms에서 84.7%에 포화했다.",
        "- **Pooling ablation:** 동일한 낮은 λ 동작점에서 Full-Wait log/linear는 "
        "92.4/90.7%, AWF log/linear는 92.2/90.6%였다. stopping 효과와 pooling 효과를 "
        "분리해서 해석할 수 있다.",
        "",
        "## 7. 결과 파일 정합성 점검",
        "",
        "기존 산출물 사이에 생성 시점이 다른 흔적이 있다. 예를 들어 현재 "
        "table-main-d50.csv의 Full-Wait–AWF 평균 차이는 1.80%p인데, paired_stats.txt는 "
        "0.144%p를 기록한다. main-numbers.txt의 DA-OS는 자연 β에서 Full-Wait와 같은 "
        "동작점이지만 논문 표는 별도 daos-point.txt에서 선택한 β=0.15 결과를 사용한다. "
        "따라서 최종 보고에서는 요약 txt를 그대로 혼합하지 말고, 동일 코드·동일 timestamp로 "
        "재생성한 raw CSV 하나를 source of truth로 삼아야 한다.",
        "",
        "## 8. 교수님께 보고할 핵심 결론",
        "",
        "1. 세 EEG 데이터셋은 사용자 수 4–10명, 클래스 수 3–7개로 난이도와 위원회 "
        "규모가 달라 AWF의 적응적 대기 특성을 비교하기 적합하다.",
        "2. BNCI2014_001은 쉬운 결정, Weibo2014는 어려운 다중 클래스 결정, Zhou2016은 "
        "소규모 위원회라는 서로 다른 역할을 한다.",
        "3. 네트워크 trace는 access별 중앙값뿐 아니라 손실과 heavy tail이 크게 다르며, "
        "D=50 ms에서 도착률이 0.67–1.00으로 벌어져 stop-or-wait 문제가 비자명하다.",
        "4. 기존 결과에서 AWF는 데이터셋 난이도에 따라 약 13–31 ms로 대기 시간을 "
        "달리하면서 Full-Wait에 근접한 정확도를 보였다. 이는 고정 대기시간 정책과 다른 "
        "핵심 관찰이다.",
        "5. 다만 잔여지연 구현을 논문 수식에 맞게 수정했으므로, 최종 보고 숫자는 재실험과 "
        "통계 검정 후 확정해야 한다.",
        "",
        "### 해석 시 반드시 밝힐 설계 한계",
        "",
        "세 데이터셋은 본래 동시 다중 사용자 기록이 아니다. 본 실험의 collaborative trial은 "
        "동일한 정답 클래스를 가진 서로 다른 피험자의 실제 trial을 사용자별로 복원추출해 "
        "구성한 모사 위원회이다. 따라서 사용자별 EEG posterior는 실제 데이터에서 얻지만, "
        "사용자 간 동시 자극·공동 환경·오류 상관은 직접 관측한 것이 아니다. 결과는 "
        "네트워크 제약 하의 fusion/stop 정책 비교에는 적합하지만, 실제 hyperscanning cBCI의 "
        "사용자 간 상관구조를 완전히 재현한다고 주장해서는 안 된다.",
        "",
        "## 9. 최종 보고 전 필수 후속 작업",
        "",
        "- 수정된 조건부 잔여지연으로 D=50/100 핵심 실험 재실행",
        "- 데이터셋별 AWF–Full-Wait paired 차이의 95% CI와 Wilcoxon 검정 재계산",
        "- seed 간 변동과 subject 간 변동을 구분하여 표기",
        "- 새 CSV로 본 보고서와 main-en.tex 표·그림 동시 갱신",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    runner = load_runner()
    eeg = eeg_summary(runner)
    policy = policy_summary()
    eeg.to_csv(OUT / "dataset-eeg-summary.csv", index=False, encoding="utf-8-sig")
    policy.to_csv(OUT / "dataset-policy-summary.csv", index=False, encoding="utf-8-sig")
    report = build_report(eeg, policy)
    (OUT / "professor-dataset-report.md").write_text(report, encoding="utf-8")
    print(OUT / "professor-dataset-report.md")


if __name__ == "__main__":
    main()
