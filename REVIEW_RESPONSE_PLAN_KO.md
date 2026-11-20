# IoTJ 리비전 대응 계획서

## 1. 기본 대응 원칙

리뷰 답변은 모든 항목을 다음 순서로 작성한다.

1. 리뷰어의 지적에 감사하고 핵심 문제를 한 문장으로 다시 확인한다.
2. 동의하는 부분과 오해를 바로잡아야 하는 부분을 구분한다.
3. 실제로 수행한 코드 수정, 추가 실험, 원고 수정을 구체적으로 설명한다.
4. 가능하면 수정 전후 수치나 새 실험 결과를 제시한다.
5. 마지막에 원고의 반영 위치를 `Page X, Lines Y--Z` 형식으로 표시한다.

이번 리비전에서는 모든 결과가 좋은 것처럼 방어하지 않는다. 다음 두
결과는 명확히 인정하고 논문의 주장을 조정한다.

- AWF와 Confidence-Stop의 정확도 차이는 세 데이터셋에서 통계적으로
  유의하지 않다.
- margin 제거 실험은 AWF의 정확도와 지연을 변화시키지 않았다.

따라서 핵심 주장은 “AWF가 모든 방법보다 정확하다”가 아니라 다음과
같이 변경한다.

> AWF는 사용자별 판단 기여도와 제한시간 내 도착 가능성을 함께
> 고려하여 Full-Wait에 가까운 정확도를 유지하면서 불필요한 대기를
> 줄인다. 특히 단순 도착률·응답 수·entropy 기반 정책보다 안정적인
> 정확도--지연 절충을 제공하며, Confidence-Stop과 비교해서는 일부
> 데이터셋에서 비슷한 정확도로 더 짧은 지연을 보인다.

---

## 2. Associate Editor 대응

### 지적 요약

실험 신뢰성, baseline 비교의 공정성, 실제 적용 가능성을 종합적으로
보완해야 한다.

### 대응 방향

단순한 설명 추가가 아니라 전체 실험 정의를 다시 점검하고 재실험했다고
답변한다.

### 반영 내용

- 4G-LTE 표본 정의 수정 및 trace cache 재생성
- 사용자 posterior calibration 추가
- 모든 정책의 parameter를 validation에서 선택
- First-k와 Confidence-Stop 전체 sweep 추가
- held-out trace, temporal replay, correlated arrival 실험 추가
- homogeneous access, correlated BCI error 실험 추가
- Zhou2016 exact discretized DP 비교 추가
- seed-level paired 통계검정과 runtime 분석 추가
- trace-driven 평가와 실제 cBCI 적용의 한계 확대

### 답변 핵심 문장

> Following the Associate Editor's recommendation, we conducted a clean
> re-evaluation rather than only revising the presentation. All primary
> results in the revised manuscript use the corrected LTE-only trace bank,
> validation-calibrated posteriors, validation-selected policy parameters,
> and held-out test trials.

---

## 3. Reviewer 1 대응

### 3.1 사용자 신뢰도의 시간 변화

#### 대응

현재 실험은 validation에서 confusion matrix를 한 번 계산하므로 장기적인
피로, 학습효과, 세션 변화까지 자동 추적하지 못한다고 인정한다.

#### 원고 수정

- limitation에 offline reliability model의 한계 추가
- future work에 exponentially weighted confusion matrix와 online calibration
  추가
- 정답 feedback이 없을 때에는 high-confidence pseudo-label 또는 scalar
  accuracy fallback이 필요하다고 설명

#### 과도하게 주장하지 않을 부분

현재 코드가 실제 non-stationary EEG를 해결했다고 주장하지 않는다.

### 3.2 one-step approximation과 optimality gap

#### 대응

Zhou2016은 사용자 4명, 클래스 3개이므로 discretized finite-horizon DP를
계산했다고 답변한다. DP는 미래 정답이나 실제 도착시간을 미리 보지 않고,
과거 trace 분포와 confusion matrix만 사용한다.

#### 결과

- AWF objective: 0.06042
- 5-ms DP objective: 0.05758
- 평균 gap: 0.00284
- 첫 3개 seed에서 2.5/5/10 ms grid의 정확도 동일

#### 원고 수정

- exact Bellman formulation과 AWF one-step approximation을 분리
- DP는 discretized model에서만 exact하다고 명시
- AWF를 전역 최적정책이라고 표현하지 않음

### 3.3 BCI 결과와 네트워크 상태의 독립성

#### 대응

현재 공개 데이터만으로 두 변수의 실제 결합분포를 알 수 없다고 인정한다.
추가로 cross-user error correlation을 0, 0.2, 0.5로 변화시킨 robustness
실험을 제시한다.

#### 원고 수정

- 사용자 오류와 네트워크 상태가 기기 성능, 배터리, 연산시간 때문에
  연관될 수 있음을 limitation에 추가
- 현재 실험은 이를 직접 측정한 것이 아니라 sensitivity analysis임을 명시

### 3.4 계산 복잡도

#### 대응

- worst case 시간복잡도: `O(N^2 C^2)`
- empirical residual delay 계산은 prefix sum 적용 후 `O(log S)`
- 메모리: `O(NC^2 + NS)`

#### 측정 결과

최대 실험 규모 `N=10, C=7, S=40,000`에서 다음과 같다.

- median runtime: 약 3.28 ms
- p95 runtime: 약 7.36 ms
- policy state: 약 6.4 MB

Python 구현 기준이며 모든 하드웨어에서 동일하다고 주장하지 않는다.

### 3.5 통계검정

#### 대응

15개 seed를 기준으로 paired Wilcoxon test를 추가한다. Trial-level exact
McNemar test는 보조 결과로 제시한다.

#### 중요한 결과

- AWF 대 Confidence-Stop 정확도: 세 데이터셋 모두 유의한 차이 없음
- AWF 지연: BNCI와 Weibo에서는 유의하게 짧음
- Zhou에서는 Confidence-Stop과 지연 차이도 유의하지 않음

### 3.6 future work 확대

online reliability, semantic communication, edge scheduling, adaptive user
selection, fusion-aware resource allocation을 추가한다.

---

## 4. Reviewer 2 대응

### 4.1 confusion matrix와 posterior fusion의 관계

#### 설명 방식

- 이미 도착한 응답: 전체 posterior vector를 log-pooling에 사용
- 아직 도착하지 않은 응답: posterior 내용을 모르므로 validation confusion
  matrix로 가능한 hard prediction을 적분하여 기대 risk 감소량을 계산

즉 confusion matrix는 “기다릴 가치” 계산에만 사용하고, posterior는 현재
그룹 판단을 만드는 데 사용한다고 명확히 구분한다.

### 4.2 4G RTT와 OWD 혼용

#### 대응

UCC의 `PINGAVG`는 RTT이고 실제 코드는 `PINGAVG/2`를 one-way proxy로
사용했다고 명시한다. 다만 기존 코드는 LTE뿐 아니라 HSPA 계열도 함께
포함했으므로 이를 수정하고 모든 결과를 다시 계산했다.

#### 수정된 통계

- `NetworkMode == LTE`만 사용
- 표본 수: 663
- one-way proxy p50/p95/p99: 42.0/53.0/162.2 ms
- 평균 loss: 1.16%
- 50 ms 유효 도착률: 88.85%

기존 67% 문장은 삭제한다.

### 4.3 baseline parameter 선택

모든 parameter를 collaborative validation trial에서 동일한 목적함수로
선택했다고 답변한다.

\[
(1-\mathrm{accuracy})+10^{-3}\times\mathrm{latency}_{\mathrm{ms}}.
\]

Test 결과를 확인한 뒤 parameter를 고르지 않았음을 강조한다.

### 4.4 실제 적용 가능성과 trace-driven 한계

UCC가 packet-level uplink trace가 아니라 1-Hz application KPI라는 점,
RTT/2가 path asymmetry를 반영하지 못한다는 점, 실제 protocol queue와
retransmission을 완전히 재현하지 못한다는 점을 명시한다.

---

## 5. Reviewer 3 대응

### 5.1 RTT/OWD 문제

Reviewer 2의 4G 대응과 동일하게 답변한다. Reviewer가 지적한 “45.2 ms를
RTT 그대로 사용했다”는 부분에는, 코드는 이미 `/2`를 적용했으나 논문의
설명이 불명확했고 HSPA 표본까지 포함한 실제 문제가 있었다고 정리한다.

방어적으로 “리뷰어가 틀렸다”고 쓰지 않고 다음과 같이 표현한다.

> We apologize that the submitted description did not make the conversion and
> source filtering sufficiently clear. We have now restricted the source to
> LTE rows, explicitly documented the RTT/2 approximation, and rerun the
> complete evaluation.

### 5.2 arrival statistics가 oracle이라는 문제

#### 추가 실험

- 각 access trace의 앞 절반: delay CDF와 loss belief 추정
- 뒤 절반: 실제 도착과 loss를 순서대로 replay
- AWF-heldout과 test 분포를 아는 AWF-oracle 비교

#### 결과

평균 objective 차이의 절댓값은 데이터셋과 replay 조건에 따라 약
0.0002--0.0015이다.

이 결과가 모든 실제 환경의 distribution shift를 보장하지는 않는다고
함께 적는다.

### 5.3 IID sampling과 temporal structure

연속 test 구간을 순서대로 replay한다. 추가로 같은 access를 사용하는
사용자가 동일 spike와 loss event를 겪는 shared-link stress를 평가한다.

### 5.4 optimality gap

Reviewer 1의 DP 대응과 동일하다.

### 5.5 50 ms deadline의 현실성

#### 대응 원칙

리뷰어의 지적에 동의한다. 수 초 길이 MI acquisition 전체에서 20--30 ms는
1% 미만이다.

#### 주장 수정

- AWF가 EEG acquisition을 단축한다고 주장하지 않음
- local posterior가 생성된 이후의 communication/fusion budget만 다룸
- 20/50/100/200 ms 결과를 모두 제시
- 실제 허용 deadline은 application feedback loop에 따라 달라짐

실제 end-to-end cBCI 실험은 future work로 둔다.

### 5.6 posterior calibration

사용자별 temperature를 validation NLL로 학습하고 test에 고정 적용한다.
Accuracy는 유지하면서 NLL/ECE가 다음과 같이 개선되었다.

| 데이터셋 | NLL | ECE |
|---|---:|---:|
| BNCI | 0.7961 → 0.7890 | 0.0774 → 0.0649 |
| Zhou | 0.5611 → 0.5493 | 0.0766 → 0.0710 |
| Weibo | 1.4769 → 1.4492 | 0.0919 → 0.0811 |

Per-user accuracy, temperature, NLL, Brier score, ECE도 supplement 또는 표에
제시한다.

동일 EEG trial, 사용자--망 배정, 실제 delay/loss, AWF \(\lambda\)를 고정한
paired 실험에서 raw posterior를 temperature-scaled posterior로 바꾸면 다음과
같다.

| 데이터셋 | AWF 정확도 | 평균 지연 | margin stop 비율 |
|---|---:|---:|---:|
| BNCI | 99.27% → 99.50% | 8.98 → 11.02 ms | 35.5% → 29.9% |
| Zhou | 95.27% → 96.44% | 22.45 → 24.80 ms | 25.8% → 13.5% |
| Weibo | 81.82% → 82.04% | 26.96 → 31.30 ms | 10.6% → 9.7% |

즉 calibration은 과도한 margin 조기정지를 줄이고 AWF를 더 보수적으로
기다리게 만든다. 정확도 증가는 Zhou에서 가장 크지만, 모든 데이터셋에서
유의하게 향상된다고 주장하지 않는다. 세 데이터셋에 Holm 보정을 적용하면
정확도 변화는 Zhou만 유의하다(`p=0.0062`; BNCI `p=0.0868`, Weibo
`p=0.6779`). 지연 증가는 세 데이터셋 모두 유의하다(보정 `p=0.000183`).

### 5.7 사용자 독립성이 데이터 구성에서 만들어진 문제

서로 다른 피험자의 class-matched trial을 독립적으로 조합했으므로 실제
동시 자극에서 나타나는 공통 오류가 없다는 점을 명시한다. Correlated-error
실험은 sensitivity analysis일 뿐 실제 동시 cBCI 데이터의 대체물이 아니라고
설명한다.

### 5.8 통계적 유의성

Reviewer 1의 통계검정 대응과 동일하다.

---

## 6. Reviewer 4 대응

### 6.1 명확한 최적화 문제와 제약식

다음 항목을 수식으로 명확히 제시한다.

- 목적함수: expected classification loss + latency cost 최소화
- hard deadline `T <= D`
- stopping time은 현재까지 관찰한 정보에만 의존
- 정지 이후 판단은 되돌릴 수 없음
- 정지시각까지 도착한 응답만 사용
- deadline에서는 반드시 결정

Exact Bellman recursion과 AWF surrogate를 별도 subsection으로 구분한다.

### 6.2 비교 방법 부족

Full-Wait, Confidence-Stop, First-k, Anytime-cBCI, Arrival-Only,
Throughput-Stop, Entropy-Gain, conventional fusion, DP reference까지 포함한다.

### 6.3 ablation 부족

다음 항목을 추가한다.

- without margin
- without arrival probability
- without delay cost
- without reliability
- scalar accuracy instead of confusion matrix

---

## 7. Reviewer 5 대응

### 7.1 baseline limiting behavior

코드와 deterministic unit test를 함께 점검했다고 답변한다.

- Confidence threshold 1: deadline까지 대기
- First-k에서 전체 위원회 사용: 최종 fused prediction과 동일
- Arrival-Only threshold 0: 도착 가능성이 0보다 큰 응답이 있을 때만 대기
- cost 기반 정책: 0 endpoint 포함

Frontier와 main table의 method 정의도 통일한다.

### 7.2 Confidence-Stop 및 First-k sweep

- Confidence-Stop: `theta=0.5,...,0.999,1.0`
- First-k: 각 데이터셋의 `k=1,...,N`

세 데이터셋 평균 frontier에는 모든 데이터셋에 공통으로 존재하는
`k=1,...,4`만 표시하고, 전체 k 결과는 CSV에 보존한다고 설명한다.

### 7.3 held-out network statistics

Reviewer 3의 held-out/replay 대응과 동일하다.

### 7.4 homogeneous access

모든 사용자를 Wi-Fi, 5G, LTE, LEO에 각각 배정한 실험을 추가한다.
AWF 정확도는 Full-Wait에 가깝고, 지연은 access 특성에 따라 Wi-Fi/5G의
수 ms부터 LTE/LEO의 약 32--45 ms까지 변화한다고 보고한다.

### 7.5 보안과 개인정보

Posterior도 사용자의 의도를 드러낼 수 있는 민감정보라고 인정한다.

원고에 다음 내용을 추가한다.

- mutual authentication
- replay-protected authenticated encryption
- key management와 access control
- 짧은 retention period
- metadata leakage와 compromised endpoint는 암호화만으로 해결되지 않음
- 현재 trace 실험에는 암호화 overhead가 포함되지 않음

### 7.6 practical latency

20/50/100/200 ms에서 모든 정책과 95% seed-level band를 제시한다.

### 7.7 margin의 실제 기여

리뷰어 지적에 동의한다.

- margin trigger 비율: 데이터셋에 따라 약 9.7--29.9%
- 그러나 no-margin과 정확도·지연이 완전히 동일
- margin이 작동한 시점에 value rule도 동일하게 stop했음

따라서 margin을 독립적인 성능 향상 요소로 주장하지 않고, calibrated
belief가 충분히 한 클래스에 집중되었음을 확인하는 보수적인 sufficient
stopping certificate로 설명한다. 이때

\[
L_{(1)}-L_{(2)}\geq \log((C-1)/\alpha)
\]

이면 선두 클래스의 normalized belief가 최소 \(1/(1+\alpha)\)임을 보일 수
있다. 기본값 \(\alpha=10^{-3}\)에서는 약 0.999이다. 다만 이는 정답을
보장한다는 의미가 아니라, calibration된 현재 belief가 매우 집중되어
있다는 sufficient condition이다.

Margin은 AWF에 유지하되 waiting-value rule보다 앞에서 검사하는 부가적인
fast-stop condition으로 위치를 명확히 한다. 평균 trigger 비율은 BNCI
29.9%, Zhou 13.5%, Weibo 9.7%이지만, 해당 시점에서 waiting-value rule도
동일하게 정지하므로 no-margin과 seed별 정확도 및 평균 지연이 같았다는
사실을 함께 보고한다.

### 7.8 Fig. 1 classifier 불일치

Neural network 그림을 실제 pipeline인 `8--30 Hz → CSP(6) → shrinkage LDA`
로 교체한다.

### 7.9 Fig. 2 index 오류

Belief와 confusion matrix의 class index를 `1,...,C`로 수정한다. 사용자
index `1,...,N`은 사용자별 posterior/profile/confusion matrix에만 사용한다.

### 7.10 Fig. 4 baseline 및 band 누락

8개 주요 정책과 95% seed-level variability band가 포함된 새 deadline
그림으로 교체한다.

---

## 8. Reviewer 6 대응

Reviewer 6은 Reject 의견이므로 방어적인 표현보다 문제 범위와 독창성을
다시 정의하는 것이 중요하다.

### 8.1 실제로 필요한 문제인지 불명확

수 초 EEG acquisition 자체를 줄이는 문제가 아니라, local inference 이후
edge feedback path에서 발생하는 post-inference communication/fusion 문제라고
범위를 제한한다. 반복적 또는 closed-loop collaborative inference에서의
의미를 설명하되, live application 검증이 아직 없다는 점도 인정한다.

### 8.2 일반 분산시스템과 비교한 독창성 부족

“분산 early stopping 자체가 최초”라고 주장하지 않는다. AWF의 차별점은
다음 조합이라고 재정의한다.

- missing BCI response의 사용자별·클래스별 예상 decision-risk 감소
- 현재까지 미도착했다는 조건을 반영한 deadline arrival probability
- hard deadline 내 residual waiting cost
- 하나의 group decision을 위한 collective stop-or-wait

Arrival/throughput/entropy/confidence 정책과의 비교를 통해 이 차이를 보인다.

### 8.3 log-pooling과 calibration

Temperature scaling과 NLL/Brier/ECE 결과로 대응한다. Log-pooling이
overconfidence를 증폭할 수 있다는 지적에 동의하고 raw/calibrated 결과를
구분한다.

### 8.4 사용자 간 상호의존성

현재 trial 구성은 독립성을 만든다고 명시하고 correlated-error와 shared-link
실험을 제시한다. 실제 simultaneous multi-user dataset이 필요하다는 점을
future work에 둔다.

### 8.5 사용자별 waiting time과 collective waiting의 관계

각 missing user마다 `V_j(t)`를 계산하지만 실제 행동은 하나의 collective
action이라고 설명한다.

- `max_j V_j(t)>0`: 그룹 전체가 계속 대기
- 그렇지 않으면: 한 번의 group decision으로 종료

사용자마다 별도의 최종 결정시각을 정하는 방식이 아니다. Exact DP와의
차이는 여러 미래 응답의 joint value를 `max` 기반 one-step 값으로 근사하는
부분이라고 명확히 설명한다.

---

## 9. 답변서와 논문 수정 순서

1. 본 대응계획을 교수님과 확인한다.
2. 각 리뷰 항목별로 논문에 들어갈 문장·수식·표·그림을 확정한다.
3. LaTeX의 기존 수치와 주장을 revision 결과로 전부 교체한다.
4. 영문 response letter에 실제 page/line 번호를 삽입한다.
5. 논문, response letter, CSV 사이의 모든 숫자를 다시 대조한다.
6. 최종 PDF에서 표·그림·수식·reference를 시각 검수한다.

## 10. 교수님께 먼저 확인받아야 할 판단 사항

- Margin 유지 결정 완료: 독립적인 성능기여가 아니라 보수적인 sufficient
  stopping certificate로 설명
- 50 ms를 대표 deadline으로 계속 사용할지, 20--200 ms 결과를 main으로
  확장할지
- Calibration 결과를 본문 표로 둘지 supplement로 둘지
- DP/held-out/homogeneous/correlation 결과 중 본문과 supplement 배치
- UCC LTE source의 한계를 감수하고 유지할지, 추가 LTE source를 더 찾을지
- Reviewer 6 대응을 위해 application use case를 어느 수준까지 구체화할지
