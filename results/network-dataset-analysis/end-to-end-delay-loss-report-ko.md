# 네트워크 데이터셋의 End-to-End Delay 및 Loss 분석 보고

## 1. 분석 목적

본 실험은 Wi-Fi, 5G, 4G-LTE, LEO 환경에서 수집된 실제 네트워크 측정 데이터를 이용하여, 사용자 단말에서 측정 endpoint까지의 응답 도착 특성을 모델링한다. 단일 무선 홉의 지연을 가정하지 않고, 실제 패킷이 전체 측정 경로를 통과한 후 관측되는 end-to-end delay와 packet loss를 사용한다.

## 2. 데이터 구성

| 데이터 | End-to-end delay | Loss 산출 방법 | 현재 평균 loss |
|---|---|---|---:|
| Wi-Fi | IRTT로 측정한 실제 uplink one-way delay | 10 ms probe 주기의 timestamp 누락 | 0.188% |
| 5G | IRTT로 측정한 실제 uplink one-way delay | 10 ms probe 주기의 timestamp 누락 | 0.482% |
| 4G-LTE | 단말과 ping endpoint 사이의 `PINGAVG/2` | 원본 `PINGLOSS` 평균 | 2.150% |
| LEO | Starlink 경로에서 측정한 실제 uplink one-way delay | IRTT sequence 번호 누락 | 0.501% |

Wi-Fi, 5G, LEO는 송신 단말에서 측정 서버까지 관측된 실제 one-way delay를 사용한다. 4G-LTE는 one-way 측정값이 제공되지 않아 end-to-end RTT의 절반을 one-way delay로 근사한다. 따라서 4G-LTE 결과에는 상향·하향 경로가 대칭이라는 가정이 포함된다.

원본 형태도 서로 다르다. Wi-Fi와 5G는 패킷별 `send`, `receive`, `rtt`, `timestamp`가 저장된 Parquet이고, LEO는 패킷별 `seq`, `sd`, `rd`, `rtt`가 기록된 IRTT 텍스트 로그이다. 4G-LTE는 위치, 속도, 무선 품질, `PINGAVG`, `PINGLOSS`가 기록된 시간 구간별 CSV이다. 따라서 Wi-Fi·5G·LEO는 패킷 단위 one-way 측정이고, 4G-LTE는 구간별 RTT 통계를 이용한 근사이다.

## 3. Delay와 Loss의 결합

각 사용자 응답의 최종 도착시간은 다음과 같이 모델링한다.

\[
a_i =
\begin{cases}
\infty, & \text{확률 } p_{\mathrm{loss}},\\
T_i, & \text{확률 } 1-p_{\mathrm{loss}},
\end{cases}
\]

여기서 \(T_i\)는 해당 네트워크 데이터에서 얻은 end-to-end delay 표본이다. 손실된 응답은 도착시간을 무한대로 처리하므로 decision deadline 내에 사용할 수 없다.

AWF가 사용하는 조건부 도착확률은 다음과 같이 delay 분포와 loss를 함께 반영한다.

\[
\rho_j(t)=
\frac{(1-p_{\mathrm{loss}})\left[\hat F(D)-\hat F(t)\right]}
{p_{\mathrm{loss}}+(1-p_{\mathrm{loss}})\left[1-\hat F(t)\right]}.
\]

따라서 loss가 크거나 지연 분포상 deadline 이전 도착 가능성이 작으면 \(\rho_j(t)\)가 감소한다. AWF는 이러한 응답을 계속 기다리는 대신 현재 도착한 BCI 증거로 결정할 가능성이 커진다.

## 4. 해석

본 데이터는 단일 access hop의 고정 지연이 아니라, 단말과 측정 endpoint 사이에서 관측된 전체 경로의 도착 특성을 나타낸다. 따라서 access, backhaul, transit 및 위성 구간을 개별적으로 분해하거나 합성하지 않고, fusion node에서 최종적으로 관측되는 end-to-end arrival distribution으로 사용할 수 있다.

네 데이터는 서로 다른 특성을 갖는다.

- Wi-Fi는 매우 빠른 지연이 중심이지만 간헐적인 tail delay가 존재한다.
- 5G는 낮고 비교적 안정적인 지연 분포를 보인다.
- 4G-LTE는 지연과 손실이 가장 크고 tight deadline에서 미도착 가능성이 높다.
- LEO는 수십 ms 수준의 지연과 시간적으로 지속되는 링크 변화를 보인다.

이와 같은 이질성은 모든 응답을 고정 시간 동안 기다리는 방법보다, 현재 시점의 조건부 도착 가능성과 응답의 결정 가치를 함께 고려하는 stop-or-wait 정책을 평가하는 데 적합하다.

## 5. 현재 모델의 한계

현재 구현은 실측 end-to-end delay 분포와 평균 loss rate를 보존하지만, loss는 평균 확률을 이용한 독립 Bernoulli 과정으로 적용한다. 따라서 다음 항목은 직접 재현하지 않는다.

- 여러 패킷이 연속해서 손실되는 burst loss
- 지연이 증가할 때 손실도 함께 증가하는 delay-loss 상관
- 시간에 따라 변하는 조건부 loss probability

따라서 본 실험은 **end-to-end delay-loss marginal model**로 표현하는 것이 적절하며, 원본의 delay와 loss를 시간 순서대로 완전히 재생하는 joint trace replay로 표현해서는 안 된다.

## 6. 결론

본 실험은 Wi-Fi, 5G, 4G-LTE, LEO의 실제 측정 데이터에서 end-to-end delay 분포와 packet-loss probability를 추출하고, 손실 응답을 deadline 내 미도착으로 처리한다. AWF는 이 두 정보를 결합한 조건부 도착확률을 이용하므로 단일 홉이나 평균 지연만을 가정하지 않는다. 다만 현재 loss 모델은 평균 손실률을 독립적으로 적용하므로, 향후에는 burst loss와 delay-loss 상관을 보존하는 시간 순서 기반 joint replay를 추가할 필요가 있다.

## 7. BCI 데이터셋과 사용자별 학습

세 데이터셋은 모두 MOABB를 통해 불러온 다중 클래스 motor-imagery EEG이다. 각 피험자를 협력 BCI의 독립 사용자 한 명으로 취급한다.

| 데이터셋 | 사용자 | 클래스 | 클래스 구성 | 무작위 기준 |
|---|---:|---:|---|---:|
| Zhou2016 | 4명 | 3 | left hand, right hand, feet | 33.3% |
| BNCI2014_001 | 9명 | 4 | left hand, right hand, feet, tongue | 25.0% |
| Weibo2014 | 10명 | 7 | 손·발·복합동작·rest | 14.3% |

각 사용자 EEG는 8–30 Hz로 band-pass filtering한 후 CSP 6개 성분과 shrinkage LDA로 분류한다. 데이터는 클래스 비율을 유지하면서 약 56% training, 14% validation, 30% test로 분리한다.

- Training set: 사용자별 CSP와 LDA 학습
- Validation set: accuracy, macro-F1 및 class-wise confusion matrix 산출
- Test set: 각 trial의 posterior probability 벡터 생성

Validation confusion matrix는 사용자가 실제 클래스 \(c\)에서 클래스 \(z\)를 보고할 확률 \(P_i(z\mid c)\)로 정규화하여 AWF의 사용자별 reliability 모델로 사용한다. Test label이나 test 정확도는 reliability 추정에 사용하지 않고 최종 fusion 평가에만 사용한다.

## 8. BCI 사용자별 결과

아래 결과에서 validation 정확도는 AWF가 사전에 알 수 있는 calibration 성능이고, test 정확도는 fusion 이전 각 사용자의 held-out local classifier 성능이다.

| 데이터셋 | 사용자 | Test trials | Validation 정확도 | Validation macro-F1 | Test 정확도 |
|---|---|---:|---:|---:|---:|
| Zhou2016 | S01 | 144 | 74.6% | 74.1% | 77.1% |
| Zhou2016 | S02 | 131 | 82.0% | 81.9% | 78.6% |
| Zhou2016 | S03 | 136 | 82.5% | 82.7% | 74.3% |
| Zhou2016 | S04 | 131 | 75.4% | 76.0% | 85.5% |
| BNCI2014_001 | S01 | 173 | 84.0% | 83.5% | 70.5% |
| BNCI2014_001 | S02 | 173 | 56.8% | 55.9% | 56.6% |
| BNCI2014_001 | S03 | 173 | 79.0% | 78.8% | 82.7% |
| BNCI2014_001 | S04 | 173 | 54.3% | 54.0% | 58.4% |
| BNCI2014_001 | S05 | 173 | 39.5% | 39.5% | 38.7% |
| BNCI2014_001 | S06 | 173 | 42.0% | 41.5% | 51.4% |
| BNCI2014_001 | S07 | 173 | 80.2% | 80.3% | 74.0% |
| BNCI2014_001 | S08 | 173 | 79.0% | 78.1% | 80.3% |
| BNCI2014_001 | S09 | 173 | 65.4% | 65.5% | 69.4% |
| Weibo2014 | S01 | 168 | 34.2% | 35.1% | 35.7% |
| Weibo2014 | S02 | 168 | 40.5% | 36.0% | 31.0% |
| Weibo2014 | S03 | 168 | 34.2% | 32.9% | 31.0% |
| Weibo2014 | S04 | 168 | 25.3% | 22.2% | 26.8% |
| Weibo2014 | S05 | 168 | 27.8% | 24.4% | 36.3% |
| Weibo2014 | S06 | 150 | 34.3% | 29.6% | 28.0% |
| Weibo2014 | S07 | 168 | 30.4% | 26.8% | 33.3% |
| Weibo2014 | S08 | 168 | 54.4% | 54.3% | 63.1% |
| Weibo2014 | S09 | 168 | 58.2% | 56.0% | 58.3% |
| Weibo2014 | S10 | 168 | 38.0% | 34.4% | 36.3% |

데이터셋별 test 정확도 평균은 Zhou2016 약 78.9%, BNCI2014_001 약 64.7%, Weibo2014 약 38.0%이다. Zhou2016은 사용자 간 성능이 비교적 균일하다. BNCI2014_001은 약 38.7–82.7%로 사용자 편차가 커 reliability-aware fusion의 필요성을 보여준다. Weibo2014는 7-class 문제로 개별 정확도가 가장 낮고 사용자 간 편차도 크기 때문에, 추가 사용자 응답의 가치와 적응적 대기 효과를 평가하기에 가장 어려운 조건이다.

## 9. Collaborative BCI 평가 구성

세 데이터셋은 동시 다중 사용자 EEG 기록이 아니므로, 같은 정답 클래스를 가진 서로 다른 사용자의 test trial을 조합해 collaborative trial을 구성한다. 각 seed에서 공통 정답 클래스를 선택한 뒤 사용자별로 해당 클래스의 held-out test posterior를 복원추출한다. 기본 trial 수는 가장 작은 사용자 test set에 맞춰 Zhou2016 131개, BNCI2014_001 173개, Weibo2014 150개이다.

각 사용자는 seed마다 Wi-Fi, 5G, 4G-LTE 또는 LEO end-to-end 경로에 고정 배정된다. 사용자별 test posterior는 해당 경로의 delay와 loss를 거쳐 fusion node에 비동기적으로 도착한다. AWF는 현재 도착한 posterior, validation confusion matrix, 경로별 조건부 도착확률을 함께 이용해 stop 또는 wait를 결정한다.

이 평가는 실제 사용자별 EEG classifier 출력과 실제 네트워크 trace를 사용하지만, 동일 자극을 동시에 수행한 hyperscanning 자료는 아니다. 따라서 사용자 간 동시 오류 상관이나 공동 환경 효과를 직접 측정한 실험으로 해석해서는 안 된다.
