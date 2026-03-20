---
title: "[논문 리뷰] HiPPO: 장거리 의존성 문제의 수학적 해결 — SSM과 Mamba의 이론적 기반"
date: 2026-03-19
tags: ["논문리뷰", "HiPPO", "SSM", "State Space Model"]
categories: ["ML/AI"]
summary: "HiPPO 논문 리뷰. 연속 시간 입력을 직교 다항식 기저로 최적 압축하는 이론을 제시하며, S4와 Mamba로 이어지는 State Space Model의 수학적 기반을 확립한다."
math: true
toc: true
draft: false
---

## 논문 정보

- **제목**: HiPPO: Recurrent Memory with Optimal Polynomial Projections
- **저자**: Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, Christopher Ré
- **발표**: NeurIPS 2020 (arXiv: 2008.07669)
- **키워드**: Long-Range Dependencies, Orthogonal Polynomials, Recurrence, Online Function Approximation

---

## 1. 문제: 시퀀스 모델의 장거리 기억

RNN/LSTM의 근본적 한계는 **장거리 의존성**이다. 시퀀스가 길어지면:

- LSTM의 게이팅이 정보를 점진적으로 소실
- Truncated BPTT가 그래디언트 전파를 제한
- 실질적 기억 범위가 수백 스텝에 한정

Transformer는 어텐션으로 이를 해결했지만 $O(L^2)$ 비용을 지불한다.

> **HiPPO의 질문**: 고정 크기 상태 벡터로 과거 입력을 **최적으로** 기억하는 수학적 방법은 무엇인가?

---

## 2. 핵심 아이디어: 함수 근사로서의 기억

### 기억 = 온라인 함수 근사

시점 $t$까지의 입력 $f(\tau), \tau \leq t$를 기억한다는 것은, 이 함수를 **유한 차원 벡터** $c(t) \in \mathbb{R}^N$으로 근사하는 것과 같다.

직교 다항식 기저 $\{g_n\}_{n=0}^{N-1}$에 대해:

$$f(\tau) \approx f^{(t)}(\tau) = \sum_{n=0}^{N-1} c_n(t) \cdot g_n(\tau)$$

여기서 $c_n(t)$는 **계수** — 이것이 상태 벡터의 각 요소이다.

### 측도(Measure)의 역할

어떤 과거를 **얼마나 중요하게** 기억할 것인가? 이것을 **측도** $\mu^{(t)}$로 정의한다.

$$c_n(t) = \int f(\tau) \cdot g_n(\tau) \, d\mu^{(t)}(\tau)$$

측도 선택에 따라 다른 기억 전략이 나온다:

| 측도 | 이름 | 기억 전략 |
|------|------|----------|
| $\mu^{(t)} = \frac{1}{t} \mathbb{1}_{[0,t]}$ | **HiPPO-LegS** | 과거 전체를 균등하게 기억 (Scaled Legendre) |
| $\mu^{(t)} = \mathbb{1}_{[t-\theta, t]}$ | **HiPPO-LegT** | 최근 $\theta$ 구간만 기억 (Translated Legendre) |
| $\mu^{(t)} = e^{-(t-\tau)} \mathbb{1}_{[0,t]}$ | **HiPPO-LagT** | 최근 입력에 지수적 가중 (Translated Laguerre) |

---

## 3. 수학적 유도: ODE로의 변환

### 핵심 정리

HiPPO의 가장 중요한 결과: 최적 계수 $c(t)$의 **시간 변화**가 선형 ODE로 표현된다.

$$\frac{dc}{dt} = A(t) \cdot c(t) + B(t) \cdot f(t)$$

여기서 $A$와 $B$는 **측도와 직교 다항식에 의해 완전히 결정**되는 행렬이다.

### HiPPO-LegS의 $A$ 행렬

과거 전체를 균등하게 기억하는 LegS의 경우:

$$A_{nk} = -\begin{cases} (2n+1)^{1/2}(2k+1)^{1/2} & \text{if } n > k \\ n+1 & \text{if } n = k \\ 0 & \text{if } n < k \end{cases}$$

$$B_n = (2n+1)^{1/2}$$

이 행렬은 **하삼각 + 대각** 구조를 가진다. 이것이 나중에 S4에서 **구조화된 상태 공간 모델**의 핵심이 된다.

### HiPPO-LagT의 경우

지수적 감쇠 가중의 LagT:

$$A_{nk} = -\begin{cases} 1 & \text{if } n \geq k \\ 0 & \text{if } n < k \end{cases}$$

$$B_n = 1$$

이것은 사실 **LSTM의 게이팅 메커니즘과 유사**하다 — 지수적 감쇠로 과거를 잊는다.

---

## 4. 이산화: 연속 → 이산 시퀀스

실제 시퀀스 데이터에 적용하려면 ODE를 이산화해야 한다.

$$c_{k+1} = \bar{A} \cdot c_k + \bar{B} \cdot f_k$$

이산화 방법에 따라:

| 방법 | $\bar{A}$ | 특성 |
|------|----------|------|
| **Euler** | $I + \Delta \cdot A$ | 단순하지만 불안정 |
| **Bilinear** | $(I - \frac{\Delta}{2}A)^{-1}(I + \frac{\Delta}{2}A)$ | 안정적, S4에서 채택 |
| **ZOH** | $\exp(\Delta \cdot A)$ | 정확, Mamba에서 채택 |

여기서 $\Delta$는 **스텝 크기** — Mamba에서 입력 의존적으로 만든 바로 그 파라미터이다.

---

## 5. HiPPO에서 S4, Mamba로의 연결

### HiPPO → S4 (2021)

S4의 핵심 기여: HiPPO의 $A$ 행렬을 **효율적으로 계산**하는 방법 발견

- HiPPO의 $A$는 밀집 행렬 → 직접 계산 시 $O(N^2L)$
- S4: $A$를 **NPLR (Normal Plus Low-Rank)** 형태로 분해

  $$A = V \Lambda V^{\ast} - PQ^{\ast}$$

- 이를 통해 **글로벌 컨볼루션 커널**로 변환 → $O(L \log L)$

### S4 → Mamba (2023)

Mamba의 핵심: HiPPO의 고정 $A$, $B$를 **입력 의존적**으로 확장

$$B(t) = \text{Linear}(x(t)), \quad \Delta(t) = \text{softplus}(\text{Linear}(x(t)))$$

| | HiPPO/S4 | Mamba |
|--|---------|-------|
| $A$ | 고정 (HiPPO 행렬) | 고정 (대각화) |
| $B$ | 고정 | **입력 의존적** |
| $\Delta$ | 고정 | **입력 의존적** |
| 시간 변동성 | LTI (Linear Time-Invariant) | **LTV (Linear Time-Varying)** |
| 컨볼루션 | 가능 (효율적) | 불가 → Parallel Scan |

Mamba에서 $A$가 대각 행렬로 단순화된 이유도 HiPPO에서 찾을 수 있다: HiPPO-LagT의 $A$는 이미 단순한 구조를 가지며, 대각화해도 표현력 손실이 제한적이다.

---

## 6. 실험 결과

### 장거리 의존성 벤치마크

| 모델 | Permuted MNIST | Sequential CIFAR |
|------|---------------|-----------------|
| LSTM | 87.1% | - |
| Transformer | 97.9% | - |
| **HiPPO-RNN** | **98.3%** | - |

HiPPO를 적용한 RNN이 Transformer를 능가한다.

### 복사 과제 (Copy Task)

시퀀스의 앞부분을 기억했다가 나중에 재현하는 과제:

| 지연 길이 | LSTM | GRU | HiPPO-RNN |
|----------|------|-----|-----------|
| 100 | 성공 | 성공 | **성공** |
| 1,000 | 실패 | 실패 | **성공** |
| 10,000 | 실패 | 실패 | **성공** |

HiPPO-RNN은 10,000 스텝 이상의 의존성도 처리한다.

### 함수 근사 품질

$N = 64$ 차원 상태로 다양한 함수를 근사:
- LegS: 과거 전체를 균등하게 잘 기억
- LagT: 최근에 편향되지만 지수적 감쇠 패턴에 최적
- LegT: 고정 윈도우 내에서 정밀

---

## 7. HiPPO의 이론적 의의

### 기억의 최적성

HiPPO가 제공하는 핵심 보장:

> 주어진 측도 $\mu^{(t)}$ 하에서, $N$차 다항식 근사의 **최적 계수를 온라인으로** 추적한다.

이것은 단순한 경험적 방법이 아니라, **직교 다항식 이론**에 기반한 수학적 최적성을 가진다.

### 기존 방법의 재해석

HiPPO 관점에서 기존 시퀀스 모델을 재해석할 수 있다:

| 모델 | HiPPO 관점 |
|------|-----------|
| **Exponential moving average** | HiPPO-LagT의 $N = 1$ 특수 경우 |
| **LSTM forget gate** | 지수적 감쇠 (LagT와 유사), but 학습된 decay |
| **Sliding window** | LegT의 극한 ($N \to \infty$, 윈도우 = $\theta$) |
| **Attention** | 이산적 측도에 대한 비압축 기억 |

### 정보 이론적 해석

$N$차 상태 벡터로 과거를 압축하면 반드시 정보 손실이 발생한다. HiPPO는 주어진 $N$에서 **최소 근사 오차**를 보장하는 압축을 제공한다. 이는:

$$\lVert f - f^{(t)} \rVert_{\mu^{(t)}} \leq C \cdot \frac{1}{N^s} \lVert f^{(s)} \rVert$$

함수 $f$가 $s$번 미분 가능하면, 근사 오차가 $N^{-s}$로 감소한다.

---

## 8. SSM 계보 정리

```
HiPPO (2020)
  │  "함수 근사로서의 기억" → A, B 행렬 도출
  │
  ├── LSSL (2021)
  │     "Linear State Space Layer" → 딥러닝에 통합
  │
  ├── S4 (2021)
  │     NPLR 분해 → 효율적 컨볼루션 커널
  │     │
  │     ├── S4D (2022): 대각화로 단순화
  │     ├── S5 (2022): 병렬 scan 도입
  │     └── H3 (2023): SSM + gating
  │
  └── Mamba (2023)
        Selective SSM (입력 의존적 B, Δ)
        │
        ├── Mamba-2 (2024): SSD (SSM = Attention)
        └── Hymba (2024): SSM + Attention 하이브리드
```

---

## 9. 한계와 열린 질문

1. **다항식 기저의 한계**: 불연속 함수, 급격한 변화에 대해 다항식 근사가 비효율적 (Gibbs 현상)
2. **측도 선택**: 어떤 측도가 주어진 과제에 최적인지 사전에 알 수 없음 → Mamba의 학습 가능한 $\Delta$가 이를 우회
3. **고차원 상태의 비용**: 이론적으로는 $N$을 키우면 좋지만, 실제로는 계산 비용과 트레이드오프
4. **비선형 확장**: HiPPO는 선형 시스템 이론에 기반 → 비선형 동역학에 대한 확장 연구 필요

---

## 10. 개인적 생각

HiPPO는 Mamba와 S4 계열 SSM의 **수학적 뿌리**다. 이 논문을 읽으면 Mamba의 설계 결정들이 왜 그렇게 이루어졌는지 깊이 이해할 수 있다:

- **$A$ 행렬**: HiPPO가 도출한 최적 상태 전이 행렬
- **$\Delta$ (스텝 크기)**: 이산화 파라미터 → Mamba에서 입력 의존적으로 확장
- **상태 차원 $N$**: 기억의 "해상도" — 크면 정밀하지만 비용 증가

가장 인상적인 것은 **"기억이란 무엇인가?"**라는 인지과학적 질문을 **직교 다항식 이론**으로 형식화한 점이다. "과거를 기억한다 = 과거 함수를 직교 기저로 투영한다"는 관점은, RNN/LSTM의 forget gate를 ad-hoc한 설계가 아닌 **수학적 원리의 특수 경우**로 재해석하게 해준다.

SancMamba에서 바이트 시퀀스의 장거리 의존성을 다루려면, HiPPO의 LegS(과거 전체 균등 기억)가 적합할 수 있다 — 바이트 수준에서는 어떤 위치의 정보가 나중에 중요해질지 예측하기 어렵기 때문이다.
