---
title: "[논문 리뷰] Mamba-3: SSM 원리로 추론 효율성의 Pareto 프론티어를 진전시키다"
date: 2026-03-19
tags: ["논문리뷰", "Mamba", "SSM", "State Space Model"]
categories: ["ML/AI"]
summary: "Mamba-3 논문 리뷰 (ICLR 2026). Exponential-Trapezoidal 이산화, 복소수 상태 공간, MIMO 공식화 — 세 가지 방법론적 개선으로 Mamba-2 대비 절반의 상태 크기에서 동등한 perplexity를 달성하고, 추론 효율성-성능 Pareto 프론티어를 진전시킨다."
math: true
toc: true
draft: false
---

## 논문 정보

- **제목**: Mamba-3: Improved Sequence Modeling using State Space Principles
- **저자**: Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, Albert Gu
- **소속**: Carnegie Mellon University, Princeton University, Together AI, Cartesia AI
- **발표**: ICLR 2026 (arXiv: 2603.15569, 2026.03.16)
- **코드**: [github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)

---

## 1. 동기: 선형 모델의 세 가지 한계

Mamba-2, Gated DeltaNet(GDN) 등 sub-quadratic 모델이 Transformer에 필적하는 성능을 보이지만, 세 가지 근본적 한계가 남아 있다:

1. **표현력 부족**: Mamba-2는 $A$를 스칼라로 단순화하면서 표현력을 희생 → 추론 효율 대비 성능 열화
2. **상태 추적 불능**: 실수 고유값 제약으로 parity 같은 단순 state-tracking 과제도 해결 불가
3. **하드웨어 비효율**: 이론적으로 선형 추론이지만, 실제 디코딩의 arithmetic intensity가 ~2.5 ops/byte에 불과 → GPU 연산 자원 대부분이 유휴

> Mamba-3의 접근: **추론 우선(inference-first)** 관점에서, SSM 원리에 기반한 세 가지 방법론적 개선을 도입한다.

---

## 2. 개선 1: Exponential-Trapezoidal 이산화

### 배경: 이산화란?

연속 시간 SSM $\dot{h}(t) = A(t)h(t) + B(t)x(t)$를 이산 시퀀스로 변환하는 과정이다. 기존 방법들:

| 이산화 방법 | $\alpha_t$ | $\gamma_t$ | 사용처 |
|-----------|-----------|-----------|-------|
| ZOH | $\exp(\Delta_t A_t)$ | $A_t^{-1}(\exp(\Delta_t A_t) - I)$ | S4D, S5 |
| **Exponential-Euler** | $\exp(\Delta_t A_t)$ | $\Delta_t$ | Mamba-1, -2 |
| **Exponential-Trapezoidal** | $\exp(\Delta_t A_t)$ | $\lambda_t \Delta_t$ | **Mamba-3** |

### Mamba-1/2의 이산화는 사실 Euler 근사였다

논문의 중요한 발견: Mamba-1/2가 "ZOH"라고 주장했지만, 실제 구현은 **Euler 근사**(1차 근사)였다. Mamba-3는 이를 이론적으로 정리하고, **2차 정확도**를 가진 사다리꼴 규칙으로 확장한다.

### Exponential-Trapezoidal 규칙

상태 입력 적분을 **양 끝점의 데이터 의존적 볼록 결합**으로 근사한다:

$$h_t = e^{\Delta_t A_t} h_{t-1} + (1-\lambda_t)\Delta_t e^{\Delta_t A_t} B_{t-1} x_{t-1} + \lambda_t \Delta_t B_t x_t$$

$$=: \alpha_t h_{t-1} + \beta_t B_{t-1} x_{t-1} + \gamma_t B_t x_t$$

여기서:
- $\alpha_t = e^{\Delta_t A_t}$: 상태 전이 (감쇠)
- $\beta_t = (1-\lambda_t)\Delta_t e^{\Delta_t A_t}$: **이전 시점** 입력 기여
- $\gamma_t = \lambda_t \Delta_t$: **현재 시점** 입력 기여
- $\lambda_t \in [0, 1]$: **데이터 의존적** 보간 파라미터

**특수 경우:**
- $\lambda_t = 1$ → Mamba-2의 Euler 이산화 복원
- $\lambda_t = 1/2$ → 고전적 사다리꼴 규칙

### 암묵적 컨볼루션으로의 해석

Exponential-Trapezoidal 재귀는 **상태 입력 $B_t x_t$에 대한 데이터 의존적 width-2 컨볼루션**과 동등하다. 이것은 기존 Mamba/GDN의 $x_t$에 대한 외부 컨볼루션(Conv1d)과 근본적으로 다르다 — **코어 재귀 내부**에서의 컨볼루션이다.

> 결과적으로 Mamba-3는 explicit $B$, $C$ 바이어스 항과 결합하여 **short causal convolution(Conv1d)을 제거**할 수 있다. 이는 기존에 recurrent 모델에 필수적이라고 여겨졌던 것이다.

---

## 3. 개선 2: 복소수 상태 공간 (Complex-Valued SSM)

### 문제: 실수 고유값의 한계

Mamba-2는 효율성을 위해 $A$를 스칼라(실수)로 단순화했다. 그러나 실수 고유값은 **회전 동역학을 표현할 수 없다**.

예시 — Parity 과제 ($\sum_i x_i \mod 2$):
- 이 과제는 2D 회전 행렬 $R(\pi x_t)$로 풀 수 있다
- 실수 대각 행렬로는 회전을 표현할 수 없다 → Mamba-2는 random guessing과 동등

### 해결: 복소수 SSM

Mamba-3는 상태 공간을 복소수로 확장한다:

$$\dot{h}(t) = \text{Diag}(A(t) + i\theta(t)) \cdot h(t) + (B(t) + i\hat{B}(t)) \cdot x(t)$$

이산화하면:

$$h_t = e^{\Delta_t A_t} R_t \cdot h_{t-1} + \Delta_t B_t x_t$$

여기서 $R_t$는 **블록 대각 2×2 회전 행렬**:

$$R_t = \text{Block}\left(\{R(\Delta_t \theta_t[i])\}_{i=1}^{N/2}\right), \quad R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

### RoPE와의 연결

논문의 흥미로운 발견: 복소수 SSM은 **데이터 의존적 Rotary Position Embedding (RoPE)**과 동등하다.

$$h_t = e^{\Delta_t A_t} R_t \cdot h_{t-1} + \left(\prod_{i=0}^{t} R_i^\top\right) \Delta_t B_t x_t$$

$$y_t = \left[\left(\prod_{i=0}^{t} R_i^\top\right) C_t\right]^\top h_t$$

- 표준 RoPE: 고정 주파수 스케줄 $\theta[i] = 10000^{-2i/N}$
- Mamba-3: **데이터 의존적** 회전 각도 $\Delta_t \theta_t[i]$

SSD 프레임워크의 $Q = C$, $K = B$와 대응시키면, RoPE가 $B$, $C$에 적용되는 것과 같은 구조다. 이를 **"RoPE trick"**이라 부르며, 효율적인 실수 연산으로 복소수 SSM을 구현할 수 있게 해준다.

### 상태 추적 능력

| 모델 | Parity 정확도 |
|------|------------|
| Mamba-2 (실수) | ~50% (random) |
| Mamba-3 without RoPE | ~50% (random) |
| **Mamba-3 (복소수, data-dep RoPE)** | **~100%** |

---

## 4. 개선 3: Multi-Input, Multi-Output (MIMO)

### 문제: SSM 디코딩의 낮은 하드웨어 활용률

기존 SISO (Single-Input, Single-Output) SSM의 디코딩:
- Arithmetic intensity ≈ **2.5 ops/byte** (H100의 matmul은 295 ops/byte)
- GPU 연산 능력 대부분이 **유휴 상태** → memory-bound

### SISO → MIMO 전환

외적(outer product) 기반 상태 업데이트를 **행렬곱 기반**으로 전환한다:

**SISO** (기존):
$$h_t \leftarrow \alpha_t h_{t-1} + \Delta_t B_t x_t^\top \quad (B_t \in \mathbb{R}^N, x_t \in \mathbb{R}^P)$$

**MIMO** (Mamba-3):
$$h_t \leftarrow \alpha_t h_{t-1} + \Delta_t B_t x_t^\top \quad (B_t \in \mathbb{R}^{N \times R}, x_t \in \mathbb{R}^{P \times R})$$

$R$ = MIMO rank. $B_t$의 차원을 $R$배 키워도 메모리 트래픽은 상태 $h_t$가 지배하므로 거의 증가하지 않지만, FLOPs는 $R$배 증가한다.

| | Arithmetic Intensity |
|--|---------------------|
| SISO | $\Theta(1)$ ≈ 2.5 |
| **MIMO** | $\Theta(R)$ (R에 비례) |

### 학습 효율

MIMO SSM은 $R^2$개의 SISO SSM으로 분해 가능하지만, 실제로는:
- 재귀(순환) 모드에서 $R$배만 오버헤드 ($R^2$가 아닌)
- 청크 크기를 $C_{\text{MIMO}} = C_{\text{SISO}} / R$로 조정하면 총 FLOPs는 $R$배 증가에 그침

> MIMO의 핵심: **디코딩 지연(latency)을 늘리지 않으면서 모델 FLOPs를 증가**시켜, 동일 wall-clock 시간에 더 좋은 모델을 만든다.

---

## 5. Mamba-3 아키텍처 블록

세 가지 개선을 결합한 Mamba-3 블록:

```
Input (B, L, D)
  │
  ├── Linear (D → E)  ──→ SiLU ──→ [Conv1d 제거] ──→ Complex SSM ──→ ×
  │                          Exp-Trapezoidal 재귀      (RoPE trick)    │
  │                          + explicit B,C bias                       │
  └── Linear (D → E)  ──→ SiLU ───────────────────────────────→ gate
                                                                       │
                                                                  Linear (E → D)
                                                                       │
                                                                    Output
```

**Mamba-2와의 구조적 차이:**
- Conv1d **제거** (Exp-Trapezoidal이 대체)
- $A_t$ **데이터 의존적**으로 전환 (Mamba-2는 데이터 독립)
- **복소수 상태** + RoPE trick
- MIMO 변형에서 $B$, $C$ 차원 확장

---

## 6. 실험 결과

### 1.5B 언어 모델링

| 모델 | Avg Downstream Acc | Perplexity |
|------|-------------------|------------|
| Transformer++ | 기준 | 기준 |
| GDN (Gated DeltaNet) | 기준 + α | - |
| Mamba-2 | GDN 대비 열등 | - |
| **Mamba-3 (SISO)** | GDN **+0.6%p** | - |
| **Mamba-3 (MIMO)** | GDN **+1.8%p**, Transformer++ **+2.2%p** | - |

Mamba-3 (MIMO)가 Transformer++ 대비 **+2.2%p**, Mamba-2 대비 **+1.9%p** 개선.

### 상태 크기 효율

| 모델 | State Size | Perplexity |
|------|-----------|------------|
| Mamba-2 | 128 | 기준 |
| **Mamba-3 (MIMO)** | **64** | **동등** |

Mamba-3는 **절반의 상태 크기**로 Mamba-2와 동등한 perplexity를 달성한다.

### 하드웨어 효율 (MIMO)

| | Decode FLOPs | Decode Latency |
|--|-------------|---------------|
| Mamba-2 (SISO) | 기준 | 기준 |
| Mamba-3 (MIMO, R=4) | **4×** | **동일** |

MIMO가 디코딩 FLOPs를 4배 늘리면서도 **wall-clock 지연은 동일** → 유휴 하드웨어를 활용.

### State-Tracking 과제

| 과제 | Mamba-2 | Mamba-3 (no RoPE) | Mamba-3 (RoPE) |
|------|---------|-------------------|----------------|
| Parity | Random | Random | **Perfect** |
| 기타 합성 과제 | 실패 | 실패 | **성공** |

복소수 SSM + data-dependent RoPE가 이전 선형 모델이 풀 수 없던 과제를 해결.

---

## 7. 이론적 의의: Mamba 계보의 통합

### 이산화 방법의 통합

Mamba-3는 SSM 이산화의 **통합적 프레임워크**를 제시한다:

$$h_t = \alpha_t h_{t-1} + \beta_t B_{t-1} x_{t-1} + \gamma_t B_t x_t$$

| $\lambda_t$ | 이산화 | 결과 |
|------------|-------|------|
| $\lambda_t = 1$ | Exponential-Euler | = Mamba-1, -2 |
| $\lambda_t = 1/2$ | 고전적 사다리꼴 | 고정 보간 |
| $\lambda_t$ 학습 | **Exp-Trapezoidal** | = **Mamba-3** |

### SSD와의 연결

Mamba-3는 여전히 SSD 프레임워크의 인스턴스지만, 마스크 $L$이 **2-band semi-separable matrix**로 확장된다:

$$L = \underbrace{\begin{bmatrix} 1 \\ \alpha_1 & 1 \\ \alpha_2\alpha_1 & \alpha_2 & 1 \\ \vdots & & \ddots & \ddots \end{bmatrix}}_{\text{decay}} \cdot \underbrace{\begin{bmatrix} \gamma_0 \\ \beta_1 & \gamma_1 \\ 0 & \beta_2 & \gamma_2 \\ \vdots & & \ddots & \ddots \end{bmatrix}}_{\text{2-band conv}}$$

Mamba-2는 $\gamma_t$만 있는 대각 행렬이었다. Mamba-3는 $\beta_t$를 추가하여 **decay와 2-band 컨볼루션의 곱** 구조를 가진다.

---

## 8. Mamba 계보 최종 정리

| 세대 | 핵심 기여 | 이산화 | 상태 | Conv1d | 발표 |
|------|----------|-------|------|--------|------|
| **Mamba-1** | Selective SSM | Exp-Euler (비공식) | 실수 대각 | 필요 | 2023.12 |
| **Mamba-2** | SSD (SSM=Attention) | Exp-Euler | 실수 스칼라×I | 필요 | 2024.05 |
| **Mamba-3** | Exp-Trap + Complex + MIMO | **Exp-Trapezoidal** | **복소수** (RoPE) | **제거** | 2026.03 |

---

## 9. 개인적 생각

Mamba-3는 "더 빠른 Mamba"가 아니라, SSM의 **이론적 기반을 재정비**하면서 실용적 성능을 끌어올린 논문이다.

가장 인상적인 점:

1. **Mamba-1/2의 이산화가 사실 Euler 근사였다**는 발견과 이의 일반화. 이론적 정당성 없이 쓰이던 heuristic을 체계적으로 정리했다.

2. **복소수 SSM = Data-dependent RoPE**라는 연결. Transformer에서 위치 인코딩으로 쓰이던 RoPE가 SSM에서는 **상태 전이의 회전 성분**으로 자연스럽게 나타난다. 이 통합은 Mamba-2의 "SSM = Attention" 이중성을 더 깊게 만든다.

3. **MIMO의 실용성**. 이론적으로 단순한 아이디어(차원 확장)지만, memory-bound인 SSM 디코딩에서 유휴 연산 자원을 활용한다는 관점이 하드웨어 친화적이다. "같은 시간에 더 많은 연산을 쓸 수 있다면, 그렇게 하라"는 원칙.

4. **Conv1d 제거**. Exp-Trapezoidal 이산화가 코어 재귀 내에서 암묵적 컨볼루션을 수행하므로, 별도의 short convolution이 불필요해졌다. 아키텍처의 단순화는 항상 좋은 신호다.

SancMamba 프로젝트에서 특히 관련 있는 것은 **복소수 상태**다 — 바이트 수준의 반복 패턴(UTF-8 continuation bytes 등)에서 회전 동역학이 유용할 수 있다. 또한 MIMO의 추론 효율화는 바이트 시퀀스의 긴 디코딩에서 직접적으로 도움이 될 것이다.
