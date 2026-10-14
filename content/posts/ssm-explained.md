---
title: "[개념 정리] State Space Model(SSM) — 수식으로 이해하는 Mamba의 수학적 기반"
date: 2026-03-24
tags: ["논문리뷰", "SSM", "State Space Model", "Mamba"]
categories: ["ML/AI"]
summary: "State Space Model을 수식 중심으로 설명한다. 연속 시간 ODE에서 이산화, 글로벌 컨볼루션 해석, LTI의 한계, Mamba의 선택적 SSM, 그리고 Mamba-2의 SSM=Attention 이중성까지 단계별로 따라간다."
math: true
toc: true
draft: false
---

<!-- LEARNING-START -->
## 처음 읽는 분을 위한 안내

**학습 위치:** [문장을 읽으며 기억하는 모델]({{< relref "learn" >}}#memory) · 2/13

긴 글을 읽으며 요약 메모를 계속 고친다고 생각해 보자. SSM은 이전 상태와 새 입력으로 다음 상태를 만들고, 거기서 출력을 읽는 구조다. 상태·입력·출력의 역할을 먼저 익히면 행렬 기호가 무엇을 하는지 따라갈 수 있다.

**먼저 알아둘 말**

- [상태·SSM]({{< relref "ai-glossary" >}}#state): 상태는 이전까지 읽은 정보를 다음 단계에 전달하는 내부 값이다. SSM은 이 상태가 입력에 따라 바뀌고 출력을 만드는 관계를 모델링한다.
- [시퀀스]({{< relref "ai-glossary" >}}#sequence): 순서가 있는 데이터. 문장의 토큰이나 시간에 따라 들어오는 관측처럼 앞뒤 순서가 계산에 영향을 주는 입력이다.
- [압축·병목]({{< relref "ai-glossary" >}}#compression): 정보를 더 작은 표현에 담는 일. 저장량과 계산량을 줄일 수 있지만 어떤 정보가 남는지도 확인해야 한다.

**이번 글의 목표:** 새 입력과 과거 요약으로 상태가 갱신되는 흐름 이해하기.

**이 학습 경로의 앞 글:** [\[개념 정리\] Bayesian Surprise — 놀라움의 수학적 정의와 뇌의 주의 메커니즘]({{< relref "posts/bayesian-surprise" >}}).

토큰과 벡터가 낯설다면 [텍스트 데이터 다루기]({{< relref "posts/build-llm-ch2-working-with-text-data" >}})의 설명부터 읽자.

**본문에서 먼저 볼 순서:** **5. $A$ 행렬의 구조: HiPPO** → **6. LTI의 한계와 Mamba의 해결**.

첫 독서에서는 문제와 방법의 흐름을 잡자. 수식·코드·세부 성능표는 두 번째에 읽어도 된다. 구현과 수학적 주장까지 확인하려면 그 부분을 다시 살펴봐야 한다.

<!-- PAPER-PREREQUISITES -->
### 선수 개념과 읽을 논문

**먼저 이해할 개념:** 벡터·행렬, 상태의 재귀 갱신, 연속과 이산 시간; 수식 독서에는 미분과 선형대수.

입력·상태·출력과 행렬·미분의 뜻을 먼저 익히자. HiPPO와 S4는 이 글의 수학을 실제 모델에 연결하는 읽기이며, 첫 직관 설명 전에 완독할 필요는 없다.

아래 순서는 이 글을 이해하기 위한 공부 순서다. ‘개념 준비’는 읽기 안내이며, 직접적인 연구 계보는 연결 칸에 따로 표시했다. 이미 해당 개념을 설명할 수 있다면 아는 부분은 넘어가도 된다.

| 순서·논문 | 연결·익힐 개념 | 먼저 볼 부분 |
|---|---|---|
| 1. [HiPPO: Recurrent Memory with Optimal Polynomial Projections](https://arxiv.org/abs/2008.07669) · [블로그 해설]({{< relref "posts/hippo-review" >}}) | **개념 준비**: 상태·과거 압축·다항식 투영 | 기억을 계수로 저장하고 새 입력으로 갱신하는 문제 설정 |
| 2. [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396) | **개념 준비**: SSM·이산화·재귀와 컨볼루션 | 상태 공간식과 긴 시퀀스를 효율적으로 계산하는 구조 |

**현재 글과 연결하기:** 위 개념을 익힌 뒤, 이 글에서 ‘새 입력과 과거 요약으로 상태가 갱신되는 흐름 이해하기’에 해당하는 부분을 읽자. 기존 방법의 어느 단계가 바뀌었는지 그림이나 입력·출력으로 설명해 보자.

[선수 논문 전체 지도와 원문 읽는 법]({{< relref "paper-prerequisites" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- PAPER-PREREQUISITES-END -->

<!-- LEARNING-START-END -->

---

## 1. 출발점: 연속 시간 상태 공간

SSM(State Space Model)은 **제어 이론**에서 온 개념이다. 시스템의 상태를 벡터 $h(t)$로 표현하고, 이 벡터가 살고 있는 공간을 "상태 공간(State Space)"이라 부른다.

연속 시간 입력 $x(t) \in \mathbb{R}$을 은닉 상태 $h(t) \in \mathbb{R}^N$을 거쳐 출력 $y(t) \in \mathbb{R}$로 변환한다:

$$\dot{h}(t) = A \cdot h(t) + B \cdot x(t)$$

$$y(t) = C \cdot h(t) + D \cdot x(t)$$

각 행렬의 역할:

| 행렬 | 크기 | 역할 |
|------|------|------|
| $A$ | $\mathbb{R}^{N \times N}$ | **상태 전이** — 은닉 상태의 시간 진화 규칙 |
| $B$ | $\mathbb{R}^{N \times 1}$ | **입력 투영** — 입력이 상태에 미치는 영향 |
| $C$ | $\mathbb{R}^{1 \times N}$ | **출력 투영** — 상태에서 출력을 읽는 방법 |
| $D$ | $\mathbb{R}$ | **스킵 연결** — 입력을 출력에 직접 전달 |

직관적으로 보면 **$h(t)$는 과거 입력을 압축한 요약**이다.

입력은 지금 들어온 값이고, 상태는 지금까지 읽은 내용을 다음 시점으로 전달하는 내부 값이다. 출력은 그 상태에서 현재 필요한 정보를 읽어낸 결과다. 아래 수식에서 상태가 변하는 식과 출력을 읽는 식이 따로 있는 것은 이 역할 차이 때문이다. 이산화한 뒤에도 새 입력을 반영하고 이전 상태를 이어받는 흐름은 그대로 남는다.

---

## 2. 이산화: 연속 → 이산 시퀀스

실제 데이터는 연속이 아니라 이산 시퀀스 $x_0, x_1, x_2, \ldots$이다. 스텝 크기 $\Delta$로 연속 ODE를 이산 재귀로 변환한다.

### Zero-Order Hold (ZOH) 이산화

$$\bar{A} = \exp(\Delta \cdot A)$$

$$\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$$

이산화된 재귀:

$$h_t = \bar{A} \cdot h_{t-1} + \bar{B} \cdot x_t$$

$$y_t = C \cdot h_t + D \cdot x_t$$

이것은 본질적으로 **RNN**이다. 하지만 $\bar{A}$, $\bar{B}$가 HiPPO 이론에 기반한 특별한 구조를 가지므로, 일반 RNN보다 장거리 의존성을 훨씬 잘 처리한다.

---

## 3. 재귀를 풀어보면: 글로벌 컨볼루션

재귀를 반복 전개하면:

$$h_0 = \bar{B} x_0$$

$$h_1 = \bar{A}\bar{B} x_0 + \bar{B} x_1$$

$$h_2 = \bar{A}^2\bar{B} x_0 + \bar{A}\bar{B} x_1 + \bar{B} x_2$$

출력 $y_t = C h_t$를 대입하면:

$$y_t = \sum_{k=0}^{t} C \bar{A}^{t-k} \bar{B} \cdot x_k$$

이것은 **컨볼루션**이다. 커널 $\bar{K}$를 정의하면:

$$\bar{K} = (C\bar{B}, \quad C\bar{A}\bar{B}, \quad C\bar{A}^2\bar{B}, \quad \ldots, \quad C\bar{A}^{L-1}\bar{B})$$

$$y = \bar{K} \ast x \quad \text{(글로벌 컨볼루션)}$$

### SSM의 이중성

| 모드 | 복잡도 | 용도 |
|------|-------|------|
| **재귀 (RNN)** | $O(L)$ 순차적, 토큰당 $O(1)$ | 추론 (생성) |
| **컨볼루션 (CNN)** | $O(L \log L)$ 병렬 (FFT) | 학습 |

학습 시에는 컨볼루션 모드로 GPU를 최대 활용하고, 추론 시에는 재귀 모드로 토큰당 $O(1)$에 생성한다.

---

## 4. 구체적 예시: 숫자로 따라가기

$N=2$ (상태 차원 2), 시퀀스 $x = [1, 0, 0, 0, 1]$로 직접 계산해보자.

$$A = \begin{bmatrix} -1 & 0 \\ 0 & -2 \end{bmatrix}, \quad B = \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \quad C = \begin{bmatrix} 1 & 1 \end{bmatrix}$$

$\Delta = 1$로 이산화:

$$\bar{A} = \exp(\Delta A) = \begin{bmatrix} e^{-1} & 0 \\ 0 & e^{-2} \end{bmatrix} \approx \begin{bmatrix} 0.37 & 0 \\ 0 & 0.14 \end{bmatrix}$$

**단계별 계산:**

| $t$ | $x_t$ | $h_t$ | $y_t = Ch_t$ | 해석 |
|-----|-------|-------|-------------|------|
| 0 | 1 | $[1.0, 1.0]$ | 2.0 | 입력 기록 |
| 1 | 0 | $[0.37, 0.14]$ | 0.51 | 감쇠하며 기억 |
| 2 | 0 | $[0.14, 0.02]$ | 0.16 | 더 감쇠 |
| 3 | 0 | $[0.05, 0.003]$ | 0.05 | 거의 잊혀짐 |
| 4 | 1 | $[1.02, 1.00]$ | 2.02 | 새 입력 + 미세한 과거 잔류 |

**$A$의 대각 원소가 감쇠 속도를 결정한다:**
- 첫 번째 상태 ($A_{11} = -1$): 느리게 감쇠 → **장기 기억**
- 두 번째 상태 ($A_{22} = -2$): 빠르게 감쇠 → **단기 기억**

$N$을 키우면 다양한 시간 스케일의 기억을 동시에 유지할 수 있다.

---

## 5. $A$ 행렬의 구조: HiPPO

일반 RNN의 문제: $A$가 임의 행렬이면 $\bar{A}^t$가 폭발하거나 소실한다.

HiPPO의 해결: **과거 입력을 직교 다항식 기저로 최적 투영**하는 $A$ 행렬을 도출했다.

$$A_{nk} = -\begin{cases} (2n+1)^{1/2}(2k+1)^{1/2} & n \gt  k \\ n+1 & n = k \\ 0 & n \lt  k \end{cases}$$

$N$차원 상태 벡터가 과거 함수를 르장드르 다항식으로 최적 근사한다. 근사 오차:

$$\lVert f - f^{(t)} \rVert \leq C \cdot \frac{1}{N^s}$$

S4D에서는 이를 **대각 행렬로 단순화**:

$$A = \text{diag}(-1, -2, -3, \ldots, -N)$$

놀랍게도 성능이 거의 동일하다.

---

## 6. LTI의 한계와 Mamba의 해결

S4까지의 SSM은 **LTI (Linear Time-Invariant)** — $A$, $B$, $C$가 모든 시점에서 동일하다.

**문제**: "이 토큰이 중요한가?"를 판단할 수 없다.

**Mamba의 해결**: $B$, $C$, $\Delta$를 **입력 의존적**으로 만든다.

$$B_t = \text{Linear}_B(x_t), \quad C_t = \text{Linear}_C(x_t), \quad \Delta_t = \text{softplus}(\text{Linear}_\Delta(x_t))$$

이산화:

$$\bar{A}_t = \exp(\Delta_t \cdot A)$$

$$h_t = \bar{A}_t \cdot h_{t-1} + \Delta_t \cdot B_t \cdot x_t$$

$$y_t = C_t \cdot h_t$$

### $\Delta_t$의 역할: 선택적 메모리

| $\Delta_t$ 값 | $\bar{A}_t = e^{\Delta_t A}$ | 효과 |
|-------------|--------------------------|------|
| **크면** | $\approx 0$ | 이전 상태 **잊고** 새 입력 강하게 기록 |
| **작으면** | $\approx I$ | 이전 상태 **보존**, 새 입력 무시 |

모델이 입력을 보고 "중요하다" ($\Delta$ 크게) 또는 "무시해도 된다" ($\Delta$ 작게)를 **스스로 결정**한다. 이것이 Transformer의 어텐션에 대응하는 **내용 기반 선택 메커니즘**이다.

---

## 7. Mamba-2의 SSD: SSM = Attention

$A$를 스칼라 $\times$ 항등행렬로 단순화하면:

$$h_t = \alpha_t \cdot h_{t-1} + \gamma_t \cdot B_t \cdot x_t$$

$$y_t = C_t^\top \cdot h_t$$

재귀를 풀면:

$$y_t = \sum_{s=1}^{t} C_t^\top \left(\prod_{k=s+1}^{t} \alpha_k\right) B_s \cdot \gamma_s \cdot x_s$$

행렬 형태:

$$Y = (L \odot C B^\top) X$$

여기서 $L$은 **감쇠 마스크** (하삼각):

$$L_{t,s} = \prod_{k=s+1}^{t} \alpha_k$$

**이것은 causal attention과 동일한 구조다:**

| Attention | SSM (SSD) |
|-----------|-----------|
| $Q$ | $C$ |
| $K$ | $B$ |
| $V$ | $X$ |
| Softmax mask | 감쇠 마스크 $L$ |

---

## 8. 전체 그림: SSM의 세 가지 얼굴

```
              SSM
           /   |   \
     재귀     컨볼루션    행렬곱
    (RNN)    (CNN)    (Attention)
      |        |         |
  O(1)/step  O(L log L)  O(L²)
  추론 최적  학습 최적   표현력 최적
```

SSM은 **같은 연산의 세 가지 다른 계산 방식**이다:
1. **재귀**: $h_t = \bar{A}h_{t-1} + \bar{B}x_t$ — 추론에 최적
2. **컨볼루션**: $y = \bar{K} \ast x$ — S4 학습에 사용
3. **행렬곱**: $Y = (L \odot CB^\top)X$ — Mamba-2 학습에 사용

이 세 가지가 수학적으로 동일하다는 것이 SSM의 가장 핵심적인 통찰이다.

---

## 9. SSM 계보 요약

| 모델 | 연도 | $A$ | $B$, $C$ | 계산 | 핵심 기여 |
|------|------|-----|---------|------|----------|
| **S4** | 2021 | HiPPO (고정) | 고정 | FFT 컨볼루션 | SSM을 실용화 |
| **S4D** | 2022 | 대각 (고정) | 고정 | FFT 컨볼루션 | 대각으로 단순화 |
| **S5** | 2023 | 대각 (고정) | 고정 | Parallel scan | MIMO + 병렬 스캔 |
| **Mamba** | 2023 | 대각 (고정) | **입력 의존** | CUDA parallel scan | 선택적 SSM |
| **Mamba-2** | 2024 | 스칼라 (고정) | **입력 의존** | Chunk matmul (SSD) | SSM = Attention 증명 |
| **Mamba-3** | 2026 | **입력 의존** | **입력 의존** | Triton/TileLang | Exp-Trap + 복소수 + MIMO |

<!-- LEARNING-END -->
---

## 읽은 뒤 확인하고 다음으로

**확인 질문:** 새 입력과 과거 요약으로 상태가 갱신되는 흐름 이해하기를 목표로, 본문에서 다룬 입력·중간 과정·결과를 자기 말로 설명해 보자. 실험이나 사례가 있다면 어떤 조건에서 확인했는지도 덧붙여 보자.

**다음 글:** [\[논문 리뷰\] HiPPO: 장거리 의존성 문제의 수학적 해결 — SSM과 Mamba의 이론적 기반]({{< relref "posts/hippo-review" >}}) — 긴 과거를 정해진 크기의 기억에 담는 문제 이해하기.

[전체 학습 순서]({{< relref "learn" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- LEARNING-END-END -->
