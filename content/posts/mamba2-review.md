---
title: "[논문 리뷰] Mamba-2: SSM과 Attention의 이중성을 밝히다"
date: 2026-03-19
tags: ["논문리뷰", "Mamba", "SSM", "State Space Model"]
categories: ["ML/AI"]
summary: "Mamba-2 논문 리뷰. Structured State Space Duality(SSD)를 통해 SSM과 어텐션이 본질적으로 동일한 연산임을 증명하고, 이를 활용해 Mamba-1 대비 2-8배 빠른 학습을 달성한다."
math: true
toc: true
draft: false
---

<!-- LEARNING-START -->
## 처음 읽는 분을 위한 안내

**학습 위치:** [문장을 읽으며 기억하는 모델]({{< relref "learn" >}}#memory) · 5/13

상태를 차례로 갱신하는 계산을 여러 구간으로 나눠 묶어서 계산할 수도 있다. Mamba-2는 같은 계산을 다른 형태로 바라보고 GPU에 맞게 실행하는 방법을 다룬다. 처음에는 계산의 의미와 실행 순서의 차이를 구분하는 데 집중하자.

**먼저 알아둘 말**

- [상태·SSM]({{< relref "ai-glossary" >}}#state): 상태는 이전까지 읽은 정보를 다음 단계에 전달하는 내부 값이다. SSM은 이 상태가 입력에 따라 바뀌고 출력을 만드는 관계를 모델링한다.
- [Attention]({{< relref "ai-glossary" >}}#attention): 현재 계산에 필요한 다른 위치의 정보를 가중치를 주어 섞는 장치. 가중치가 크다는 사실만으로 판단의 원인을 확정할 수는 없다.
- [GPU·병렬 처리]({{< relref "ai-glossary" >}}#gpu): GPU는 많은 계산을 함께 처리하는 데 쓰는 장치다. 병렬 처리는 작업을 나눠 동시에 수행하는 방식이다.

**이번 글의 목표:** 같은 계산을 순서대로 하거나 묶어서 하는 관점 이해하기.

**이 학습 경로의 앞 글:** [\[논문 리뷰\] Mamba: 선택적 상태 공간 모델로 Transformer에 도전하다]({{< relref "posts/mamba-ssm-review" >}}).

토큰과 벡터가 낯설다면 [텍스트 데이터 다루기]({{< relref "posts/build-llm-ch2-working-with-text-data" >}})의 설명부터 읽자.

**본문에서 먼저 볼 순서:** **1. 핵심 발견: SSM = Structured Attention** → **4. 실험 결과** → **6. 의의와 한계**.

첫 독서에서는 문제와 방법의 흐름을 잡자. 수식·코드·세부 성능표는 두 번째에 읽어도 된다. 구현과 수학적 주장까지 확인하려면 그 부분을 다시 살펴봐야 한다.

<!-- PAPER-PREREQUISITES -->
### 선수 개념과 읽을 논문

**먼저 이해할 개념:** 벡터·행렬, 상태의 재귀 갱신, 연속과 이산 시간; 수식 독서에는 미분과 선형대수.

아래 순서는 이 글을 이해하기 위한 공부 순서다. ‘개념 준비’는 읽기 안내이며, 직접적인 연구 계보는 연결 칸에 따로 표시했다. 이미 해당 개념을 설명할 수 있다면 아는 부분은 넘어가도 된다.

| 순서·논문 | 연결·익힐 개념 | 먼저 볼 부분 |
|---|---|---|
| 1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | **개념 준비**: Attention·위치 정보·인코더/디코더 | 현재 토큰이 다른 위치의 정보를 읽는 구조 그림과 Attention 설명 |
| 2. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) · [블로그 해설]({{< relref "posts/mamba-ssm-review" >}}) | **앞선 모델과의 비교**: 입력 의존 상태 갱신·선택 메커니즘 | 입력에 따라 바뀌는 값과 selective scan의 역할 |

**현재 글과 연결하기:** 위 개념을 익힌 뒤, 이 글에서 ‘같은 계산을 순서대로 하거나 묶어서 하는 관점 이해하기’에 해당하는 부분을 읽자. 기존 방법의 어느 단계가 바뀌었는지 그림이나 입력·출력으로 설명해 보자.

[선수 논문 전체 지도와 원문 읽는 법]({{< relref "paper-prerequisites" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- PAPER-PREREQUISITES-END -->

<!-- LEARNING-START-END -->

---

## 논문 정보

- **제목**: Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality
- **저자**: Tri Dao, Albert Gu
- **발표**: ICML 2024 (arXiv: 2405.21060)
- **키워드**: State Space Duality, Structured Attention, Semi-Separable Matrices

---

## 1. 핵심 발견: SSM = Structured Attention

Mamba-2의 가장 중요한 기여는 **이론적 통합**이다.

> **SSM과 어텐션은 본질적으로 동일한 연산의 다른 표현이다.**

이 대응 관계를 **Structured State Space Duality (SSD)**라 부른다.

### 수학적 연결

Mamba-1의 selective SSM:
$$h_t = \bar{A}_t \cdot h_{t-1} + \bar{B}_t \cdot x_t$$

$$y_t = C_t \cdot h_t$$

이를 풀어쓰면:
$$y_t = \sum_{s=1}^{t} C_t \left(\prod_{k=s+1}^{t} \bar{A}_k \right) \bar{B}_s \cdot x_s$$

이 합은 **마스크된 어텐션**과 동일한 구조다:
$$y_t = \sum_{s=1}^{t} M_{t,s} \cdot x_s$$

여기서 $M_{t,s} = C_t \left(\prod_{k=s+1}^{t} \bar{A}_k\right) \bar{B}_s$는 **structured mask**이다.

### Semi-Separable Matrix

행렬 $M$은 **semi-separable matrix** — 하삼각 부분이 저랭크 분해 가능한 행렬이다. 이 구조적 성질이 SSM의 효율적 계산(순환, 컨볼루션)과 어텐션의 병렬 계산을 하나의 프레임워크로 통합한다.

같은 상태 갱신을 재귀 형태로 쓰면 이전 상태를 이어받는 계산으로 보이고, 전체 시퀀스에 펼치면 구조를 가진 행렬 연산으로 보인다. SSD는 이 두 계산을 연결해 읽는 관점이다. 아래 청크 단위 알고리즘은 구간 안의 계산과 구간 사이의 상태 전달을 나눈다. 수학적 대응과 실제 GPU 계산 방식이 어떻게 이어지는지 함께 보면 이해하기 쉽다.

---

## 2. SSD 레이어 설계

### Mamba-1 vs Mamba-2

| 항목 | Mamba-1 | Mamba-2 |
|------|---------|---------|
| State 차원 | $N = 16$ | $N = 64\text{-}256$ |
| Head 구조 | 단일 | **Multi-head** (Transformer 유사) |
| $A$ 행렬 | 대각 (복소수 가능) | **스칼라 × Identity** |
| 계산 방식 | Parallel scan | **SSD 알고리즘 (chunk-wise)** |
| 학습 속도 | 기준 | **2-8배 빠름** |

### $A$의 단순화

Mamba-2의 핵심 설계 결정: $A$를 **스칼라**로 단순화한다.

$$\bar{A}_t = a_t \cdot I \quad (a_t \in \mathbb{R})$$

이렇게 하면 $\prod_{k=s+1}^{t} \bar{A}_k = \left(\prod_{k=s+1}^{t} a_k\right) \cdot I$가 되어, mask 행렬의 구조가 극적으로 단순화된다.

성능 손실 없이 계산 효율이 크게 향상되는 것이 실험적으로 확인되었다.

### Multi-Head 구조

Transformer의 multi-head attention과 유사하게, 입력을 여러 head로 분할하고 각 head에서 독립적으로 SSM을 수행한다:

```
Input (B, L, D)
  → split into P heads: (B, L, P, D/P)
  → 각 head: SSD layer with state dim N
  → concat → Linear → Output
```

이 설계가 Transformer와의 **아키텍처적 호환성**을 높인다.

---

## 3. SSD 알고리즘: Chunk-wise 계산

### 아이디어

시퀀스를 길이 $c$의 청크로 나누고, **청크 내부**는 행렬곱(어텐션 방식)으로, **청크 간**은 순환(SSM 방식)으로 계산한다.

```
[chunk 1] ──→ [chunk 2] ──→ [chunk 3] ──→ ...
  ↕ 내부         ↕ 내부         ↕ 내부
  행렬곱         행렬곱         행렬곱
  (병렬)         (병렬)         (병렬)
      ──── 순환 ────── 순환 ──── (순차)
```

### 장점

- **청크 내부**: GPU 텐서 코어 활용 극대화 (matmul 연산)
- **청크 간**: 상태 벡터만 전달하므로 IO 최소화
- **총 복잡도**: $O(L)$ 유지하면서 실제 wall-clock 시간 대폭 감소

### 추론 모드

추론 시에는 Mamba-1과 동일하게 **순환 모드** ($O(1)$ per token):

$$h_t = a_t \cdot h_{t-1} + B_t \cdot x_t$$

$$y_t = C_t \cdot h_t$$

---

## 4. 실험 결과

### 학습 속도

| 모델 | Throughput (상대) |
|------|-----------------|
| Mamba-1 | 1.0× |
| **Mamba-2** | **2-8×** |
| Transformer++ (FlashAttention-2) | 비교 가능 |

State 차원을 $N = 256$으로 키워도 Mamba-1의 $N = 16$보다 빠르다.

### 언어 모델링 성능

Pile 데이터셋에서:

| 모델 | Params | PPL |
|------|--------|-----|
| Transformer++ | 2.7B | 기준 |
| Mamba-1 | 2.8B | Transformer++와 동등 |
| **Mamba-2** | **2.7B** | **Mamba-1보다 개선** |

동일 파라미터 대비 Mamba-1을 일관되게 능가한다.

### 하이브리드 아키텍처

Mamba-2 + Attention 하이브리드를 실험:
- 8개 SSD 레이어 중 2개를 Attention으로 교체
- 순수 SSD 대비 **추가 개선** — In-context learning 능력 보완

---

## 5. 이론적 의미

### Transformer ↔ SSM 스펙트럼

SSD는 Transformer와 SSM이 **연속 스펙트럼의 양 끝**임을 보여준다:

```
Full Attention ←── SSD (structured mask) ──→ Pure SSM
 (O(L²), 유연)                                (O(L), 제한적)
```

- Full Attention: mask 없음, 모든 위치에 접근
- SSD: semi-separable mask, 구조적 패턴
- Pure SSM: 가장 강한 구조적 제약, 최대 효율

연구자는 이 스펙트럼에서 **최적 지점**을 선택할 수 있다.

### 기존 연구와의 연결

| 연구 | SSD 관점에서의 해석 |
|------|------------------|
| Linear Attention | $A = 0$인 SSD (decay 없음) |
| RetNet | 고정 decay를 가진 SSD |
| RWKV | 특수한 decay 패턴의 SSD |
| H3 | SSM + gating의 조합 |

모든 "efficient attention" 변형이 SSD의 특수 경우임을 보여준다.

---

## 6. 의의와 한계

### 의의

1. **이론적 통합**: SSM과 Attention의 이중성 증명 — 두 분야를 하나로 통합
2. **실용적 가속**: 2-8배 학습 속도 향상 (state 차원 증가에도 불구하고)
3. **설계 유연성**: Multi-head, 하이브리드 아키텍처 등 Transformer 생태계와 호환
4. **이론 → 알고리즘**: Semi-separable 구조에서 chunk-wise 알고리즘 자연스럽게 도출

### 한계

1. **$A$의 단순화**: 스칼라 $A$가 대각/복소수 $A$보다 표현력이 제한될 수 있음
2. **긴 문맥 성능**: 고정 크기 상태의 근본적 한계는 여전
3. **In-context learning**: 순수 SSD로는 여전히 Attention 대비 약함 (하이브리드 필요)

---

## 7. 개인적 생각

Mamba-2의 SSD 프레임워크는 단순한 "더 빠른 Mamba"를 넘어, **시퀀스 모델링의 통합 이론**을 제시한다. 특히:

- "Transformers are SSMs"라는 제목이 도발적이면서도 수학적으로 정당하다
- Chunk-wise 알고리즘이 이론(semi-separable matrix)에서 자연스럽게 도출되는 과정이 아름답다
- Multi-head 구조의 도입으로 Transformer 생태계의 기술(하이브리드, MoE 등)을 SSM에 이식할 수 있게 되었다

실용적으로는 Mamba-2가 "Attention을 완전히 대체"하기보다, **하이브리드의 SSM 컴포넌트**로서 자리잡을 가능성이 높아 보인다.

<!-- LEARNING-END -->
---

## 읽은 뒤 확인하고 다음으로

**확인 질문:** 같은 계산을 순서대로 하거나 묶어서 하는 관점 이해하기를 목표로, 본문에서 다룬 입력·중간 과정·결과를 자기 말로 설명해 보자. 실험이나 사례가 있다면 어떤 조건에서 확인했는지도 덧붙여 보자.

**다음 글:** [\[논문 리뷰\] Mamba-3: SSM 원리로 추론 효율성의 Pareto 프론티어를 진전시키다]({{< relref "posts/mamba3-review" >}}) — 상태 표현·갱신 근사·입출력 변경을 구분하기.

[전체 학습 순서]({{< relref "learn" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- LEARNING-END-END -->
