---
title: "SancMamba: 바이트 수준 계층적 시퀀스 압축 언어 모델 설계기 (1)"
date: 2026-03-18
tags: ["연구노트", "SancMamba", "Mamba", "SSM", "BLT"]
categories: ["ML/AI"]
summary: "토크나이저 없이 바이트에서 출발해 의미 단위를 스스로 발견하는 언어 모델을 설계하기까지 — Forward-Forward부터 Mamba, SANC(E3), BLT를 거쳐 SancMamba 프로토타입 구현과 의미 단위 청킹 문제 발견까지의 여정."
math: true
toc: true
---

<!-- LEARNING-START -->
## 처음 읽는 분을 위한 안내

**학습 위치:** [문장을 읽으며 기억하는 모델]({{< relref "learn" >}}#memory) · 12/13

글을 일정 길이로 자르는 것과 뜻이 바뀌는 경계에서 자르는 것은 다를 수 있다. 이 프로젝트는 더 작은 입력 단위에서 출발해 경계를 배우려는 시도다. 잘 묶였는지를 청크 길이만으로 판단할 수 있는지 생각하며 실패 사례를 읽자.

**먼저 알아둘 말**

- [바이트]({{< relref "ai-glossary" >}}#byte): 컴퓨터가 데이터를 저장하는 단위. 글자를 바이트로 읽는 모델은 고정된 단어 조각 대신 더 작은 데이터 단위에서 출발한다.
- [패치·청크]({{< relref "ai-glossary" >}}#patch): 작은 단위 여러 개를 한 묶음으로 만든 것. 이미지 패치, 바이트 패치, 문서 청크는 서로 다른 대상을 묶는다.
- [상태·SSM]({{< relref "ai-glossary" >}}#state): 상태는 이전까지 읽은 정보를 다음 단계에 전달하는 내부 값이다. SSM은 이 상태가 입력에 따라 바뀌고 출력을 만드는 관계를 모델링한다.

**이번 글의 목표:** 문자 경계와 의미 경계가 같은지 확인하는 실험 이해하기.

**이 학습 경로의 앞 글:** [\[서베이\] 해마를 어떻게 구현할 것인가 — attention 옆에 기억을 따로 두는 방법들]({{< relref "posts/hippocampal-memory-attention-survey" >}}).

토큰과 벡터가 낯설다면 [텍스트 데이터 다루기]({{< relref "posts/build-llm-ch2-working-with-text-data" >}})의 설명부터 읽자.

**본문에서 먼저 볼 순서:** **동기** → **2. Mamba SSM 핵심 메커니즘** → **4. SancMamba 아키텍처**.

첫 독서에서는 문제와 방법의 흐름을 잡자. 수식·코드·세부 성능표는 두 번째에 읽어도 된다. 구현과 수학적 주장까지 확인하려면 그 부분을 다시 살펴봐야 한다.

<!-- PAPER-PREREQUISITES -->
### 선수 개념과 읽을 논문

**먼저 이해할 개념:** 벡터·행렬, 상태의 재귀 갱신, 연속과 이산 시간; 수식 독서에는 미분과 선형대수.

아래 순서는 이 글을 이해하기 위한 공부 순서다. ‘개념 준비’는 읽기 안내이며, 직접적인 연구 계보는 연결 칸에 따로 표시했다. 이미 해당 개념을 설명할 수 있다면 아는 부분은 넘어가도 된다.

| 순서·논문 | 연결·익힐 개념 | 먼저 볼 부분 |
|---|---|---|
| 1. [The Forward-Forward Algorithm: Some Preliminary Investigations](https://arxiv.org/abs/2212.13345) · [블로그 해설]({{< relref "posts/forward-forward-review" >}}) | **개념 준비**: positive/negative 입력·층별 goodness | 두 forward pass와 층별 학습 목표 |
| 2. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) · [블로그 해설]({{< relref "posts/mamba-ssm-review" >}}) | **개념 준비**: 입력 의존 상태 갱신·선택 메커니즘 | 입력에 따라 바뀌는 값과 selective scan의 역할 |
| 3. [Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2412.09871) · [블로그 해설]({{< relref "posts/blt-byte-level-lm-survey" >}}) | **개념 준비**: 엔트로피 패칭·바이트 인코더/디코더 | 경계를 정하고 묶음을 백본에 넘기는 흐름 |

**현재 글과 연결하기:** 위 개념을 익힌 뒤, 이 글에서 ‘문자 경계와 의미 경계가 같은지 확인하는 실험 이해하기’에 해당하는 부분을 읽자. 기존 방법의 어느 단계가 바뀌었는지 그림이나 입력·출력으로 설명해 보자.

[선수 논문 전체 지도와 원문 읽는 법]({{< relref "paper-prerequisites" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- PAPER-PREREQUISITES-END -->

<!-- LEARNING-START-END -->

---

## 동기

대부분의 LLM은 BPE 같은 토크나이저에 의존한다. 토크나이저는 학습 데이터의 통계에 기반해 고정된 서브워드 단위를 만드는데, 이 고정 단위가 모델이 "의미"를 다루는 최소 해상도를 결정한다. 한국어처럼 교착어인 경우 형태소 경계와 BPE 토큰 경계가 불일치하는 문제가 빈번하다.

**SancMamba**는 이 고정 토큰화를 없애고 **바이트(UTF-8)에서 출발해 모델 스스로 의미 단위를 계층적으로 발견**하게 하려는 시도다.

---

## 1. 선행 연구 조사

### 1-1. Forward-Forward Algorithm

역전파(BP) 대신 두 번의 forward pass(positive/negative)로 각 레이어를 독립적으로 학습하는 Hinton의 방법론을 조사했다. 레이어별 local loss는 다음과 같다.

$$\mathcal{L} = \log(1 + e^{-(g_{pos} - \theta)}) + \log(1 + e^{g_{neg} - \theta})$$

여기서 $g$는 "goodness" (hidden 활성값의 제곱합), $\theta$는 임계값이다.

**LM 적용 한계**: 자기회귀 생성에서는 다음 토큰이 확률적이라 positive/negative 경계가 모호하다. 다만 RAG 임베딩의 triplet 구조(query, relevant, irrelevant)에는 자연스럽게 적용 가능함을 확인했다.

최신 연구인 **CLAPP++** (EPFL/TU Wien, 2026.01)는 local-SSL이 deep linear network에서 BP-SSL과 동일한 weight update를 수행할 수 있음을 이론적으로 증명했다.

### 1-2. No-BP 학습 방법론 서베이

| 방법론 | 핵심 | SancMamba 적용성 |
|--------|------|-----------------|
| **MeZO** | Zeroth-order optimizer, 12x 메모리 절감 | 파인튜닝에 적합, 사전학습 부적합 |
| **Predictive Coding** | Local energy minimization | 소규모 CNN에서만 검증 |
| **Evolution Strategies** | Fully BP-free, 고병렬 | RLHF 대안으로 LLM 파인튜닝 성공 |
| **EBT** | 에너지 기반 반복 추론 | Transformer++ 대비 35% 나은 스케일링 |

### 1-3. KAN (Kolmogorov-Arnold Networks)

MLP의 구조적 대안으로 조사했다. 엣지에 학습 가능한 B-Spline 함수를 두는 아키텍처 혁신이다 (No-BP 방법론이 아니라 **일반 역전파로 학습**).

$$\phi(x) = w_b \cdot \text{SiLU}(x) + w_s \cdot \text{Spline}(x)$$

해석 가능성과 과학적 태스크 스케일링에서 장점이 있지만 MLP 대비 ~10x 느리고 LLM 규모 미검증이라 현 단계에서는 미채택.

---

## 2. Mamba SSM 핵심 메커니즘

SancMamba의 기반 아키텍처인 Mamba의 핵심은 **입력 의존적 상태 전이**이다.

기존 SSM은 A, B, C가 학습 후 고정이었다. Mamba는 이를 **매 토큰마다 다르게** 만들었다:

$$B(t) = \text{Linear}_B(x(t)), \quad C(t) = \text{Linear}_C(x(t))$$

$$\Delta(t) = \text{softplus}(\text{Linear}_\Delta(x(t)))$$

이산화 후 상태 전이:

$$h(t) = \bar{A}(t) \cdot h(t-1) + \bar{B}(t) \cdot x(t)$$

$$y(t) = C(t) \cdot h(t) + D \cdot x(t)$$

여기서 $\bar{A}(t) = \exp(\Delta(t) \cdot A)$이다. **$\Delta$가 크면** 이전 상태를 많이 잊고 새 입력을 강하게 반영하고, **$\Delta$가 작으면** 상태를 보존한다. 이 선택적 메커니즘이 Mamba가 Transformer에 필적하는 성능을 내는 핵심이다.

---

## 3. SANC(E3) 프레임워크

SANC(E3) 논문 (arXiv:2601.08224)의 핵심은 표현 단위가 **자기조직화로 출현**한다는 것이다.

에너지 함수:

$$E_3 = \lambda_1 \cdot L_{rec} + \lambda_2 \cdot C_{struct} + \lambda_3 \cdot C_{update}$$

- $L_{rec} = \lVert e - G(\Sigma_t) \rVert^2$ : 재구성 오차
- $C_{struct} = |\Sigma_t| + H(\Sigma_t)$ : 구조적 복잡도
- $C_{update} = \sum|{\Delta w_{ij}}|$ : 갱신 비용

**게슈탈트 완성**: $G(A_{partial}) = \arg\min_{A \supseteq A_{partial}} E_3(A)$

모든 인지 활동(지각, 예측, 계획)이 $E_3$ 최소화의 인스턴스라는 통합적 관점이 SancMamba의 "데이터에서 의미 단위를 발견한다"는 설계 철학에 이론적 근거를 제공한다.

---

## 4. SancMamba 아키텍처

### 전체 파이프라인

```
Input bytes (B, L) → Embedding(256, 128)
       │
  Encoder Stage 0: MambaBlock → ChunkModule → proj_up(128→256)  — L→L₁ (×0.53)
  Encoder Stage 1: MambaBlock → ChunkModule → proj_up(256→384)  — L₁→L₂
  Encoder Stage 2: MambaBlock → ChunkModule → proj_up(384→512)  — L₂→L₃
  Encoder Stage 3: MambaBlock → ChunkModule                     — L₃→L₄
       │  총 압축: ~8.5% (64바이트 → ~6토큰)
       │
  Concept Block: MambaBlock×2 + CausalSelfAttention×1
       │
  Dechunk Stage 0~3: EMA scan → causal gather → Encoder residual + STE
       │  (U-Net 구조로 원본 길이 복원)
       │
  Decoder: MambaBlock×1 → LM Head(128, 256) → logits
```

### ChunkModule — Δ 기반 경계 탐지

각 Stage의 MambaBlock이 시퀀스를 처리할 때 계산하는 $\Delta(t)$를 재활용한다:

```python
delta_diff[t] = |Δ(t) - Δ(t-1)|   # Δ 변화량
boundary_prob = sigmoid(proj(hidden) + delta_diff × 2.0)
boundary_mask = Bernoulli(sharpen(boundary_prob))  # 학습 시
chunked = scatter_mean(hidden, cumsum(boundary_mask))
```

별도 파라미터 없이 SSM의 부산물을 재활용하는 우아한 설계지만 여기에 근본적인 문제가 있었다.

---

## 5. 발견한 문제: 의미 단위 청킹 실패

학습 결과의 압축 시각화를 분석하고 나서 핵심 문제를 발견했다.

### 증상

한국어 "고양이가"(UTF-8: 12바이트)가 글자 단위로 묶이지 않고 바이트 중간에서 잘렸다:

```
원하는 결과: [고양이] [가] [숲에서]    ← 형태소 단위
실제 결과:   [EA B3] [A0 EC 96] [91 EC 9D B4 ...]  ← 바이트 중간에서 절단
```

### 원인 분석 — 3가지 근본 원인

**1) Δ는 local signal**: 인접 두 위치만 비교해서 "지금까지 쌓인 의미 vs 새 의미"의 전환을 포착하지 못한다.

**2) 바이트 구조 무지**: UTF-8 continuation 바이트(0x80-0xBF)가 경계가 될 수 없다는 구조를 모른다. 한국어 한 글자가 3바이트인데, lead 바이트(0xE0-0xEF)에서 $\Delta$가 크고 continuation에서 작아지면서 글자 내부에서 경계가 잡힌다.

**3) Supervision 부재**: Compression loss("경계를 줄여라")만 있고, "의미 단위로 묶어라"는 신호가 없다.

### 더 근본적 문제 — Δ의 역할 충돌

$\Delta(t)$가 **두 역할**을 동시에 담당한다:
- 본래 역할: CE loss를 위한 SSM 상태 전이율 조절
- 추가 역할: 청크 경계 신호

하나의 파라미터가 두 목적을 동시 최적화 → 둘 다 차선이 된다.

경계가 생겼다는 사실만으로 의미 단위를 찾았다고 보기는 어렵다. 바이트를 일정 길이로 나눈 결과와 의미에 따라 묶은 결과는 겉으로 비슷한 청크 수를 만들 수도 있다. 그래서 이 단계에서는 청크 길이의 분포와 실제 경계가 놓인 위치를 함께 살펴본다. 아래 해결 방향도 압축이 일어나는지와 원하는 경계가 학습되는지를 구분해 확인하는 데 초점을 둔다.

---

## 6. 해결 방향

### 즉시 적용 — UTF-8 경계 강제

Stage 0의 ChunkModule에 한 줄 추가로 글자 파괴를 방지:

```python
if 0x80 <= input_ids[t] <= 0xBF:
    boundary_prob[t] = 0.0  # continuation 바이트는 절대 경계 아님
```

### 근본 해결 — SSM 상태 전이 수정 (3가지 방향)

| 방향 | 핵심 | 장단점 |
|------|------|--------|
| **A** | SSM 유지, 경계 탐지만 분리 | 안전하지만 SSM이 청크 모름 |
| **B** | 상태 전이에 boundary gate 추가 | 한 줄 변경, SSM이 청크 인식 |
| **C** | Content + Boundary 2-track SSM | 완전 분리, 복잡도 증가 |

**방향 B**가 가장 유력하다:

$$h(t) = (1 - g(t)) \cdot [\bar{A}(t) \cdot h(t-1)] + \bar{B}(t) \cdot x(t)$$

$g(t) = \sigma(\text{boundary_head}(h(t)))$로, $g \to 1$이면 이전 상태를 거의 잊고 새 의미 단위를 시작한다.

---

## 다음 단계

1. SSM 상태 전이 수정 방향 확정 (A/B/C)
2. SemanticChunkModule 통합
3. A100에서 학습 테스트 및 청킹 시각화 검증
4. Stage별 경계가 형태소/어절/문장 수준과 일치하는지 확인

---

*이 시리즈는 SancMamba 모델의 설계, 구현, 실험 과정을 기록합니다.*

<!-- LEARNING-END -->
---

## 읽은 뒤 확인하고 다음으로

**확인 질문:** 문자 경계와 의미 경계가 같은지 확인하는 실험 이해하기를 목표로, 본문에서 다룬 입력·중간 과정·결과를 자기 말로 설명해 보자. 실험이나 사례가 있다면 어떤 조건에서 확인했는지도 덧붙여 보자.

**다음 글:** [Hymba-SANC: 유동 청킹 기반 개념 토큰 언어 모델 구현 가이드]({{< relref "posts/hymba-sanc-implementation-guide" >}}) — 청킹·기억·상위 예측을 모듈로 연결하는 설계 따라가기.

[전체 학습 순서]({{< relref "learn" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- LEARNING-END-END -->
