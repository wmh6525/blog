---
title: "SancMamba: 바이트 수준 계층적 시퀀스 압축 언어 모델 설계기 (1)"
date: 2026-03-18
tags: ["연구노트", "SancMamba", "Mamba", "SSM", "BLT"]
categories: ["ML/AI"]
summary: "토크나이저 없이 바이트에서 출발해 의미 단위를 스스로 발견하는 언어 모델을 설계하기까지 — Forward-Forward부터 Mamba, SANC(E3), BLT를 거쳐 SancMamba 프로토타입 구현과 의미 단위 청킹 문제 발견까지의 여정."
math: true
toc: true
---

## 동기

대부분의 LLM은 BPE 같은 토크나이저에 의존한다. 토크나이저는 학습 데이터의 통계에 기반해 고정된 서브워드 단위를 만드는데, 이 고정 단위가 모델이 "의미"를 다루는 최소 해상도를 결정한다. 한국어처럼 교착어인 경우 형태소 경계와 BPE 토큰 경계가 불일치하는 문제가 빈번하다.

**SancMamba**는 이 고정 토큰화를 없애고, **바이트(UTF-8)에서 출발해 모델 스스로 의미 단위를 계층적으로 발견**하게 하려는 시도다.

---

## 1. 선행 연구 조사

### 1-1. Forward-Forward Algorithm

역전파(BP) 대신 두 번의 forward pass(positive/negative)로 레이어별 독립 학습하는 Hinton의 방법론을 조사했다. 각 레이어의 local loss는:

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

해석 가능성과 과학적 태스크 스케일링에서 장점이 있지만, MLP 대비 ~10x 느리고 LLM 규모 미검증이라 현 단계에서는 미채택.

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

- $L_{rec} = \|e - G(\Sigma_t)\|^2$ : 재구성 오차
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

별도 파라미터 없이 SSM의 부산물을 재활용하는 우아한 설계지만, 여기에 근본적인 문제가 있었다.

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

$g(t) = \sigma(\text{boundary\_head}(h(t)))$로, $g \to 1$이면 이전 상태를 거의 잊고 새 의미 단위를 시작한다.

---

## 다음 단계

1. SSM 상태 전이 수정 방향 확정 (A/B/C)
2. SemanticChunkModule 통합
3. A100에서 학습 테스트 및 청킹 시각화 검증
4. Stage별 경계가 형태소/어절/문장 수준과 일치하는지 확인

---

*이 시리즈는 SancMamba 모델의 설계, 구현, 실험 과정을 기록합니다.*
