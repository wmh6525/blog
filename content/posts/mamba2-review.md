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

## 논문 정보

- **제목**: Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality
- **저자**: Tri Dao, Albert Gu
- **발표**: ICML 2024 (arXiv: 2405.21060)
- **키워드**: State Space Duality, Structured Attention, Semi-Separable Matrices

---

## 1. 핵심 발견: SSM = Structured Attention

Mamba-2의 가장 중요한 기여는 **이론적 통합**이다.

> **SSM과 어텐션은 본질적으로 동일한 연산의 다른 표현이다.**

이를 **Structured State Space Duality (SSD)**라 부른다.

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
