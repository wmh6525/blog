---
title: "[논문 리뷰] Mamba: 선택적 상태 공간 모델로 Transformer에 도전하다"
date: 2026-03-19
tags: ["논문리뷰", "Mamba", "SSM", "State Space Model"]
categories: ["ML/AI"]
summary: "Mamba 논문 리뷰. 입력 의존적 상태 전이(Selective SSM)로 Transformer의 어텐션 메커니즘 없이 동등한 성능을 달성한다. 선형 시간 복잡도와 하드웨어 친화적 설계가 핵심이다."
math: true
toc: true
draft: false
---

<!-- LEARNING-START -->
## 처음 읽는 분을 위한 안내

**학습 위치:** [문장을 읽으며 기억하는 모델]({{< relref "learn" >}}#memory) · 4/13

요약 메모를 고칠 때 모든 입력을 똑같이 취급하면 중요한 내용과 덜 중요한 내용을 구분하기 어렵다. Mamba는 입력 내용에 따라 상태를 갱신하는 방식을 조정한다. 지금 들어온 내용이 기억에 어떻게 반영되는지 먼저 살펴보자.

**먼저 알아둘 말**

- [상태·SSM]({{< relref "ai-glossary" >}}#state): 상태는 이전까지 읽은 정보를 다음 단계에 전달하는 내부 값이다. SSM은 이 상태가 입력에 따라 바뀌고 출력을 만드는 관계를 모델링한다.
- [시퀀스]({{< relref "ai-glossary" >}}#sequence): 순서가 있는 데이터. 문장의 토큰이나 시간에 따라 들어오는 관측처럼 앞뒤 순서가 계산에 영향을 주는 입력이다.
- [학습과 추론]({{< relref "ai-glossary" >}}#training): 학습은 데이터로 모델 안의 숫자를 바꾸는 일이고, 추론은 모델에 입력을 넣어 결과를 얻는 일이다.

**이번 글의 목표:** 입력에 따라 기억 갱신이 달라지는 이유 이해하기.

**이 학습 경로의 앞 글:** [\[논문 리뷰\] HiPPO: 장거리 의존성 문제의 수학적 해결 — SSM과 Mamba의 이론적 기반]({{< relref "posts/hippo-review" >}}).

토큰과 벡터가 낯설다면 [텍스트 데이터 다루기]({{< relref "posts/build-llm-ch2-working-with-text-data" >}})의 설명부터 읽자.

**본문에서 먼저 볼 순서:** **1. 배경: SSM의 한계와 Mamba의 등장** → **2. Mamba의 핵심: 선택적 상태 전이** → **4. Mamba 블록 아키텍처**.

첫 독서에서는 문제와 방법의 흐름을 잡자. 수식·코드·세부 성능표는 두 번째에 읽어도 된다. 구현과 수학적 주장까지 확인하려면 그 부분을 다시 살펴봐야 한다.

<!-- PAPER-PREREQUISITES -->
### 선수 개념과 읽을 논문

**먼저 이해할 개념:** 벡터·행렬, 상태의 재귀 갱신, 연속과 이산 시간; 수식 독서에는 미분과 선형대수.

아래 순서는 이 글을 이해하기 위한 공부 순서다. ‘개념 준비’는 읽기 안내이며, 직접적인 연구 계보는 연결 칸에 따로 표시했다. 이미 해당 개념을 설명할 수 있다면 아는 부분은 넘어가도 된다.

| 순서·논문 | 연결·익힐 개념 | 먼저 볼 부분 |
|---|---|---|
| 1. [HiPPO: Recurrent Memory with Optimal Polynomial Projections](https://arxiv.org/abs/2008.07669) · [블로그 해설]({{< relref "posts/hippo-review" >}}) | **기억 표현의 선행 개념**: 상태·과거 압축·다항식 투영 | 기억을 계수로 저장하고 새 입력으로 갱신하는 문제 설정 |
| 2. [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396) | **구조화 SSM의 선행 연구**: SSM·이산화·재귀와 컨볼루션 | 상태 공간식과 긴 시퀀스를 효율적으로 계산하는 구조 |

**현재 글과 연결하기:** 위 개념을 익힌 뒤, 이 글에서 ‘입력에 따라 기억 갱신이 달라지는 이유 이해하기’에 해당하는 부분을 읽자. 기존 방법의 어느 단계가 바뀌었는지 그림이나 입력·출력으로 설명해 보자.

[선수 논문 전체 지도와 원문 읽는 법]({{< relref "paper-prerequisites" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- PAPER-PREREQUISITES-END -->

<!-- LEARNING-START-END -->

---

## 논문 정보

- **제목**: Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- **저자**: Albert Gu, Tri Dao
- **발표**: ICLR 2024 (arXiv: 2312.00752)
- **키워드**: State Space Model, Selective Mechanism, Linear Complexity

---

## 1. 배경: SSM의 한계와 Mamba의 등장

### State Space Model (SSM)이란?

연속 시간 시스템을 이산화한 시퀀스 모델:

$$h'(t) = A \cdot h(t) + B \cdot x(t)$$

$$y(t) = C \cdot h(t)$$

- $h(t)$: 은닉 상태 (state)
- $A$: 상태 전이 행렬
- $B$: 입력 투영
- $C$: 출력 투영

이산화 후:

$$h_t = \bar{A} \cdot h_{t-1} + \bar{B} \cdot x_t$$

$$y_t = C \cdot h_t + D \cdot x_t$$

### 기존 SSM의 문제: LTI (Linear Time-Invariant)

S4 등 기존 SSM의 $A$, $B$, $C$는 학습 후 **고정**된다. 모든 입력에 같은 상태 전이 규칙을 적용하는 셈이다.

이 LTI 속성 덕분에 **컨볼루션으로 변환**해 학습 시 병렬화가 가능했지만 동시에 **내용 기반 추론(content-based reasoning)**이 불가능했다.

> 핵심 한계: "이 토큰이 중요한가?"를 판단할 수 없다.

---

## 2. Mamba의 핵심: 선택적 상태 전이

Mamba의 혁신은 $B$, $C$, $\Delta$를 **입력 의존적**으로 만든 것이다:

$$B(t) = \text{Linear}_B(x(t))$$

$$C(t) = \text{Linear}_C(x(t))$$

$$\Delta(t) = \text{softplus}(\text{Linear}_\Delta(x(t)))$$

이산화:

$$\bar{A}(t) = \exp(\Delta(t) \cdot A)$$

$$\bar{B}(t) = (\Delta(t) \cdot A)^{-1} (\exp(\Delta(t) \cdot A) - I) \cdot \Delta(t) \cdot B(t)$$

상태 전이:

$$h(t) = \bar{A}(t) \cdot h(t-1) + \bar{B}(t) \cdot x(t)$$

$$y(t) = C(t) \cdot h(t) + D \cdot x(t)$$

### $\Delta$의 의미: 선택적 메모리

$\Delta(t)$는 **매 토큰마다 다른 시간 스케일**을 적용한다:

- **$\Delta$ 큼** → $\bar{A}(t) \approx 0$ → 이전 상태를 **잊고** 새 입력을 강하게 반영
- **$\Delta$ 작음** → $\bar{A}(t) \approx I$ → 이전 상태를 **보존**하고 새 입력을 약하게 반영

이것이 Transformer의 어텐션과 유사한 **내용 기반 선택** 메커니즘이다:
- 중요한 토큰 → $\Delta$ 크게 → 상태에 강하게 기록
- 불필요한 토큰 → $\Delta$ 작게 → 상태 유지, 입력 무시

선택적이라는 말은 입력 내용에 따라 상태에 넣고 남길 정보를 달리한다는 뜻이다. 모든 위치에 같은 갱신 규칙을 적용하던 SSM에서 입력 의존성을 더한 것이다. 다만 과거 정보는 계속 상태 벡터 안에 담긴다. 아래 선택 메커니즘과 하드웨어 설계는 이 입력 의존적 갱신을 어떻게 계산하고 효율적으로 실행하는지에 각각 대응한다.

---

## 3. 하드웨어 친화적 설계

### 문제: 선택적 SSM은 컨볼루션 불가

LTI SSM은 $A$, $B$, $C$가 고정이므로 글로벌 컨볼루션으로 변환해 $O(L \log L)$로 계산할 수 있었다. 그러나 입력 의존적 파라미터는 이 트릭을 깨뜨린다.

### 해결: Parallel Scan + Kernel Fusion

Mamba는 **parallel scan (prefix sum)** 알고리즘으로 순환 연산을 병렬화한다:

1. **커널 퓨전**: 이산화, 상태 전이, 출력 계산을 하나의 GPU 커널로 통합
2. **SRAM 활용**: HBM ↔ SRAM 간 IO를 최소화
3. **순환 모드**: 추론 시 $O(1)$ per step (상태만 유지)

결과적으로:
- **학습**: $O(L)$ 시간, FlashAttention과 유사한 IO 효율
- **추론**: $O(1)$ per token (RNN 모드), Transformer의 KV cache 없음

---

## 4. Mamba 블록 아키텍처

```
Input (B, L, D)
    │
    ├── Linear (D → 2·E)  ──→ SiLU ──→ Conv1d ──→ SSM ──→ ×
    │                                                        │
    └── Linear (D → 2·E)  ──→ SiLU ─────────────────────→ gate
                                                             │
                                                         Linear (E → D)
                                                             │
                                                          Output
```

핵심 구성:
- **Expand**: 입력 차원을 확장 (보통 2배)
- **Conv1d**: 짧은 범위의 로컬 패턴 포착
- **Selective SSM**: 장거리 의존성 + 선택적 메모리
- **Gated connection**: 정보 흐름 조절

---

## 5. 실험 결과

### 언어 모델링

| 모델 | Params | Pile PPL |
|------|--------|----------|
| Transformer++ | 1.4B | - |
| RWKV-4 | 1.5B | - |
| H3 | 1.3B | - |
| **Mamba** | **1.4B** | **최고 성능** |

Mamba-1.4B가 Transformer++ 2.8B와 동등한 성능을 달성. 즉 **절반의 파라미터로 동일 품질**.

### Selective Copying Task

LTI SSM이 실패하는 대표적 과제. 시퀀스 중간에 있는 특정 토큰만 기억하고 나머지는 무시해야 한다.

| 모델 | 정확도 |
|------|-------|
| S4 (LTI) | 실패 |
| H3 | 부분 성공 |
| **Mamba** | **완벽** |

### 추론 속도 (Throughput)

Mamba-1.4B의 추론 throughput은 Transformer++의 **5배** 수준. 시퀀스 길이가 길어질수록 격차가 커진다 (선형 vs 이차).

---

## 6. 왜 Mamba가 중요한가?

### Transformer 대비 장점

| 항목 | Transformer | Mamba |
|------|------------|-------|
| 시간 복잡도 | $O(L^2)$ | $O(L)$ |
| 추론 per token | $O(L)$ (KV cache) | $O(1)$ |
| 메모리 (추론) | KV cache 증가 | 고정 크기 state |
| 긴 시퀀스 | 비용 급증 | 선형 확장 |
| 하드웨어 효율 | IO bound | Kernel fusion |

### Attention과의 관계

Mamba의 selective mechanism은 사실 **소프트 어텐션의 일반화**로 볼 수 있다:
- Attention: 모든 이전 토큰에 대해 가중합 (명시적)
- Mamba: 상태 벡터에 선택적으로 정보 압축 (암묵적)

차이: Attention은 과거 전체에 접근 가능하지만 비용이 $O(L^2)$. Mamba는 고정 크기 상태로 압축하므로 $O(L)$이지만, 정보 손실 가능성이 있다.

---

## 7. 후속 연구

### Mamba-2 (2024)

- Structured State Space Duality (SSD): SSM = structured masked attention임을 증명
- 더 효율적인 병렬 알고리즘
- Mamba-1 대비 2-8배 빠른 학습

### Jamba (AI21, 2024)

- Transformer + Mamba 하이브리드
- Attention 레이어와 Mamba 레이어를 교차 배치
- 52B 파라미터, 256K 컨텍스트

### Vision Mamba, Video Mamba 등

- 이미지/비디오 도메인으로 확장
- ViT 대비 선형 복잡도로 고해상도 처리

---

## 8. 한계와 열린 질문

1. **In-context learning**: Transformer 대비 ICL 능력이 다소 약하다는 보고
2. **정보 압축 손실**: 고정 크기 상태로는 모든 과거 정보를 완벽히 보존 불가
3. **하이브리드의 필요성**: Jamba처럼 Attention과 결합할 때 최고 성능
4. **학습 안정성**: 대규모(70B+)에서의 학습 안정성 추가 검증 필요

---

## 9. 개인적 생각

Mamba는 "Attention is All You Need" 이후 가장 임팩트 있는 아키텍처 논문 중 하나다. 특히:

- **$\Delta$를 통한 선택적 메모리** 아이디어가 깔끔하다. "무엇을 기억하고 무엇을 잊을지"를 입력 자체가 결정하게 한 것은 생물학적 뉴런의 게이팅과도 유사하다.
- **하드웨어 수준의 최적화**를 논문에서 직접 다룬 것이 실용적이다.
- SSM → Selective SSM의 전환이 LTI → LTV (Linear Time-Varying)라는 오래된 제어이론 개념의 재발견이라는 점이 흥미롭다.

다만 순수 Mamba만으로는 Transformer를 완전히 대체하기 어렵고 **하이브리드 아키텍처**가 현실적 방향이라는 것이 현재까지의 합의인 것 같다.

바이트 수준 언어 모델(SancMamba)의 기반으로 Mamba를 선택한 이유도 여기에 있다 — 바이트 시퀀스는 토큰 시퀀스보다 3~6배 길어지므로, 선형 복잡도가 필수적이다.

<!-- LEARNING-END -->
---

## 읽은 뒤 확인하고 다음으로

**확인 질문:** 입력에 따라 기억 갱신이 달라지는 이유 이해하기를 목표로, 본문에서 다룬 입력·중간 과정·결과를 자기 말로 설명해 보자. 실험이나 사례가 있다면 어떤 조건에서 확인했는지도 덧붙여 보자.

**다음 글:** [\[논문 리뷰\] Mamba-2: SSM과 Attention의 이중성을 밝히다]({{< relref "posts/mamba2-review" >}}) — 같은 계산을 순서대로 하거나 묶어서 하는 관점 이해하기.

[전체 학습 순서]({{< relref "learn" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- LEARNING-END-END -->
