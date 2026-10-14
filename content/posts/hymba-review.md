---
title: "[논문 리뷰] Hymba: SSM과 Attention을 헤드 수준에서 융합한 하이브리드 아키텍처"
date: 2026-03-20
tags: ["논문리뷰", "Mamba", "SSM", "Hymba", "하이브리드"]
categories: ["ML/AI"]
summary: "NVIDIA Hymba 논문 리뷰. SSM과 Attention을 레이어가 아닌 헤드 수준에서 융합하고, 학습 가능한 메타 토큰으로 글로벌 메모리를 도입한다. Hymba-1.5B가 Llama-3.2-3B를 능가하면서 캐시는 ~20%만 사용한다."
math: true
toc: true
draft: false
---

<!-- LEARNING-START -->
## 처음 읽는 분을 위한 안내

**학습 위치:** [문장을 읽으며 기억하는 모델]({{< relref "learn" >}}#memory) · 10/13

과거를 요약한 상태를 쓰는 방법과 필요한 위치를 다시 참고하는 방법을 함께 사용할 수 있다. Hymba는 두 계산 경로를 같은 블록 안에서 결합한다. 두 경로가 무엇을 맡고 어디서 결과가 합쳐지는지 표시하며 읽어 보자.

**먼저 알아둘 말**

- [상태·SSM]({{< relref "ai-glossary" >}}#state): 상태는 이전까지 읽은 정보를 다음 단계에 전달하는 내부 값이다. SSM은 이 상태가 입력에 따라 바뀌고 출력을 만드는 관계를 모델링한다.
- [Attention]({{< relref "ai-glossary" >}}#attention): 현재 계산에 필요한 다른 위치의 정보를 가중치를 주어 섞는 장치. 가중치가 크다는 사실만으로 판단의 원인을 확정할 수는 없다.
- [메모리·키·값]({{< relref "ai-glossary" >}}#memory): 메모리는 정보를 저장하고 다시 읽는 구조다. 키는 항목을 찾는 단서, 값은 그 항목에서 꺼내는 내용으로 생각하면 된다.

**이번 글의 목표:** 요약 기억과 Attention을 함께 쓰는 구조 이해하기.

**이 학습 경로의 앞 글:** [\[연구 분석\] Selective SSM만으로는 왜 거대 언어모델이 될 수 없는가?]({{< relref "posts/why-pure-ssm-cannot-be-llm" >}}).

토큰과 벡터가 낯설다면 [텍스트 데이터 다루기]({{< relref "posts/build-llm-ch2-working-with-text-data" >}})의 설명부터 읽자.

**본문에서 먼저 볼 순서:** **1. 동기: SSM과 Attention, 각자의 한계** → **2. Hymba의 핵심: 헤드 내 융합** → **4. 아키텍처 세부 사항**.

첫 독서에서는 문제와 방법의 흐름을 잡자. 수식·코드·세부 성능표는 두 번째에 읽어도 된다. 구현과 수학적 주장까지 확인하려면 그 부분을 다시 살펴봐야 한다.

<!-- PAPER-PREREQUISITES -->
### 선수 개념과 읽을 논문

**먼저 이해할 개념:** 벡터·행렬, 상태의 재귀 갱신, 연속과 이산 시간; 수식 독서에는 미분과 선형대수.

아래 순서는 이 글을 이해하기 위한 공부 순서다. ‘개념 준비’는 읽기 안내이며, 직접적인 연구 계보는 연결 칸에 따로 표시했다. 이미 해당 개념을 설명할 수 있다면 아는 부분은 넘어가도 된다.

| 순서·논문 | 연결·익힐 개념 | 먼저 볼 부분 |
|---|---|---|
| 1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | **개념 준비**: Attention·위치 정보·인코더/디코더 | 현재 토큰이 다른 위치의 정보를 읽는 구조 그림과 Attention 설명 |
| 2. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) · [블로그 해설]({{< relref "posts/mamba-ssm-review" >}}) | **개념 준비**: 입력 의존 상태 갱신·선택 메커니즘 | 입력에 따라 바뀌는 값과 selective scan의 역할 |
| 3. [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) · [블로그 해설]({{< relref "posts/mamba2-review" >}}) | **개념 준비**: 구조화된 Attention·상태 공간 이중성·청크 계산 | 재귀와 행렬 표현의 대응, 청크 안과 청크 사이 계산 |

**현재 글과 연결하기:** 위 개념을 익힌 뒤, 이 글에서 ‘요약 기억과 Attention을 함께 쓰는 구조 이해하기’에 해당하는 부분을 읽자. 기존 방법의 어느 단계가 바뀌었는지 그림이나 입력·출력으로 설명해 보자.

[선수 논문 전체 지도와 원문 읽는 법]({{< relref "paper-prerequisites" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- PAPER-PREREQUISITES-END -->

<!-- LEARNING-START-END -->

---

## 논문 정보

- **제목**: Hymba: A Hybrid-head Architecture for Small Language Models
- **저자**: Xin Dong, Yonggan Fu, Shizhe Diao, Wonmin Byeon, Zijia Chen, Ameya Sunil Mahabaleshwarkar, Shih-Yang Liu, Matthijs Douze, Zangwei Zheng, Jan Kautz, Pavlo Molchanov
- **소속**: NVIDIA
- **발표**: 2024.11 (arXiv: 2411.13676)
- **키워드**: Hybrid Architecture, SSM-Attention Fusion, Meta Tokens, Small Language Models

---

## 1. 동기: SSM과 Attention, 각자의 한계

### SSM과 Attention의 상보성

| 능력 | SSM (Mamba) | Attention |
|------|------------|-----------|
| 장거리 의존성 | 상태 압축 (정보 손실 가능) | **완전 접근** |
| 계산 복잡도 | **$O(L)$** | $O(L^2)$ |
| In-context learning | 약함 | **강함** |
| 추론 효율 | **$O(1)$ per token** | $O(L)$ (KV cache) |
| 로컬 패턴 | 보통 | 보통 |

두 방식에는 각각 한계가 있다. Hymba는 **SSM의 효율성**과 **Attention의 표현력**을 결합해 이를 보완한다.

### 기존 하이브리드의 한계

Jamba(AI21), Griffin(Google) 등 기존 하이브리드는 SSM 레이어와 Attention 레이어를 **교차 배치**한다.

```
[Mamba] → [Attention] → [Mamba] → [Attention] → ...
```

이 방식의 문제:
- 같은 입력에 대해 SSM 관점과 Attention 관점이 **다른 레이어에서 분리**되어 처리
- 두 관점의 정보가 합류하려면 다음 레이어까지 기다려야 함
- 최적의 SSM/Attention 레이어 배치 비율을 찾기 어려움

---

## 2. Hymba의 핵심: 헤드 내 융합

### 같은 레이어에서 SSM과 Attention 동시 수행

Hymba는 **하나의 레이어 안에서** SSM과 Attention을 동시에 수행하고 결합한다:

```
Input
  ├── SSM heads (일부 헤드)
  │     └── Mamba-2 SSD 연산
  └── Attention heads (나머지 헤드)
        └── 표준 causal attention

  → 두 출력을 concat/합산 → Output
```

이것은 Multi-Head Attention에서 각 헤드가 다른 subspace를 학습하는 것과 유사하되, 일부 헤드는 **SSM으로**, 나머지는 **Attention으로** 작동한다.

### 장점

- SSM 헤드가 **효율적 장거리 문맥 압축** 담당
- Attention 헤드가 **정밀한 토큰 간 관계** 포착
- 같은 레이어에서 두 관점이 동시에 작동 → **정보 흐름이 풍부**
- 레이어 교차보다 **세밀한 통합** 가능

레이어를 번갈아 놓는 하이브리드에서는 입력이 SSM과 Attention을 순서대로 지난다. Hymba의 헤드 수준 결합에서는 같은 블록 안에서 두 경로의 출력을 함께 다룬다. 아래 구조를 읽을 때는 경로별로 무엇을 기억하는지와 두 출력이 어디서 합쳐지는지를 나눠 보면 된다. 메타 토큰과 캐시 공유는 그 기본 구조 위에 붙는 별도의 설계다.

---

## 3. Learnable Meta Tokens

Hymba의 두 번째 혁신은 **학습 가능한 메타 토큰**이다.

### 개념

입력 시퀀스 앞에 고정된 학습 가능 토큰을 추가한다. 이들은 KV cache에 포함되어:

- 모든 위치에서 접근 가능한 **글로벌 메모리** 역할
- 일종의 "항상 참조 가능한 요약 정보"

### Attention Sink 문제 해소

Transformer에서 관찰되는 **attention sink** 현상 — 모델이 첫 번째 토큰에 비정상적으로 높은 어텐션을 주는 문제 — 을 메타 토큰이 자연스럽게 흡수한다. 메타 토큰이 "기본 어텐션 대상"이 되어 의미 없는 첫 토큰 편향을 방지한다.

### 효과

- 적은 수(보통 128개)로도 유의미한 성능 개선
- 추가 파라미터/계산 비용 미미
- 학습 과정에서 자동으로 유용한 정보를 인코딩

---

## 4. 아키텍처 세부 사항

### 전체 구조

각 Hymba 레이어:

1. **입력 프로젝션**: $X \to Q, K, V$ (Attention용) + SSM 입력
2. **병렬 처리**:
   - SSM 헤드: Mamba-2의 SSD 알고리즘으로 처리
   - Attention 헤드: 표준 causal self-attention (+ meta tokens)
3. **출력 합산**: 두 종류의 헤드 출력을 concat → 프로젝션
4. **FFN**: SwiGLU 기반 feed-forward

### SSM 헤드 vs Attention 헤드 비율

Hymba는 SSM 헤드를 **다수**, Attention 헤드를 **소수**로 배치한다. 대부분의 시퀀스 처리는 SSM으로 충분하고 Attention은 ICL 등 정밀한 추론에만 필요하기 때문이다.

이 비율은 모델 크기와 용도에 따라 조정 가능하다.

---

## 5. 실험 결과

### 소규모 모델 성능 (1.5B)

| 모델 | Params | 평균 벤치마크 | Cache 크기 |
|------|--------|-------------|-----------|
| Llama-3.2 | 1B | 기준 | 100% |
| Mamba-2 | 1.3B | Llama-3.2 대비 열등 | 매우 작음 |
| **Hymba** | **1.5B** | **Llama-3.2-3B 능가** | **~20%** |

Hymba-1.5B가 **2배 큰** Llama-3.2-3B보다 높은 성능을 보인다.

### 캐시 효율

| 모델 | KV Cache (상대) |
|------|----------------|
| Transformer | 100% |
| Mamba (순수 SSM) | ~0% (상태만) |
| **Hymba** | **~20%** (Attention 헤드만) |

SSM 헤드는 KV cache가 불필요하므로, Attention 헤드 비율만큼만 캐시를 사용한다. 이는 긴 시퀀스에서 **메모리 절약**에 큰 이점이 된다.

### 벤치마크별 성능

Hymba는 다양한 벤치마크에서 일관된 우위를 보인다:

- **상식 추론**: SSM의 장거리 문맥 + Attention의 정밀 추론
- **수학/코드**: Attention 헤드가 정확한 토큰 매칭 담당
- **긴 문맥 이해**: SSM 헤드의 효율적 압축 + 메타 토큰의 글로벌 정보

---

## 6. 하이브리드 아키텍처 비교

| 모델 | 방식 | SSM 종류 | 특징 |
|------|------|---------|------|
| **Jamba** (AI21) | 레이어 교차 | Mamba-1 | MoE 결합, 52B 확장 |
| **Zamba** (Zyphra) | 공유 Attention | Mamba-1 | Attention 레이어 공유로 효율화 |
| **Griffin** (Google) | 게이트 결합 | Gated Linear Recurrence | RLHF까지 검증 |
| **Hymba** (NVIDIA) | **헤드 내 융합** | Mamba-2 SSD | 메타 토큰, 가장 세밀한 통합 |

Hymba의 차별점:
1. **레이어가 아닌 헤드 수준 융합** — 같은 입력에 두 관점 동시 적용
2. **메타 토큰** — 글로벌 메모리로 SSM 정보 손실 보완
3. **Mamba-2 SSD 기반** — 최신 SSM 알고리즘 활용

---

## 7. 설계 원칙 정리

Hymba에서 도출되는 SSM-Attention 하이브리드 설계 원칙:

### 원칙 1: SSM 비율 > Attention 비율

대부분의 시퀀스 처리는 SSM으로 충분하다. Attention은 **소수의 헤드**에만 배치하되, ICL 같은 정밀한 추론에 집중시킨다.

### 원칙 2: 같은 레이어에서 융합

레이어 교차보다 **헤드 수준 융합**이 더 효과적이다. 같은 입력에 대해 SSM과 Attention이 동시에 다른 관점의 표현을 생성하고 결합한다.

### 원칙 3: 글로벌 메모리 도입

Learnable meta tokens 같은 **글로벌 메모리**가 SSM의 정보 압축 손실을 보완한다.

### 원칙 4: KV cache 최적화

SSM 헤드는 KV cache 불필요 → 전체 캐시 사용량을 Attention 헤드 비율에 비례하여 줄일 수 있다.

---

## 8. 한계와 열린 질문

1. **스케일링**: 1.5B에서 검증 — 7B, 13B, 70B+에서의 유효성 미확인
2. **최적 헤드 비율**: SSM과 Attention 헤드의 최적 비율이 과제/규모에 따라 달라질 수 있음
3. **메타 토큰 수**: 128개가 최적인지, 과제에 따라 조정해야 하는지
4. **추론 인프라**: SSM 헤드(순환 모드)와 Attention 헤드(KV cache)를 동시에 효율적으로 서빙하는 인프라 필요

---

## 9. 개인적 생각

Hymba는 "SSM vs Attention" 논쟁에 대한 가장 실용적인 답을 제시한다: **둘 다 쓰되, 최적으로 결합하라**.

가장 인상적인 점:

1. **헤드 수준 융합**이라는 간단하면서도 효과적인 아이디어. Multi-Head Attention의 각 헤드가 이미 다른 subspace를 학습하므로, 일부를 SSM으로 대체하는 것은 자연스러운 확장이다.

2. **메타 토큰**이 attention sink 문제까지 해결하는 부수 효과. 단순한 글로벌 메모리로 도입했는데 기존 Transformer의 알려진 문제도 함께 해결된다.

3. **1.5B로 3B를 이기면서 캐시는 1/5** — 소형 모델에서의 효율성이 모바일/엣지 배포에 직접적으로 유용하다.

Mamba-3와 함께 보면, SSM 계열의 발전이 두 축으로 진행되고 있음을 알 수 있다:
- **Mamba-3**: SSM 자체의 이론적 기반 강화 (이산화, 복소수, MIMO)
- **Hymba**: SSM과 Attention의 최적 결합 방법론

장기적으로는 이 두 축이 합쳐져, Mamba-3 수준의 개선된 SSM을 Hymba 스타일로 Attention과 융합하는 방향으로 갈 것으로 예상된다.

<!-- LEARNING-END -->
---

## 읽은 뒤 확인하고 다음으로

**확인 질문:** 요약 기억과 Attention을 함께 쓰는 구조 이해하기를 목표로, 본문에서 다룬 입력·중간 과정·결과를 자기 말로 설명해 보자. 실험이나 사례가 있다면 어떤 조건에서 확인했는지도 덧붙여 보자.

**다음 글:** [\[서베이\] 해마를 어떻게 구현할 것인가 — attention 옆에 기억을 따로 두는 방법들]({{< relref "posts/hippocampal-memory-attention-survey" >}}) — 비슷한 경험을 구분해 저장하고 다시 찾는 기능 나누기.

[전체 학습 순서]({{< relref "learn" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- LEARNING-END-END -->
