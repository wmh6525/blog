---
title: "[논문 리뷰] TurboRAG — KV 캐시 사전 계산으로 RAG 추론 9.4배 가속"
date: 2026-04-03
tags: ["논문리뷰", "RAG", "LLM", "최적화"]
categories: ["ML/AI"]
summary: "TurboRAG 논문 상세 리뷰. 청크별 KV 캐시를 오프라인 사전 계산하고, Independent Attention + Reordered RoPE로 프리필 연산을 제거하여 TTFT를 최대 9.4배 줄인다."
math: true
toc: true
draft: false
---

<!-- LEARNING-START -->
## 처음 읽는 분을 위한 안내

**학습 위치:** [자료를 찾아 답하는 AI]({{< relref "learn" >}}#rag) · 14/17

같은 문서를 질문마다 처음부터 계산하면 반복 작업이 생긴다. 이 방법은 문서에서 미리 할 수 있는 계산을 저장해 두었다가 질문이 들어오면 가져온다. 사전에 끝낸 작업과 요청마다 해야 하는 작업을 구분해 읽어 보자.

**먼저 알아둘 말**

- [KV 캐시·프리필]({{< relref "ai-glossary" >}}#kv): KV 캐시는 Attention에 쓸 중간 계산 결과를 저장한 것이다. 프리필은 주어진 입력 문맥을 먼저 처리하는 단계다.
- [패치·청크]({{< relref "ai-glossary" >}}#patch): 작은 단위 여러 개를 한 묶음으로 만든 것. 이미지 패치, 바이트 패치, 문서 청크는 서로 다른 대상을 묶는다.
- [RAG]({{< relref "ai-glossary" >}}#rag): 질문에 관련된 자료를 검색해 모델에 함께 제공하고 답변을 만드는 방식. 자료를 찾는 단계와 답을 쓰는 단계를 구분해서 본다.

**이번 글의 목표:** 문서 계산을 미리 해 두는 방법 이해하기.

**이 학습 경로의 앞 글:** [\[서베이\] RAG 검색 속도를 획기적으로 줄이는 방법들 — KV 캐시, 서브밀리초 인덱스, 예측적 프리페칭]({{< relref "posts/rag-latency-optimization" >}}).

**본문에서 먼저 볼 순서:** **1. 문제: RAG의 프리필 병목** → **2. 두 가지 핵심 관찰** → **3. TurboRAG 아키텍처**.

첫 독서에서는 문제와 방법의 흐름을 잡자. 수식·코드·세부 성능표는 두 번째에 읽어도 된다. 구현과 수학적 주장까지 확인하려면 그 부분을 다시 살펴봐야 한다.

<!-- PAPER-PREREQUISITES -->
### 선수 개념과 읽을 논문

**먼저 이해할 개념:** 벡터의 유사도, 관련/무관 예시의 비교, 검색과 생성의 구분.

아래 순서는 이 글을 이해하기 위한 공부 순서다. ‘개념 준비’는 읽기 안내이며, 직접적인 연구 계보는 연결 칸에 따로 표시했다. 이미 해당 개념을 설명할 수 있다면 아는 부분은 넘어가도 된다.

| 순서·논문 | 연결·익힐 개념 | 먼저 볼 부분 |
|---|---|---|
| 1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | **개념 준비**: Attention·위치 정보·인코더/디코더 | 현재 토큰이 다른 위치의 정보를 읽는 구조 그림과 Attention 설명 |
| 2. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) · [블로그 해설]({{< relref "posts/rag-survey-2026" >}}) | **개념 준비**: 검색기·생성기·외부 지식 | 질문에서 문서를 검색하고 문서와 함께 답을 만드는 구성 |

**현재 글과 연결하기:** 위 개념을 익힌 뒤, 이 글에서 ‘문서 계산을 미리 해 두는 방법 이해하기’에 해당하는 부분을 읽자. 기존 방법의 어느 단계가 바뀌었는지 그림이나 입력·출력으로 설명해 보자.

[선수 논문 전체 지도와 원문 읽는 법]({{< relref "paper-prerequisites" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- PAPER-PREREQUISITES-END -->

<!-- LEARNING-START-END -->

---

## 논문 정보

- **제목**: TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text
- **저자**: Songshuo Lu, Hua Wang, Yutian Rong, Zhi Chen, Yaohua Tang
- **소속**: Moore Threads AI
- **학회**: EMNLP 2025
- **코드**: [github.com/MooreThreads/TurboRAG](https://github.com/MooreThreads/TurboRAG)

---

## 1. 문제: RAG의 프리필 병목

기존 RAG 시스템은 다음 순서로 추론한다.

```
질의 → 검색(top-k 청크) → 청크들을 프롬프트에 연결 → 프리필(KV 캐시 계산) → 생성
```

**프리필이 병목인 이유**:
1. 프리필 비용이 입력 길이에 대해 **이차적**으로 증가
2. 동일한 청크가 다른 질의에서 반복 검색되어도 **매번 재계산**
3. 긴 연결 문서의 프리필이 GPU 자원을 독점하여 배치 크기 제한

> TurboRAG의 질문: "프리필을 오프라인으로 미리 해놓을 수는 없을까?"

---

## 2. 두 가지 핵심 관찰

### 관찰 1: 청크 간 Cross-Attention은 극히 희소하다

어텐션 맵을 시각화하면, 각 문서 청크는 **자기 자신 내부**에만 주로 어텐드한다. 서로 다른 청크 간의 어텐션 점수는 매우 낮다.

→ 청크 간 cross-attention을 **제거해도** 성능 하락이 미미하다.

### 관찰 2: RoPE는 상대 위치만 중요하다

RoPE(Rotary Position Embedding)의 어텐션 점수는 두 토큰의 **상대 거리**에만 의존한다:

$$\text{RoPE}(q_t, k_s) = f(q, k, t - s)$$

→ 각 청크를 **독립적으로** 프리필해도, 나중에 위치를 재배열하면 올바른 상대 위치를 복원할 수 있다.

---

## 3. TurboRAG 아키텍처

### 3.1 Independent Attention

문서 간 cross-attention을 완전히 제거한 어텐션 마스크:

```
기존 Causal Attention:        Independent Attention:
┌──────────────┐              ┌──────────────┐
│█             │              │█             │
│██            │              │██            │
│███           │              │ ██           │
│████          │              │ ███          │
│█████         │              │   ██         │
│██████        │              │   ███        │
│███████       │              │█████████     │  ← 질의는 모든 청크에 어텐드
│████████      │              │██████████    │
└──────────────┘              └──────────────┘
  청크 A  청크 B  질의            청크 A  청크 B  질의
```

각 청크는 자기 자신에만 어텐드. 질의/응답은 모든 청크와 자기 자신에 어텐드.

### 3.2 Reordered RoPE

**문제**: 각 청크를 독립적으로 프리필하면 모든 KV 캐시의 position ID가 $[0, 1, \ldots, l]$로 동일. 단순 연결 시 위치가 겹침.

**해결**: KV 캐시를 RoPE 적용 **전**(raw 상태)으로 저장. 연결 후 연속 position ID $[0, 1, \ldots, kl]$로 RoPE 적용.

```python
# Key를 raw 상태로 캐시에 저장 (RoPE 미적용)
key_states, value_states = past_key_value.update(key_states, value_states)

# 연결 후 연속 position ID로 RoPE 적용
full_position_ids = torch.arange(0, past_key_value.seen_tokens)
key_states = apply_rotary_pos_emb(key_states, cos, sin, full_position_ids)
```

### 3.3 전체 파이프라인

```
=== 오프라인 (1회) ===
각 청크 c_i → LLM 프리필 → KV 캐시 저장 (RoPE 미적용 상태)

=== 온라인 (매 질의) ===
질의 → 검색 (top-k 청크)
     → 저장된 KV 캐시 k개 로드
     → 연결 + Reordered RoPE 적용
     → 질의 토큰만 프리필 (~128 토큰)
     → 즉시 생성 시작!
```

**핵심**: 프리필 대상이 전체 컨텍스트(~8192 토큰)에서 질의 토큰(~128 토큰)으로 축소. **연산량 98.46% 감소**.

일반적인 프리필은 질문이 들어온 뒤 문서를 읽으며 캐시를 만든다. TurboRAG는 문서 청크에서 미리 계산할 수 있는 부분을 앞당겨 두고, 요청 때 필요한 캐시를 불러온다. 이때 청크가 서로 독립적으로 계산됐다는 조건이 Attention과 위치 인코딩 설계에 연결된다. 아래 구조를 따라갈 때는 오프라인에서 끝낸 작업과 요청마다 남는 작업을 나눠 표시하면 된다.

---

## 4. Fine-tuning

Independent Attention + Reordered RoPE 패턴에 모델을 적응시키기 위한 SFT 수행.

- **베이스 모델**: Qwen2-7B
- **하드웨어**: 32× A100 80GB
- **데이터**: 18개 Document QA 데이터셋 (~190K 샘플) + 일반 대화/추론/코드
- **학습률**: 1e-5, AdamW

---

## 5. 실험 결과

### 5.1 Document QA 정확도 (RGB 벤치마크)

5개 문서 검색, Noise Ratio = 불필요 문서 비율.

| 모델 | NR=0.2 | NR=0.4 | NR=0.6 | NR=0.8 | 평균 |
|------|--------|--------|--------|--------|------|
| GPT-4o | 99.0 | 99.3 | 98.3 | 96.3 | 98.2 |
| Naive RAG | 99.7 | 99.3 | 99.3 | 94.3 | 98.2 |
| TurboRAG (FT 전) | 98.0 | 97.3 | 90.7 | 85.7 | 92.9 |
| **TurboRAG (FT 후)** | **99.0** | **98.3** | **96.0** | **93.7** | **96.8** |

파인튜닝 후 Naive RAG 대비 **1.4%p 이내** 격차.

### 5.2 TTFT 속도 향상 (LongBench)

| 데이터셋 | 컨텍스트 토큰 | Naive TTFT | Turbo TTFT | 속도 향상 |
|---------|------------|-----------|-----------|---------|
| musique | 16,349 | 1,610ms | 171ms | **9.4배** |
| 2wikimqa | 7,553 | 709ms | 101ms | **7.0배** |
| dureader | 10,642 | 1,007ms | 116ms | **8.7배** |
| hotpotqa | 13,453 | 1,333ms | 147ms | **9.1배** |
| **평균** | **11,999** | **1,165ms** | **134ms** | **8.6배** |

### 5.3 배치 확장 (h2d 전송 제외 시)

| 배치 | Naive TTFT | Turbo TTFT | 속도 향상 |
|------|-----------|-----------|---------|
| 1 | 711ms | 44ms | **16.1배** |
| 4 | 2,842ms | 97ms | **29.3배** |
| 8 | 5,812ms | 177ms | **32.8배** |

KV 캐시가 GPU 메모리에 프리페치되면 **최대 32.8배** 가속.

### 5.4 범용 능력 저하 없음

| 벤치마크 | Naive RAG | TurboRAG | 차이 |
|---------|----------|---------|------|
| MMLU | 69.57 | 70.73 | +1.16 |
| GSM-8K | 79.12 | 79.45 | +0.33 |
| MATH | 39.54 | 40.58 | +1.04 |

---

## 6. 스토리지 오버헤드

Qwen2-7B, 청크 512 토큰 기준:

$$\text{KV 캐시/청크} = 2 \times 28 \times 4 \times 128 \times 512 \times 2 \approx \textbf{29.4 MB}$$

원본 텍스트(~2KB) 대비 **~15,000배** 스토리지. 이것이 TurboRAG의 핵심 트레이드오프: **공간 ↔ 시간**.

---

## 7. 한계

1. **스토리지 비용**: 청크당 ~30MB KV 캐시 저장 필요
2. **h2d 전송 병목**: CPU→GPU KV 캐시 전송이 프리페치 없으면 속도 향상 제한 (~4배)
3. **Cross-attention 손실**: 문서 간 상호 참조가 필요한 복잡한 multi-hop 추론에서 약점
4. **Fine-tuning 필수**: 기존 모델에 바로 적용 시 정확도 하락
5. **모델 업데이트 시 재계산**: LLM이 업데이트되면 모든 KV 캐시 재계산 필요

<!-- LEARNING-END -->
---

## 읽은 뒤 확인하고 다음으로

**확인 질문:** 문서 계산을 미리 해 두는 방법 이해하기를 목표로, 본문에서 다룬 입력·중간 과정·결과를 자기 말로 설명해 보자. 실험이나 사례가 있다면 어떤 조건에서 확인했는지도 덧붙여 보자.

**다음 글:** [\[논문 리뷰\] CacheBlend (EuroSys 2025 Best Paper) — 비접두사 KV 캐시 재사용으로 RAG 5배 가속]({{< relref "posts/cacheblend-review" >}}) — 저장한 계산을 재사용할 때 문맥 차이를 반영하는 이유 이해하기.

[전체 학습 순서]({{< relref "learn" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- LEARNING-END-END -->
