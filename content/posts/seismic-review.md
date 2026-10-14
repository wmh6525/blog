---
title: "[논문 리뷰] Seismic (SIGIR 2024 준최우수) — 학습된 희소 검색을 서브밀리초로 가속"
date: 2026-04-03
tags: ["논문리뷰", "RAG", "검색", "역색인", "SPLADE"]
categories: ["ML/AI"]
summary: "Seismic 논문 상세 리뷰. 학습된 희소 표현(SPLADE 등)을 위한 역색인을 기하학적 블록 + 요약 벡터로 재설계하여 기존 대비 84-105배 빠른 서브밀리초 검색을 달성한다."
math: true
toc: true
draft: false
---

<!-- LEARNING-START -->
## 처음 읽는 분을 위한 안내

**학습 위치:** [자료를 찾아 답하는 AI]({{< relref "learn" >}}#rag) · 17/17

후보 문서를 모두 자세히 비교하면 검색 비용이 커진다. 이 방법은 후보를 묶고 요약을 먼저 보며 자세히 계산할 대상을 줄인다. 빠르게 걸러내는 단계가 관련 문서를 너무 많이 버리지 않는지도 함께 확인해야 한다.

**먼저 알아둘 말**

- [Dense·Sparse 검색]({{< relref "ai-glossary" >}}#dense-sparse): Dense 검색은 주로 밀집 벡터의 유사도로 찾고, Sparse 검색은 소수 차원에 값이 있는 표현으로 찾는다. 키워드 기반 검색도 후자에 속한다.
- [검색기·리랭킹]({{< relref "ai-glossary" >}}#retriever): 검색기는 후보 문서를 찾는다. 리랭킹은 그 후보를 다시 평가해 질문에 더 관련 있는 순서로 정렬한다.
- [지연·처리량]({{< relref "ai-glossary" >}}#latency): 지연은 요청 하나의 결과를 기다리는 시간이고, 처리량은 일정 시간에 처리하는 작업의 양이다. 둘은 같은 지표가 아니다.

**이번 글의 목표:** 검색 후보를 묶고 걸러 계산을 줄이는 흐름 이해하기.

**이 학습 경로의 앞 글:** [\[논문 리뷰\] CAGRA — GPU에서 ANN 검색을 10-77배 가속하는 NVIDIA의 그래프 인덱스]({{< relref "posts/cagra-review" >}}).

**본문에서 먼저 볼 순서:** **1. 문제: 기존 역색인은 SPLADE에서 느리다** → **2. 핵심 관찰: 중요도의 집중 (Concentration of Importance)** → **3. Seismic 아키텍처**.

첫 독서에서는 문제와 방법의 흐름을 잡자. 수식·코드·세부 성능표는 두 번째에 읽어도 된다. 구현과 수학적 주장까지 확인하려면 그 부분을 다시 살펴봐야 한다.

<!-- PAPER-PREREQUISITES -->
### 선수 개념과 읽을 논문

**먼저 이해할 개념:** 희소 벡터와 내적 점수, 역색인, 후보 제거와 검색 재현율의 관계.

아래 순서는 이 글을 이해하기 위한 공부 순서다. ‘개념 준비’는 읽기 안내이며, 직접적인 연구 계보는 연결 칸에 따로 표시했다. 이미 해당 개념을 설명할 수 있다면 아는 부분은 넘어가도 된다.

| 순서·논문 | 연결·익힐 개념 | 먼저 볼 부분 |
|---|---|---|
| 1. [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) · [블로그 해설]({{< relref "posts/retriever-training-survey" >}}) | **개념 준비**: 질문/문서 인코더·대조 학습·negative | 관련 문서와 오답 문서를 비교하는 학습 데이터 구성 |
| 2. [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720) | **개념 준비**: 학습된 희소 표현·어휘 확장·역색인 | 모델 출력이 희소한 검색 표현이 되는 과정 |

**현재 글과 연결하기:** 위 개념을 익힌 뒤, 이 글에서 ‘검색 후보를 묶고 걸러 계산을 줄이는 흐름 이해하기’에 해당하는 부분을 읽자. 기존 방법의 어느 단계가 바뀌었는지 그림이나 입력·출력으로 설명해 보자.

[선수 논문 전체 지도와 원문 읽는 법]({{< relref "paper-prerequisites" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- PAPER-PREREQUISITES-END -->

<!-- LEARNING-START-END -->

---

## 논문 정보

- **제목**: Efficient Inverted Indexes for Approximate Retrieval over Learned Sparse Representations
- **저자**: Sebastian Bruch (Pinecone), Franco Maria Nardini, Cosimo Rulli (ISTI-CNR), Rossano Venturini (University of Pisa)
- **학회**: **SIGIR 2024 Best Paper Runner-up**
- **코드**: [github.com/TusKANNy/seismic](https://github.com/TusKANNy/seismic) (Rust, Python 바인딩)
- **약어**: **S**pilled Clust**e**ring of **I**nverted Lists with **S**ummaries for **M**aximum **I**nner Produ**c**t Search

---

## 1. 문제: 기존 역색인은 SPLADE에서 느리다

BM25용 역색인 알고리즘(WAND, MaxScore)의 속도는 다음 3가지 가정에 기대고 있다.
1. 쿼리가 **짧다** (2-5개 단어)
2. 단어 빈도가 **Zipf 분포** (역색인 리스트 길이 분포가 편향적)
3. **비음수 정수 가중치** + 높은 희소성

**SPLADE 같은 학습된 희소 표현(LSR)은 이 가정을 모두 위반한다:**

| 항목 | BM25 | SPLADE |
|------|------|--------|
| 쿼리 비영 엔트리 | 2-5개 | **119개** (평균) |
| 리스트 길이 분포 | Zipfian (편향) | **균일에 가까움** |
| 가중치 | 정수 | **실수** |

→ WAND/MaxScore의 조기 종료(early termination)가 작동하지 않아 **worst-case 복잡도**에 근접.

---

## 2. 핵심 관찰: 중요도의 집중 (Concentration of Importance)

SPLADE 벡터의 L1 mass는 **극소수 좌표에 집중**된다:

- 쿼리: **상위 10개** 좌표로 전체 L1 mass의 75% 보존
- 문서: **상위 50개** 좌표로 전체 L1 mass의 75% 보존
- 쿼리 12개 + 문서 25개 좌표만으로 inner product의 **90%** 보존

**$\alpha$-mass subvector** 정의: 벡터 $x$의 엔트리를 절대값 기준 내림차순 정렬했을 때, L1 mass의 $\alpha$ 비율을 커버하는 최소 개수의 엔트리로 구성된 부분 벡터.

---

## 3. Seismic 아키텍처

### 3.1 인덱스 구조

두 가지 데이터 구조를 결합:
1. **역색인**: 평가할 문서 후보를 파악 (좌표별 문서 리스트)
2. **순방향 색인**: 정확한 inner product 계산용 완전한 문서 벡터

역색인의 각 리스트는 **기하학적으로 응집된 블록**으로 분할되고 각 블록에 **요약 벡터**가 부착된다.

### 3.2 인덱스 구축 (3단계)

**Step 1 — Static Pruning**: 각 좌표의 역색인 리스트를 값 기준 내림차순 정렬 → 상위 $\lambda$개만 유지.

**Step 2 — Geometric Blocking**: 각 역색인 리스트를 K-Means 변형으로 최대 $\beta$개 블록으로 분할. 기하학적으로 유사한 문서끼리 그룹화.

**Step 3 — Summary Vectors**: 각 블록 $B$의 요약 벡터:

$$\phi(B)_i = \max_{x \in B} x_i$$

블록 내 모든 문서의 $i$번째 좌표값의 **최대값**. 이것은 **보수적 상한**:

$$\langle q, \phi(B) \rangle \geq \langle q, x \rangle \quad \forall x \in B$$

→ 블록을 **안전하게** 건너뛸 수 있는 조건을 제공.

요약 벡터는 $\alpha$-mass subvector + 8-bit 양자화로 **4배 압축**.

### 3.3 쿼리 처리 알고리즘

```
입력: 쿼리 q, 결과 수 k, cut (쿼리 좌표 수), heap_factor

1. q의 상위 cut개 좌표 선택 (중요도 집중 활용)
2. 각 선택된 좌표 i에 대해:
   3. 해당 좌표의 모든 블록 B_j 순회:
      4. 요약 내적 계산: r = <q, 요약(B_j)>
      5. IF r < HEAP.min / heap_factor → 블록 건너뜀! (Block-level pruning)
      6. ELSE → 블록 내 모든 문서와 정확한 inner product 계산
         → HEAP 업데이트
7. 결과: top-k 문서 반환
```

**블록 수준 가지치기가 핵심**: 개별 문서 대신 **블록 단위**로 판단하여 결정 횟수가 수십-수백 배 감소.

역색인은 질의에 나온 차원을 가진 문서 후보를 찾는다. Seismic은 후보를 블록으로 묶고 요약 표현을 먼저 보면서, 자세히 점수를 계산할 범위를 줄인다. 아래 구조는 후보 수집과 정밀 계산 사이에 걸러내는 단계를 둔 것으로 읽으면 된다. 블록 크기와 요약의 설정을 바꿀 때도 계산을 얼마나 줄였는지와 관련 문서를 얼마나 남겼는지를 함께 봐야 한다.

---

## 4. 왜 서브밀리초를 달성하는가?

1. **Static pruning**: 역색인 리스트 길이를 $\lambda$로 제한하여 검색 공간 대폭 축소
2. **쿼리 좌표 선택**: 상위 cut개만 사용 → 역색인 접근 수 자체를 감소
3. **블록 수준 가지치기**: 요약 벡터로 블록 단위 스킵. PyANN이 ~40,000개 문서를 방문할 때 Seismic은 **~2,198개**만 평가
4. **행렬 곱셈**: 한 리스트의 모든 양자화된 요약을 일괄 계산
5. **Prefetching**: 순방향 색인 접근의 cache miss를 하드웨어 프리페칭으로 완화
6. **Rust 구현**: 최고 수준 최적화 컴파일

---

## 5. 실험 결과

### 하드웨어

Intel i9-9900K (3.60 GHz), 64 GiB RAM, **단일 스레드**.

### 5.1 검색 속도 (MS MARCO, SPLADE)

| 방법 | 95% 정확도 (μs) | Seismic 대비 |
|------|----------------|-------------|
| IOQP | 31,843 | **105배** 느림 |
| SparseIVF | 10,254 | **34배** 느림 |
| GrassRMA | 1,271 | **4.2배** 느림 |
| PyANN | 1,016 | **3.4배** 느림 |
| **Seismic** | **303** | **기준** |

### 5.2 uniCOIL-T5에서 (MS MARCO)

| 방법 | 95% 정확도 (μs) | Seismic 대비 |
|------|----------------|-------------|
| IOQP | 34,061 | **189배** 느림 |
| SparseIVF | 12,308 | **68배** 느림 |
| PyANN | 2,973 | **17배** 느림 |
| **Seismic** | **180** | **기준** |

### 5.3 인덱스 빌드 시간

| 방법 | 빌드 시간 | 인덱스 크기 |
|------|---------|-----------|
| GrassRMA | 267분 | 10,489 MiB |
| PyANN | 137분 | 5,262 MiB |
| SparseIVF | 44분 | 8,830 MiB |
| **Seismic** | **5분** | 6,416 MiB |

빌드 시간 **5분** — 경쟁자 대비 27-53배 빠름.

### 5.4 MRR@10 분석

핵심 발견: Seismic + SPLADE가 같은 시간 예산에서 **E-SPLADE보다 더 높은 MRR@10** 달성. 즉, Seismic이 SPLADE를 충분히 빠르게 만들어 정확도를 희생하는 E-SPLADE가 불필요해짐.

---

## 6. Ablation Study

### 기하학적 블로킹 vs 고정 블로킹

| 방법 | 효과 |
|------|------|
| Fixed (고정 크기 분할) | 기준 |
| **Geometric (K-Means)** | **모든 하이퍼파라미터에서 유의미하게 우수** |

기하학적으로 유사한 문서끼리 그룹화하면 요약 벡터의 상한이 더 타이트 → 가지치기 효과 증가.

### 중요도 기반 vs 고정 크기 요약

| 방법 | 효과 |
|------|------|
| 고정 128개 엔트리 | 기준 |
| **$\alpha$-mass subvector** | **같은 시간 예산에서 더 높은 정확도** |

---

## 7. 하이퍼파라미터

| 파라미터 | 의미 | 최적값 (MS MARCO) |
|---------|------|-----------------|
| $\lambda$ | 역색인 리스트 최대 길이 | 6,000 |
| $\beta$ | 리스트당 최대 블록 수 | 400 |
| $\alpha$ | 요약 벡터의 mass 보존 비율 | 0.4 |
| cut | 쿼리에서 사용할 좌표 수 | 데이터 의존적 |
| heap\_factor | 블록 가지치기 보정 계수 | 데이터 의존적 |

---

## 8. 한계

1. **근사 검색**: 정확도와 속도의 트레이드오프 (recall 100%가 아님)
2. **비음수 벡터만**: 컬렉션이 $\mathbb{R}_+^d$ (비음수)여야 함
3. **인덱스 크기**: IOQP(2,195 MiB)보다 큰 6,416 MiB (요약 벡터 + 순방향 색인)
4. **하이퍼파라미터 튜닝**: 5개 파라미터를 데이터셋/모델별 그리드 서치
5. **극저지연에서 MRR 약간 하락**: 공격적 가지치기 시 랭킹 품질 약간 감소

---

## 9. 후속 연구

- **CIKM 2024**: k-NN 그래프와 결합한 확장
- **ECIR 2025**: 대규모 데이터셋 확장성 (Best Short Student Paper 수상)
- **ECIR 2026**: Forward Index 압축

<!-- LEARNING-END -->
---

## 읽은 뒤 확인하고 다음으로

**확인 질문:** 검색 후보를 묶고 걸러 계산을 줄이는 흐름 이해하기를 목표로, 본문에서 다룬 입력·중간 과정·결과를 자기 말로 설명해 보자. 실험이나 사례가 있다면 어떤 조건에서 확인했는지도 덧붙여 보자.

이 경로의 마지막 글이다. [학습 순서]({{< relref "learn" >}})로 돌아가 다른 경로를 고르거나, 어려웠던 수식·코드 부분을 다시 읽어 보자.

[전체 학습 순서]({{< relref "learn" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- LEARNING-END-END -->
