---
title: "[논문 리뷰] CAGRA — GPU에서 ANN 검색을 10-77배 가속하는 NVIDIA의 그래프 인덱스"
date: 2026-04-03
tags: ["논문리뷰", "RAG", "검색", "ANN", "GPU", "NVIDIA"]
categories: ["ML/AI"]
summary: "CAGRA 논문 상세 리뷰. GPU 병렬성을 극대화한 고정 차수 그래프 ANN 인덱스. HNSW 대비 인덱스 빌드 10-30배, 대규모 배치 검색 33-77배, 단일 쿼리도 3.4-53배 가속."
math: true
toc: true
draft: false
---

<!-- LEARNING-START -->
## 처음 읽는 분을 위한 안내

**학습 위치:** [자료를 찾아 답하는 AI]({{< relref "learn" >}}#rag) · 16/17

많은 문서 중 가까운 문서를 찾을 때 여러 후보를 함께 살펴보면 시간을 줄일 수 있다. 하지만 후보마다 일이 달라지면 동시에 계산하는 장치가 비효율적으로 움직인다. 이 연구는 검색 구조를 GPU의 병렬 계산에 맞추는 방법을 다룬다.

**먼저 알아둘 말**

- [GPU·병렬 처리]({{< relref "ai-glossary" >}}#gpu): GPU는 많은 계산을 함께 처리하는 데 쓰는 장치다. 병렬 처리는 작업을 나눠 동시에 수행하는 방식이다.
- [그래프·지식 그래프]({{< relref "ai-glossary" >}}#graph): 대상들을 점으로, 그 사이 관계를 연결로 나타낸 구조. 지식 그래프는 대상과 관계에 의미를 붙여 정보를 표현한다.
- [검색기·리랭킹]({{< relref "ai-glossary" >}}#retriever): 검색기는 후보 문서를 찾는다. 리랭킹은 그 후보를 다시 평가해 질문에 더 관련 있는 순서로 정렬한다.

**이번 글의 목표:** 문서 검색을 GPU에서 병렬로 수행하는 설계 이해하기.

**이 학습 경로의 앞 글:** [\[논문 리뷰\] CacheBlend (EuroSys 2025 Best Paper) — 비접두사 KV 캐시 재사용으로 RAG 5배 가속]({{< relref "posts/cacheblend-review" >}}).

**본문에서 먼저 볼 순서:** **1. 문제: CPU ANN 알고리즘이 GPU에서 느린 이유** → **6. 실험 결과** → **9. 한계**.

첫 독서에서는 문제와 방법의 흐름을 잡자. 수식·코드·세부 성능표는 두 번째에 읽어도 된다. 구현과 수학적 주장까지 확인하려면 그 부분을 다시 살펴봐야 한다.

<!-- PAPER-PREREQUISITES -->
### 선수 개념과 읽을 논문

**먼저 이해할 개념:** 벡터의 거리·유사도, 그래프의 이웃 탐색, 근사 검색의 재현율, GPU의 병렬 실행.

아래 순서는 이 글을 이해하기 위한 공부 순서다. ‘개념 준비’는 읽기 안내이며, 직접적인 연구 계보는 연결 칸에 따로 표시했다. 이미 해당 개념을 설명할 수 있다면 아는 부분은 넘어가도 된다.

| 순서·논문 | 연결·익힐 개념 | 먼저 볼 부분 |
|---|---|---|
| 1. [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320) | **개념 준비**: 근사 이웃 검색·그래프 탐색 | 이웃 연결을 따라 후보를 찾는 기본 탐색 흐름 |
| 2. [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) · [블로그 해설]({{< relref "posts/retriever-training-survey" >}}) | **개념 준비**: 질문/문서 인코더·대조 학습·negative | 관련 문서와 오답 문서를 비교하는 학습 데이터 구성 |

**현재 글과 연결하기:** 위 개념을 익힌 뒤, 이 글에서 ‘문서 검색을 GPU에서 병렬로 수행하는 설계 이해하기’에 해당하는 부분을 읽자. 기존 방법의 어느 단계가 바뀌었는지 그림이나 입력·출력으로 설명해 보자.

[선수 논문 전체 지도와 원문 읽는 법]({{< relref "paper-prerequisites" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- PAPER-PREREQUISITES-END -->

<!-- LEARNING-START-END -->

---

## 논문 정보

- **제목**: CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs
- **저자**: Hiroyuki Ootomo, Akira Naruse, Corey Nolet, Ray Wang, Tamas Feher, Yong Wang (모두 NVIDIA)
- **학회**: IEEE ICDE 2024
- **코드**: [github.com/rapidsai/cuvs](https://github.com/rapidsai/cuvs) (NVIDIA cuVS 라이브러리)
- **약어**: **C**uda **A**nns **GRA**ph-based

---

## 1. 문제: CPU ANN 알고리즘이 GPU에서 느린 이유

HNSW, NSG 같은 CPU 기반 그래프 ANN의 탐색 구조는 GPU의 SIMT 실행 모델과 맞지 않는다. 충돌하는 지점은 다음과 같다.

| CPU ANN 특성 | GPU 문제 |
|-------------|---------|
| **가변 차수** 그래프 | 스레드 간 **로드 불균형** (일부 빨리 끝남) |
| **계층 구조** (HNSW) | 순차적 탐색 → GPU 병렬성 활용 불가 |
| **순차 그리디 탐색** | SIMT의 **워프 발산(warp divergence)** |
| 메모리 접근 패턴 | GPU의 높은 대역폭(A100: 2 TB/s) 활용 불가 |

그래프 검색에서는 각 탐색이 어느 이웃으로 이동하느냐에 따라 작업량이 달라진다. CPU에서는 자연스러운 이 불규칙성이 GPU에서는 함께 실행되는 스레드의 발을 맞추기 어렵게 만든다. 아래의 고정 차수와 병렬 탐색 설계는 이 작업량을 더 규칙적으로 만들기 위한 장치다. 검색 알고리즘과 하드웨어 배치를 함께 바꾼다는 점이 CAGRA를 이해하는 출발점이다.

---

## 2. CAGRA 그래프의 4가지 특징

| 특징 | 설명 | GPU 이점 |
|------|------|---------|
| **고정 차수 $d$** | 모든 노드가 정확히 $d$개의 엣지 | 균일한 워크로드 → 로드 밸런싱 |
| **방향 그래프** | 고정 차수이므로 자연스럽게 유향 | 탐색 방향 명확 |
| **단층 구조** | 계층 없음 (HNSW의 multi-layer와 대조) | 랜덤 시작점 + 대규모 병렬 탐색 |
| **검색 최적화** | 이론적 그래프 성질보다 검색 성능에 최적화 | GPU 실행 효율 극대화 |

---

## 3. 그래프 구축 알고리즘

### Stage 1: 초기 k-NN 그래프

GPU에서 **NN-Descent** 알고리즘으로 초기 $d_{\text{init}}$-NN 그래프 구축 ($d_{\text{init}} = 2d \sim 3d$).

### Stage 2: 그래프 최적화 (2단계)

**(a) Rank-based Reordering + Pruning**:

각 엣지 $(X, Y)$에 대해 **우회 경로(detour)** 존재 여부를 확인:

$$\text{Detourable}(X, Y) = \exists Z : \max(w_{XZ}, w_{ZY}) \lt  w_{XY}$$

우회 가능한 엣지를 우선 제거 → 상위 $d$개만 유지.

**핵심**: 거리 계산 대신 **순위(rank)**를 프록시로 사용 → 거리 계산 비용 제거, 1.9배 빠름.

**(b) 역엣지 추가**:

```
가지치기된 그래프 + 역방향 그래프 → 인터리빙 병합

각 노드: d/2개는 원래 그래프, d/2개는 역엣지
```

"나를 중요하게 여기는 사람은 나에게도 중요하다" — 도달 가능성 대폭 개선.

---

## 4. 검색 알고리즘

### 버퍼 구조

```
[Internal top-M 리스트 (길이 M)] + [후보 리스트 (길이 p × d)]
```

### 4단계 반복

1. **랜덤 샘플링** (초기화): $p \times d$개 노드를 균일 랜덤 선택 → 거리 계산
2. **Top-M 업데이트**: 전체 버퍼에서 최소 거리 $M$개 선택 (비토닉 정렬)
3. **이웃 인덱스 수집**: top-$p$ 노드의 이웃 인덱스 수집 (거리 계산 없음)
4. **거리 계산**: 새 노드에 대해서만 거리 계산 (해시 테이블로 중복 방지)

→ top-M이 변하지 않을 때까지 2-4 반복.

---

## 5. GPU 병렬화 기법

### 5.1 워프 분할 ("팀")

GPU 워프(32 스레드)를 더 작은 **팀**으로 분할:

| 차원 | 최적 팀 크기 |
|------|-----------|
| 96 (DEEP) | 4 또는 8 |
| 960 (GIST) | 32 (전체 워프) |

한 팀이 한 후보의 거리를 계산하는 동안, 다른 팀은 다른 후보를 동시 계산. **워프 발산 없음**.

### 5.2 두 가지 실행 모드

| | single-CTA | multi-CTA |
|--|-----------|-----------|
| **용도** | 대규모 배치 (≥100) | 소규모 배치 (1~100) |
| **구조** | 1 쿼리 = 1 CTA | 1 쿼리 = 여러 CTA |
| **해시 테이블** | 공유 메모리 (~4 KB) | 디바이스 메모리 |
| **반복당 노드** | $p \times d$ | $\text{num_CTAs} \times d$ |

**핵심**: multi-CTA 모드 덕분에 **단일 쿼리에서도** CPU보다 빠른 최초의 GPU ANN.

### 5.3 잊어버리는 해시 테이블

공유 메모리 제한으로 해시 테이블을 주기적으로(1-4 반복마다) 리셋. 리셋 후 현재 top-M 노드만 재등록. 약간의 중복 계산을 감수하되, 극적인 메모리 절약과 병렬 효율 확보.

---

## 6. 실험 결과

### 하드웨어

DGX A100 (AMD EPYC 7742 CPU + NVIDIA A100 80GB GPU)

### 6.1 그래프 구축 시간

| 데이터셋 | CAGRA (GPU) | HNSW (CPU) | 속도 향상 |
|---------|-----------|-----------|---------|
| SIFT-1M | 14.5s | 32.3s | **2.2배** |
| GIST-1M | 28.9s | 691.6s | **24배** |
| GloVe-200 | 25.1s | 172.2s | **6.9배** |
| NYTimes | 7.6s | 226.3s | **30배** |

### 6.2 대규모 배치 검색 (배치=10K)

90-95% recall@10에서:
- HNSW 대비: **33-77배** 빠름
- GPU SOTA (GGNN/GANNS) 대비: **3.8-8.8배** 빠름
- SIFT에서 **$10^7$ QPS** 달성

### 6.3 단일 쿼리 검색

| 데이터셋 | CAGRA vs HNSW (95% recall) |
|---------|--------------------------|
| SIFT | **53배** 빠름 |
| GIST | **3.4배** 빠름 |
| GloVe | **10배** 빠름 |

**이전 GPU 구현(GGNN, GANNS)은 단일 쿼리에서 오히려 HNSW보다 느렸다.** CAGRA가 최초로 **모든 배치 크기에서** CPU를 능가.

### 6.4 대규모 데이터셋 스케일링 (DEEP)

| 데이터셋 | CAGRA 빌드 | HNSW 빌드 |
|---------|-----------|----------|
| 1M | 14.6s | 27.3s |
| 10M | 130.4s | 236.5s |
| 100M | 1,305.6s | 2,623.3s |

데이터 규모에 비례하여 스케일링, **일관된 ~2배 우위 유지**.

---

## 7. 벡터 DB 통합 현황

| 시스템 | 통합 방식 | 주요 성과 |
|--------|---------|---------|
| **Milvus** | GPU 인덱스 직접 | 인덱스 빌드 21배↑, 6.35억 벡터 56분(8× DGX H100) |
| **Weaviate** | GPU 빌드 → HNSW 변환 | 빌드 4.7-8배↑, 쿼리 2.6-3.5배↑ |
| **Elasticsearch** | cuvs-java (Panama FFI) | 인덱싱 처리량 12배↑, force-merge 7배↑ |
| **OpenSearch** | GPU 인덱싱 | 인덱싱 9.3배↑, 비용 3.75배↓ |
| **Faiss** | cuVS 백엔드 | HNSW→CAGRA: 빌드 6.4-12.3배↑, 검색 2.4-4.7배↑ |
| **Google AlloyDB** | pgvector 가속 | pgvector CPU 대비 9배↑ |

---

## 8. cuVS 라이브러리

NVIDIA의 오픈소스 벡터 검색 라이브러리. CAGRA 외에도:

- IVF-Flat, IVF-PQ, HNSW, Brute-force, K-means
- CAGRA-Q (양자화 버전)
- **언어 바인딩**: C, C++, Python, Rust, Java, Go
- **GPU↔CPU 호환**: CAGRA로 빌드 → HNSW로 변환해 CPU 서빙

```python
# cuVS Python 사용 예
from cuvs.neighbors import cagra

index_params = cagra.IndexParams(graph_degree=64)
index = cagra.build(index_params, dataset)

search_params = cagra.SearchParams()
distances, indices = cagra.search(search_params, index, queries, k=10)
```

---

## 9. 한계

1. **GPU 메모리 제한**: 데이터셋 + 그래프가 GPU 메모리에 들어가야 함
2. **데이터 크기 상한**: 1-bit 부모 노드 관리로 $2^{31} - 1$ (~21.5억) 벡터
3. **최적 차수 경험적 결정**: $d$ 값을 데이터셋별로 실험으로 찾아야 함
4. **그래프 최적화 CPU 수행**: 초기 k-NN은 GPU, 최적화는 CPU에서 실행
5. **NVIDIA GPU 전용**: CUDA 기반, AMD/Intel GPU 미지원
6. **IVF보다 높은 메모리 비용**: 전체 그래프를 GPU 메모리에 저장 필요

<!-- LEARNING-END -->
---

## 읽은 뒤 확인하고 다음으로

**확인 질문:** 문서 검색을 GPU에서 병렬로 수행하는 설계 이해하기를 목표로, 본문에서 다룬 입력·중간 과정·결과를 자기 말로 설명해 보자. 실험이나 사례가 있다면 어떤 조건에서 확인했는지도 덧붙여 보자.

**다음 글:** [\[논문 리뷰\] Seismic (SIGIR 2024 준최우수) — 학습된 희소 검색을 서브밀리초로 가속]({{< relref "posts/seismic-review" >}}) — 검색 후보를 묶고 걸러 계산을 줄이는 흐름 이해하기.

[전체 학습 순서]({{< relref "learn" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- LEARNING-END-END -->
