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

## 논문 정보

- **제목**: CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs
- **저자**: Hiroyuki Ootomo, Akira Naruse, Corey Nolet, Ray Wang, Tamas Feher, Yong Wang (모두 NVIDIA)
- **학회**: IEEE ICDE 2024
- **코드**: [github.com/rapidsai/cuvs](https://github.com/rapidsai/cuvs) (NVIDIA cuVS 라이브러리)
- **약어**: **C**uda **A**nns **GRA**ph-based

---

## 1. 문제: CPU ANN 알고리즘이 GPU에서 느린 이유

HNSW, NSG 같은 CPU 기반 그래프 ANN은 GPU의 SIMT 실행 모델과 근본적으로 호환되지 않는다:

| CPU ANN 특성 | GPU 문제 |
|-------------|---------|
| **가변 차수** 그래프 | 스레드 간 **로드 불균형** (일부 빨리 끝남) |
| **계층 구조** (HNSW) | 순차적 탐색 → GPU 병렬성 활용 불가 |
| **순차 그리디 탐색** | SIMT의 **워프 발산(warp divergence)** |
| 메모리 접근 패턴 | GPU의 높은 대역폭(A100: 2 TB/s) 활용 불가 |

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

$$\text{Detourable}(X, Y) = \exists Z : \max(w_{XZ}, w_{ZY}) < w_{XY}$$

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
- **GPU↔CPU 호환**: CAGRA로 빌드 → HNSW로 변환하여 CPU 서빙

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
