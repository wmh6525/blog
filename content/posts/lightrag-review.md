---
title: "[논문 리뷰] LightRAG — 그래프 기반 RAG를 가볍고 빠르게, GraphRAG의 대안"
date: 2026-04-13
tags: ["논문리뷰", "RAG", "LLM", "지식그래프", "LightRAG"]
categories: ["ML/AI"]
summary: "LightRAG 논문 상세 리뷰. 지식 그래프 + 이중 수준 검색(엔터티/관계)으로 GraphRAG의 커뮤니티 탐색 비용 문제를 해결한다. 단일 API 호출로 검색, 점진적 업데이트 지원, GraphRAG 대비 다양성 77% 승률."
math: true
toc: true
draft: false
---

## 논문 정보

- **제목**: LightRAG: Simple and Fast Retrieval-Augmented Generation
- **저자**: Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang
- **소속**: Beijing University of Posts and Telecommunications / University of Hong Kong (HKUDS)
- **학회**: EMNLP 2025 Findings
- **코드**: [github.com/HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) (33K+ stars)

---

## 1. 문제: GraphRAG는 너무 무겁다

### Naive RAG의 한계

청크 기반 벡터 검색은 **엔터티 간 관계**를 이해하지 못한다.

질의: "전기차의 부상이 도시 공기질과 대중교통 인프라에 어떤 영향을 미치는가?"

→ Naive RAG는 전기차, 공기 오염, 대중교통에 대한 **단편적 문서**를 각각 검색하지만, **상호 연결 관계**를 합성하지 못한다.

### GraphRAG의 한계

Microsoft GraphRAG는 이를 지식 그래프로 해결했지만:

| 문제 | 구체적 수치 |
|------|----------|
| **커뮤니티 탐색 비용** | Legal 데이터: 1,399개 커뮤니티, 각 ~1,000 토큰 → **검색당 610K 토큰** |
| **수백 번의 API 호출** | 커뮤니티마다 순회 → 검색 1회에 수백 회 LLM 호출 |
| **점진적 업데이트 불가** | 새 문서 추가 시 **전체 커뮤니티 재구축** 필요 |

---

## 2. LightRAG의 해결: 커뮤니티 없는 그래프 RAG

### 핵심 차이

| | GraphRAG | LightRAG |
|--|---------|---------|
| 그래프 구조 | 엔터티 → **커뮤니티** → 요약 | 엔터티 + 관계 **직접 검색** |
| 검색 방식 | 커뮤니티 순회 (수백 API 호출) | 벡터 매칭 (**1회 API 호출**) |
| 검색 토큰 | ~610K/쿼리 | **< 100/쿼리** |
| 점진적 업데이트 | 전체 재구축 | **그래프 합집합** (추가만) |
| 검색 수준 | 커뮤니티 수준 | **엔터티(로컬) + 관계(글로벌) 이중 수준** |

---

## 3. 아키텍처 상세

### 3.1 그래프 기반 텍스트 인덱싱

$$\hat{D} = (V, E) = \text{Dedupe} \circ \text{Prof}(V, E), \quad V, E = \bigcup_{D_i \in D} \text{Recog}(D_i)$$

3단계 파이프라인, 모두 **LLM 기반**:

```
문서 → [1] 청킹 (1,200 토큰)
         ↓
       [2] 엔터티/관계 추출 (LLM)
         ↓
         엔터티: (이름, 유형, 설명)
         관계: (소스, 타겟, 키워드, 설명)
         ↓
       [3] 프로파일링 (Key-Value 쌍 생성)
         ↓
         엔터티 → (이름, 설명 요약)
         관계 → (키워드들, 연결된 글로벌 테마 포함)
         ↓
       [4] 중복 제거 (이름 매칭 + 설명 병합)
         ↓
       지식 그래프 + 벡터 인덱스 완성
```

### 엔터티 유형 (기본 11개)

Person, Creature, Organization, Location, Event, Concept, Method, Content, Data, Artifact, NaturalObject

### Gleaning 메커니즘

LLM에게 한 번 더 "놓친 엔터티/관계가 있는가?" 재질의하여 추출 품질 향상.

### 3.2 이중 수준 검색 (Dual-Level Retrieval)

**질의 유형에 따라 다른 검색 전략**:

| 질의 유형 | 예시 | 검색 수준 |
|---------|------|---------|
| **구체적** | "Pride and Prejudice의 저자는?" | **Low-level** (엔터티 중심) |
| **추상적** | "AI가 현대 교육에 미치는 영향은?" | **High-level** (관계/테마 중심) |

### 검색 알고리즘 (3단계)

```
질의 q
  ↓
[1] 키워드 추출 (LLM, 1회 호출)
    → low_level_keywords: ["Pride and Prejudice", "author"]
    → high_level_keywords: ["literary influence", "British novels"]
  ↓
[2] 벡터 매칭
    → 로컬 키워드 → 엔터티 VDB 검색
    → 글로벌 키워드 → 관계 VDB 검색
  ↓
[3] 이웃 확장 (1-hop)
    → 검색된 엔터티/관계의 1-hop 이웃 노드 수집
  ↓
검색 결과 → LLM 생성 (1회 호출)
```

**총 LLM 호출: 2회** (키워드 추출 1회 + 답변 생성 1회). GraphRAG의 수백 회와 비교.

### 3.3 검색 모드 (6가지)

| 모드 | 설명 | 용도 |
|------|------|------|
| **naive** | 벡터 검색만 (그래프 미사용) | 기본 RAG 비교용 |
| **local** | 엔터티 중심 검색 | 구체적 질의 |
| **global** | 관계 중심 검색 | 추상적 질의 |
| **hybrid** | local + global 결합 | 균형잡힌 검색 |
| **mix** (기본값) | hybrid + 원본 청크 벡터 검색 | **권장** |
| **bypass** | 검색 없이 LLM 직접 | 디버깅용 |

---

## 4. 점진적 업데이트 — GraphRAG와의 가장 큰 차이

```
=== GraphRAG: 새 문서 추가 시 ===
전체 커뮤니티 해체 → 재구축 (1,399 커뮤니티 × 2 × 5,000 토큰)

=== LightRAG: 새 문서 추가 시 ===
새 문서 → 엔터티/관계 추출 → 기존 그래프에 합집합
V_new = V_hat ∪ V'
E_new = E_hat ∪ E'
+ 동일 엔터티 중복 제거 (이름 매칭 + 설명 병합)
```

**기존 그래프를 건드리지 않고 새 노드/엣지만 추가**. 커뮤니티 재구축이 없으므로 계산 비용이 새 문서 크기에만 비례.

---

## 5. 실험 결과

### 5.1 데이터셋

| 데이터셋 | 문서 수 | 총 토큰 |
|---------|--------|---------|
| Agriculture | 12 | 2,017,886 |
| CS | 10 | 2,306,535 |
| Legal | 94 | 5,081,069 |
| Mix | 61 | 619,009 |

각 125개 질의, GPT-4o-mini로 쌍별 비교 평가.

### 5.2 LightRAG vs GraphRAG (핵심 비교)

| 메트릭 | Agriculture | CS | Legal | Mix |
|--------|-----------|-----|-------|-----|
| **포괄성** | 45.6 vs **54.4** | 48.4 vs **51.6** | 48.4 vs **51.6** | **50.4** vs 49.6 |
| **다양성** | 22.8 vs **77.2** | 40.8 vs **59.2** | 26.4 vs **73.6** | 36.0 vs **64.0** |
| **역량강화** | 41.2 vs **58.8** | 45.2 vs **54.8** | 43.6 vs **56.4** | **50.8** vs 49.2 |
| **종합** | 45.2 vs **54.8** | 48.0 vs **52.0** | 47.2 vs **52.8** | **50.4** vs 49.6 |

(왼쪽: GraphRAG 승률, 오른쪽: LightRAG 승률)

**핵심**: LightRAG가 대부분 메트릭에서 우세. 특히 **다양성에서 59-77% 승률**로 압도적. Mix에서만 GraphRAG가 근소하게 우세.

### 5.3 vs 다른 베이스라인 (Legal 데이터셋 종합 승률)

| 비교 대상 | LightRAG 승률 |
|---------|-------------|
| vs NaiveRAG | **84.8%** |
| vs RQ-RAG | **85.6%** |
| vs HyDE | **73.6%** |
| vs GraphRAG | **52.8%** |

### 5.4 토큰 비용 비교 (Legal 데이터셋)

| | GraphRAG | LightRAG |
|--|---------|---------|
| 검색당 토큰 | ~**610,000** | **< 100** |
| 검색당 API 호출 | **수백 회** | **1-2회** |
| 업데이트 비용 | 전체 재구축 | 새 문서만 처리 |

---

## 6. Ablation Study

| 변형 | 효과 |
|------|------|
| **-High** (Low-level만) | 포괄성 크게 하락 — 구체적 답만 가능, 넓은 맥락 파악 못함 |
| **-Low** (High-level만) | 관계/테마는 잡지만 세부 엔터티 정보 부족 |
| **-Origin** (원본 텍스트 제거) | **성능 하락 없음!** 오히려 일부 개선 — 그래프가 충분한 정보 보존 |
| **Hybrid** (전체) | 넓이(High) + 깊이(Low) 균형 — 최적 성능 |

**가장 흥미로운 발견**: 원본 텍스트를 제거해도 성능이 떨어지지 않는다는 것은, 그래프 인덱싱이 **핵심 정보를 충분히 추출**하고 있음을 의미한다.

---

## 7. 구현 세부사항

### 기본 설정값

| 항목 | 값 |
|------|-----|
| 청크 크기 | 1,200 토큰 |
| 청크 오버랩 | 100 토큰 |
| top\_k (엔터티/관계) | 40 |
| 토큰 예산 (엔터티) | 6,000 |
| 토큰 예산 (관계) | 8,000 |
| 토큰 예산 (전체) | 30,000 |
| 유사도 임계값 | 0.2 (코사인) |
| 요약 최대 토큰 | 1,200 |
| Gleaning 횟수 | 1 |

### 지원 백엔드

| 컴포넌트 | 옵션 |
|---------|------|
| 그래프 저장 | NetworkX (기본), Neo4j, Memgraph |
| 벡터 저장 | nano-vector-db (기본), Milvus, Qdrant, Faiss, PostgreSQL |
| KV 저장 | JSON (기본), MongoDB, PostgreSQL, Redis |

### 권장 모델

| 컴포넌트 | 권장 |
|---------|------|
| LLM | 32B+ 파라미터, 컨텍스트 ≥ 32K (64K 권장) |
| 임베딩 | BAAI/bge-m3 또는 text-embedding-3-large |
| 리랭커 | BAAI/bge-reranker-v2-m3 (mix 모드에 필요) |

### 사용법

```python
import lightrag

rag = lightrag.LightRAG(
    working_dir="./my_rag",
    llm_model_func=llm_model_func,
    embedding_func=embedding_func,
)

# 문서 추가 (점진적 업데이트)
rag.insert("새로운 문서 내용...")

# 검색 + 생성
result = rag.query("질의 내용", mode="mix")
```

---

## 8. GraphRAG vs LightRAG 선택 가이드

| 상황 | 추천 |
|------|------|
| 전역 요약 질의가 많음 | **GraphRAG** (커뮤니티 요약이 강점) |
| 구체적 + 추상적 질의 혼합 | **LightRAG** (이중 수준 검색) |
| 문서가 자주 업데이트됨 | **LightRAG** (점진적 업데이트) |
| 비용이 중요 | **LightRAG** (수천 배 적은 토큰) |
| 다양한 관점의 답변 필요 | **LightRAG** (다양성 77% 승률) |
| 대규모 코퍼스 (수백만 문서) | **LightRAG** (커뮤니티 재구축 불필요) |

---

## 9. 후속 연구

| 프로젝트 | 핵심 |
|---------|------|
| **MiniRAG** (2025.01) | 소형 LLM(8B 이하)에서도 그래프 RAG. 저장 75% 절감 |
| **VideoRAG** (2025.02, KDD 2026) | 비디오용 RAG. 134시간+ 영상 테스트 |
| **RAG-Anything** (2025) | 텍스트+이미지+테이블+수식 멀티모달 RAG |

---

## 10. 한계

1. **LLM 의존적 추출**: 엔터티/관계 추출 품질이 LLM 능력에 비례 (32B+ 권장)
2. **인덱싱 비용**: 초기 인덱싱에 (토큰/청크 크기)번의 LLM 호출 필요
3. **이름 기반 중복 제거**: 동일 엔터티의 다른 표현("AI" vs "인공지능")을 놓칠 수 있음
4. **LLM-as-judge 평가**: GPT-4o-mini 평가의 편향 가능성
5. **단일 벤치마크**: UltraDomain 데이터셋만 평가, 다국어/대규모 미검증
