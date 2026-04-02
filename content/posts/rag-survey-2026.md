---
title: "[서베이] 최신 RAG 동향 총정리 (2026) — 모델, 워크플로우, 프로덕션 패턴"
date: 2026-04-02
tags: ["연구노트", "RAG", "LLM", "검색증강생성"]
categories: ["ML/AI"]
summary: "2026년 기준 RAG(Retrieval-Augmented Generation)의 최신 동향을 총정리한다. 임베딩 모델 SOTA, 청킹 전략, 고급 RAG 기법(GraphRAG, Self-RAG, Agentic RAG), 벡터 DB, 프레임워크, 프로덕션 아키텍처까지."
math: true
toc: true
draft: false
---

## 1. RAG의 진화: Naive → Advanced → Modular → Agentic

### 원조 RAG (Lewis et al., NeurIPS 2020)

FAIR(Meta)의 Patrick Lewis et al.이 제안. BART + DPR(Dense Passage Retrieval) + Wikipedia 인덱스를 결합한 최초의 검색 증강 생성 모델. 3개 Open-domain QA 벤치마크에서 SOTA 달성.

### 세대별 진화

| 세대 | 핵심 | 정확도 |
|------|------|-------|
| **Naive RAG** | 청킹 → 임베딩 → 검색 → 생성 | ~25% |
| **Advanced RAG** | + 쿼리 변환, 리랭킹, 필터링 | ~90% |
| **Modular RAG** | 모듈 단위 교체 가능 (Search, Memory, Fusion, Routing) | 유연 |
| **Agentic RAG** | LLM이 검색 전략을 자율 결정 | 최고 |

---

## 2. 표준 워크플로우

```
사용자 질의
  ↓
[1] 쿼리 변환 (재작성, 분해, HyDE)
  ↓
[2] 하이브리드 검색 (Dense + Sparse 병렬)
  ↓
[3] RRF 합산 (Reciprocal Rank Fusion)
  ↓
[4] 리랭킹 (Cross-encoder, top-k 선별)
  ↓
[5] 프롬프트 구성 (검색 결과 + 질의)
  ↓
[6] LLM 생성
  ↓
[7] 자체 검증 (Self-RAG / CRAG)
  ↓
최종 응답 + 인용 출처
```

---

## 3. 임베딩 모델 SOTA (2026년 3월 기준)

### Dense Retriever

| 모델 | 제공자 | 파라미터 | 차원 | 핵심 특징 |
|------|--------|---------|------|----------|
| **Gemini Embedding 2** | Google | - | 3,072 | 5개 모달리티 (텍스트/이미지/영상/오디오/PDF), MRL |
| **Qwen3-Embedding-8B** | Alibaba | 8B | 1,024 | MTEB 1위급, 100+ 언어, Apache 2.0 |
| **Cohere Embed v4** | Cohere | - | 256-1,536 | 멀티모달, 128K 컨텍스트, Matryoshka |
| **Jina Embeddings v4** | Jina AI | 3.8B | 2,048 | Qwen2.5-VL 기반, 3개 LoRA 어댑터, 텍스트+이미지+PDF |
| **voyage-3-large** | Voyage AI | - | - | OpenAI 대비 9.74% 성능 향상, int8/binary 양자화 |
| **BGE-M3** | BAAI | - | - | Dense + Multi-vector + Sparse 동시 지원, 8192 토큰 |
| **text-embedding-3-large** | OpenAI | - | 3,072 | MRL 지원, 안정적 |

**트렌드**: 멀티모달 임베딩이 표준이 되었다. MRL(Matryoshka Representation Learning)로 차원을 유동적으로 축소하여 비용/성능 트레이드오프가 가능.

### Sparse Retriever

| 모델 | 특징 |
|------|------|
| **BM25** | 키워드 검색의 기본, 비용 0, 여전히 필수 베이스라인 |
| **SPLADE v2/v3** | 학습된 희소 검색 — "study"가 "learn", "research"도 활성화 |

### Late Interaction 모델

| 모델 | 특징 |
|------|------|
| **ColBERTv2** | 토큰 수준 MaxSim, 문서 표현 사전 계산 가능 |
| **Jina-ColBERT-v2** | 89개 언어, 560M, 8192 토큰, Matryoshka |
| **ColPali** | PaliGemma 기반, PDF를 이미지로 처리 — OCR 불필요 |

### 리랭커

| 모델 | 특징 |
|------|------|
| **Cohere Rerank 3.5** | 100+ 언어, 4096 컨텍스트, BEIR SOTA |
| **BGE-reranker-v2-m3** | 오픈소스, Apache 2.0, 셀프호스팅 |
| **Qwen3-Reranker** | 0.6B/4B/8B 크기 선택 가능 |

---

## 4. 하이브리드 검색

2026년 현재 모든 주요 벡터 DB가 하이브리드 검색을 지원한다.

```
[1] Sparse (BM25/SPLADE) ──┐
                           ├── RRF 합산 ──→ 결과
[2] Dense (임베딩)    ──────┘
```

**Reciprocal Rank Fusion (RRF)**:

$$\text{RRF}(d) = \sum_{q \in Q} \frac{1}{\text{rank}_q(d) + k}, \quad k = 60$$

각 검색 방법에서의 순위를 역수로 합산. 단일 방법 대비 일관되게 성능 향상.

---

## 5. 청킹 전략

| 전략 | 방식 | 장점 | 단점 |
|------|------|------|------|
| **고정 크기** | 512 토큰 + 50 오버랩 | 빠르고 단순 | 의미 경계 무시 |
| **재귀 분할** | `\n\n` → `\n` → ` ` → `` 순으로 재귀 분할 | 구조 보존 | 최적 크기 결정 어려움 |
| **시맨틱 청킹** | 인접 문장 임베딩 유사도로 경계 탐지 | 의미 보존, ~70% 정확도 향상 | 계산 비용 |
| **Late Chunking** (Jina) | 전체 문서 먼저 임베딩 → 청크 표현 추출 | 청크 간 문맥 보존 | 긴 컨텍스트 모델 필요 |
| **Contextual Chunking** (Anthropic) | LLM으로 청크에 문서 맥락 설명 추가 | **67% 실패율 감소** | LLM 호출 비용 |

### Contextual Retrieval (Anthropic, 2024.09) 상세

"매출이 3% 성장했다"라는 청크에 **"이 청크는 Acme Corp의 2024년 2분기 실적 보고서에서 발췌한 것입니다"**라는 맥락을 LLM이 붙여줌.

| 구성 | 검색 실패율 감소 |
|------|-------------|
| Contextual Embeddings 단독 | **35%** |
| + Contextual BM25 | **49%** |
| + 리랭킹 | **67%** |

---

## 6. 고급 RAG 기법

### 6.1 GraphRAG (Microsoft, 2024.04)

문서에서 LLM으로 엔터티/관계를 추출하여 **지식 그래프** 구축 → Leiden 알고리즘으로 커뮤니티 분할 → 커뮤니티별 요약 생성.

```
문서 → 엔터티/관계 추출 → 지식 그래프
  → 커뮤니티 분할 (Leiden)
  → 커뮤니티 요약

질의 시:
  Global Search: 커뮤니티 요약으로 전체적 질의 응답
  Local Search: 엔터티 중심 팬아웃
  DRIFT Search: 로컬 + 커뮤니티 결합
```

**장점**: "이 문서 전체의 핵심 주제는?" 같은 전역 질의에 강함
**단점**: 인덱싱 비용이 높음 (LLM 호출 다수)

### 6.2 RAPTOR

재귀적 추상 요약 트리. 청크 → 클러스터링 → 요약 → 다시 클러스터링 → 요약... 을 반복하여 계층적 트리 구축.

```
[루트] 전체 문서 요약
  ├── [중간] 섹션별 요약
  │     ├── [리프] 원본 청크 1
  │     └── [리프] 원본 청크 2
  └── [중간] 섹션별 요약
        ├── [리프] 원본 청크 3
        └── [리프] 원본 청크 4
```

GraphRAG보다 단순하면서 다중 해상도 검색 가능.

### 6.3 Self-RAG (ICLR 2024)

LLM이 스스로:
1. **검색이 필요한지** 판단
2. 검색 결과의 **관련성** 평가
3. 자기 생성이 **증거에 의해 뒷받침되는지** 검증

모델에 "반성 토큰(reflection tokens)"을 학습시켜 할루시네이션 감소.

### 6.4 Corrective RAG (CRAG)

검색 결과의 품질을 경량 평가기로 3단계 분류:

| 등급 | 조치 |
|------|------|
| **Correct** | 그대로 사용 |
| **Ambiguous** | 쿼리 재구성 후 재검색 |
| **Incorrect** | 폐기 + 웹 검색 폴백 |

### 6.5 HyDE (Hypothetical Document Embeddings)

```
질의: "트랜스포머의 어텐션 메커니즘이란?"
  ↓ LLM이 가상 답변 생성
가상 답변: "어텐션 메커니즘은 쿼리와 키의 내적으로..."
  ↓ 가상 답변을 임베딩
  ↓ 이 임베딩으로 유사도 검색
```

질의-문서 간 분포 격차를 해소. **Multi-HyDE** (2025)는 여러 가상 답변을 생성하여 검색 커버리지 확대.

### 6.6 Agentic RAG

LLM이 **자율 오케스트레이터**로 작동:

```
질의 분석 → 검색 전략 선택 → 도구 호출
  → 중간 결과 검증 → 필요시 재검색
  → 최종 답변 합성
```

- **System 1**: 사전 정의된 모듈형 파이프라인 (빠르고 예측 가능)
- **System 2**: 자율 에이전트 추론 (유연하지만 느림)

**A-RAG** (2026.02): 계층적 검색 인터페이스 (키워드 검색, 시맨틱 검색, 청크 읽기)를 에이전트에 직접 노출.

### 6.7 RAG Fusion

```
원본 질의 → LLM이 N개 변형 생성
  ↓ 각 변형으로 독립 검색
  ↓ RRF로 결과 합산
```

리콜과 견고성 동시 향상.

### 6.8 Speculative RAG (ICLR 2025)

작은 RAG 모델이 여러 초안을 **병렬** 생성 → 큰 LLM이 최선의 초안을 검증·통합. 추론 수준의 speculative decoding을 답변 수준으로 확장.

### 6.9 MemoRAG

이중 시스템:
1. **경량 장거리 모델**: 코퍼스 전체의 글로벌 메모리에서 초안 답변 생성 + 검색 가이드
2. **강력한 모델**: 정밀 검색된 증거로 최종 답변 생성

모호한 질의, 비정형 지식에서 기존 RAG 대비 우수.

---

## 7. 벡터 데이터베이스

| DB | 최적 용도 | 규모 | 핵심 강점 |
|----|---------|------|----------|
| **Pinecone** | 가장 쉬운 프로덕션 시작 | 수십억 | 관리형, 탁월한 쿼리 속도 |
| **Weaviate** | 하이브리드 검색 | 대규모 | 내장 임베딩 모듈, 네이티브 하이브리드 |
| **Qdrant** | 필터링 + 성능 | 수십억 | Rust 기반, 정교한 메타데이터 필터링, 멀티벡터 |
| **Milvus/Zilliz** | 엔터프라이즈 대규모 | 수십억 | 최저 레이턴시, GPU 가속 |
| **Chroma** | 프로토타이핑 | 소규모 | 경량, 인프로세스 |
| **pgvector** | PostgreSQL 네이티브 | ~1억 | 트랜잭션 일관성, 별도 DB 불필요 |

---

## 8. 프레임워크 비교

| 프레임워크 | 최적 용도 | 오버헤드 | 핵심 강점 |
|-----------|---------|---------|----------|
| **LangChain/LangGraph** | 복잡한 에이전틱 워크플로우 | ~10-14ms | 최대 통합 생태계 |
| **LlamaIndex** | 문서 Q&A / 순수 RAG | ~6ms | 검색 40% 빠름, 정확도 35% 향상 |
| **Haystack** | 프로덕션 NLP, 규제 산업 | ~5.9ms | 타입 안전 컴포넌트, 단계별 계측 |
| **DSPy** | 프롬프트 최적화 | ~3.53ms | 최적 프롬프트 자동 탐색 |
| **RAGFlow** | 문서 이해 | - | 비주얼 DAG 에디터, PDF/테이블 추출 |

**실무 패턴**: LlamaIndex로 인제스션/인덱싱 + LangGraph로 오케스트레이션 조합이 일반적.

---

## 9. 평가: RAGAS

**RAGAS** (Retrieval Augmented Generation Assessment, EACL 2024)

참조 답변 없이(reference-free) RAG 품질을 평가:

| 메트릭 | 측정 대상 |
|--------|---------|
| **Faithfulness** | 답변의 모든 주장이 검색된 맥락으로 뒷받침되는가? |
| **Answer Relevancy** | 답변이 질문에 적절한가? |
| **Context Precision** | 관련 문서가 상위에 검색되었는가? |
| **Context Recall** | 관련 문서가 충분히 검색되었는가? |

2025-2026 확장: 멀티모달 Faithfulness, Tool Call Accuracy, Agent Goal Accuracy 추가.

---

## 10. 프로덕션 아키텍처 (2026)

### 엔터프라이즈 RAG = "지식 런타임"

```
┌─────────────────────────────────────────────┐
│                 거버넌스 레이어               │
│    감사 추적 · 편향 탐지 · 출처 추적 · 컴플라이언스   │
├─────────────────────────────────────────────┤
│  [1] 데이터 인제스션                          │
│    PDF/테이블/이미지 파싱 → 청킹 → 임베딩 → 메타데이터  │
├─────────────────────────────────────────────┤
│  [2] 검색 레이어                             │
│    하이브리드 검색 → 멀티 인덱스 라우팅 → 쿼리 변환    │
├─────────────────────────────────────────────┤
│  [3] 후처리 레이어                           │
│    리랭킹 → 컨텍스트 압축 → 관련성 필터링          │
├─────────────────────────────────────────────┤
│  [4] 생성 레이어                             │
│    프롬프트 구성 → LLM 추론 → 인용 생성          │
├─────────────────────────────────────────────┤
│  [5] 검증 레이어                             │
│    Self-RAG · CRAG · 할루시네이션 탐지          │
└─────────────────────────────────────────────┘
```

S&P 500 기업의 엔터프라이즈 RAG 도입이 **2025년에 280% 성장**.

### Long Context vs RAG

| | Long Context | RAG |
|--|-------------|-----|
| **강점** | Wikipedia QA, 단일 문서 전체 이해 | 다양한 소스, 대화형 질의, 실시간 데이터 |
| **비용** | 높은 토큰당 비용 | 낮은 셋업 비용 |
| **2026 정답** | **둘 다 사용**: 벡터 검색으로 관련 문서 선별 → 긴 컨텍스트로 추론 |

### RAG vs Fine-tuning

| | RAG | Fine-tuning |
|--|-----|-------------|
| **용도** | 변동성 높은/외부 지식, 실시간 데이터, 인용 필요 | 행동 일관성 (톤, 포맷, 분류, 정책) |
| **2026 합의** | 둘을 결합: "변동 지식은 검색에, 안정 행동은 파인튜닝에" |

---

## 11. KV 캐시 최적화 — 차세대 RAG 가속

| 기법 | 핵심 결과 |
|------|---------|
| **TurboRAG** (EMNLP 2025) | 청크별 KV 캐시 사전 계산 → TTFT **8.6배 감소** (4.13s → 0.65s) |
| **CacheGen** (SIGCOMM 2024) | KV 캐시 크기 3.5-4.3배 축소 |
| **DynamicKV** | KV 캐시의 1.7%만 유지하면서 90%+ 정확도 보존 |
| **Task-Aware Compression** | 30배 압축에서 RAG 대비 +7 포인트, 레이턴시 0.43s → 0.16s |

---

## 12. 패러다임 변화 요약

1. **RAG → Context Engine**: 단순 검색+생성에서 지능적 컨텍스트 계층으로 진화
2. **Agentic RAG**: LLM이 검색의 수동 소비자에서 자율 오케스트레이터로
3. **Reasoning RAG**: Chain-of-thought / System-2 사고를 검색 루프에 통합
4. **Multimodal-first**: 임베딩과 검색이 처음부터 다중 모달 지원 (Gemini, ColPali, Cohere v4)
5. **Cache-Augmented Generation**: KV 캐시 사전 계산으로 중복 인코딩 제거 (TurboRAG)
6. **Graph + Agent 수렴**: 그래프 지식과 에이전트 프레임워크의 융합

---

## 참고 자료

| 논문/자료 | 핵심 |
|---------|------|
| Lewis et al., NeurIPS 2020 (arXiv: 2005.11401) | 원조 RAG |
| GraphRAG (arXiv: 2404.16130) | 지식 그래프 기반 RAG |
| Self-RAG (ICLR 2024) | 자기 반성적 검색 |
| Speculative RAG (ICLR 2025, arXiv: 2407.08223) | 추론 수준 speculative decoding |
| Contextual Retrieval (Anthropic, 2024.09) | 문맥 청킹 |
| Late Chunking (arXiv: 2409.04701) | 임베딩 후 청킹 |
| A-RAG (arXiv: 2602.03442) | 계층적 에이전틱 RAG |
| TurboRAG (EMNLP 2025, arXiv: 2410.07590) | KV 캐시 가속 |
| RAGAS (arXiv: 2309.15217) | RAG 평가 프레임워크 |
| Agentic RAG Survey (arXiv: 2501.09136) | 에이전틱 RAG 서베이 |
| RAG Architectures Survey (arXiv: 2506.00054) | RAG 아키텍처 분류 체계 |
