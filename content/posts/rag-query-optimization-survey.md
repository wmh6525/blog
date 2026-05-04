---
title: "[서베이] RAG의 쿼리 성능을 높이는 모든 방법 — Rewriting, Expansion, Decomposition, Routing부터 RL까지"
date: 2026-04-22
tags: ["서베이", "RAG", "쿼리최적화", "HyDE", "강화학습"]
categories: ["ML/AI"]
summary: "RAG 시스템에서 검색 직전 쿼리 단계의 성능을 높이는 모든 기법을 정리한다. Query Rewriting (Query2Doc, RaFe), Expansion (HyDE, OPQE, CSQE), Decomposition (Self-Ask, RQ-RAG), Routing (Adaptive-RAG), Multi-Query (RAG-Fusion), Active (FLARE), 그리고 RL 기반 (DeepRetrieval, Search-R1) 까지 50+ 논문 종합."
math: true
toc: true
draft: false
---

## 왜 쿼리 단계가 중요한가?

RAG 시스템에서 **쿼리는 검색기의 입력**이다. 사용자가 입력한 쿼리가 그대로 검색기에 전달되면 다음 문제가 발생한다:

| 문제 | 예시 |
|------|------|
| **어휘 불일치** | 질의 "고양이 키우는 법" vs 문서 "feline care guide" |
| **너무 짧은 쿼리** | "당뇨병" → 어떤 측면? |
| **모호한 의도** | "사과" → 과일? 회사? |
| **다중 홉 필요** | "엘론 머스크가 설립한 회사의 CEO?" → 두 번 검색 |
| **대화 맥락 누락** | "그건 언제 생겼어?" → 무엇을 가리키는지 모름 |

쿼리를 **그대로 쓰지 말고 변형/확장/분해**하면 검색 품질이 크게 올라간다. 이 글은 쿼리 단계 최적화 기법을 **8개 카테고리, 50+ 논문**으로 정리한다.

---

## 전체 지도

```
[원본 사용자 쿼리]
        ↓
   ┌────┴────┐
   ↓         ↓
[Routing]  [Decomposition]
 (어디로?) (분해할까?)
   ↓         ↓
[Rewriting / Expansion]
 (재작성 / 확장)
   ↓
[Multi-Query]
 (여러 쿼리 변형)
   ↓
[Active / Iterative]
 (생성 중 재검색)
   ↓
   검색기
```

---

## 1. Query Rewriting (쿼리 재작성)

원본 쿼리를 더 검색 친화적인 형태로 **다시 쓰는** 방식.

### 1.1 Query2Doc (Microsoft, EMNLP 2023)

- **arXiv**: 2303.07678
- **방법**: Few-shot으로 LLM에게 가상 답변 문서를 생성시키고 → `[원본 쿼리] + [가상 문서]`를 새 쿼리로 사용
- **결과**: BM25 기준 MS-MARCO와 TREC DL에서 +3% ~ +15% 향상
- **언제**: Sparse 검색기(BM25)에서 어휘 불일치 문제가 심할 때

### 1.2 Rewrite-Retrieve-Read (Microsoft, EMNLP 2023)

- **arXiv**: 2305.14283
- **방법**: 기존 "retrieve-then-read" 대신 **rewrite-retrieve-read**. T5 rewriter를 RL로 학습 (보상: 다운스트림 LM loss)
- **결과**: HotpotQA, AmbigNQ, PopQA에서 개선
- **장점**: 학습된 T5가 few-shot LLM rewrite보다 우수

### 1.3 RaFe (EMNLP Findings 2024)

- **arXiv**: 2405.14431
- **방법**: 공개된 **Reranker를 보상 모델로 활용**해 query rewriter 학습 (DPO 또는 PPO)
- **장점**: **인간 라벨 불필요** — reranker 점수가 자연스러운 보상
- **언제**: 도메인 라벨 데이터가 없을 때

### 1.4 Crafting the Path (2024)

- **arXiv**: 2407.12529
- **방법**: 3단계 구조화 재작성:
  1. Query Concept Comprehension
  2. Query Type Identification
  3. Expected Answer Extraction

### 1.5 ERRR (2024)

- **arXiv**: 2411.07820
- **방법**: Extract-Refine-Retrieve-Read. LLM의 파라메트릭 지식을 먼저 추출 → 그 빈틈에 맞는 쿼리로 최적화

### 1.6 DeepRetrieval (COLM 2025)

- **arXiv**: 2503.00223
- **방법**: 3B LLM을 **RL로 학습**해 쿼리 생성. 보상 = 검색 recall. **지도 학습 라벨 불필요** (DeepSeek-R1-Zero 스타일)
- **결과**: NQ/TriviaQA에서 GPT-4o 수준 달성
- **한계**: Sparse 검색기에는 강하지만 dense에서는 오히려 성능 하락

---

## 2. Query Expansion (쿼리 확장)

원본 쿼리에 **추가 단서**를 더하는 방식.

### 2.1 Pseudo-Relevance Feedback (고전)

- **Rocchio (1971), RM3 (Lavrenko & Croft 2001)**
- 초기 검색 top-k에서 단어 추출 → 쿼리에 추가
- **장점**: 학습 무필요, 강력한 베이스라인. **단점**: 토픽 드리프트

### 2.2 HyDE (ACL 2023)

- **arXiv**: 2212.10496
- **방법**: LLM이 가상 문서 생성 → Contriever로 임베딩 → 임베딩으로 검색
- **결과**: 비지도 SOTA Contriever 능가, fine-tuned 검색기와 경쟁
- **한계**: 환각된 가상 문서가 long-tail 쿼리에서 검색을 왜곡

### 2.3 OPQE (2025) — HyDE + RL

- **arXiv**: 2510.17139
- **방법**: HyDE의 가상 문서 생성을 PPO로 학습 (보상 = NDCG)
- **결과**: Sparse + Dense **양쪽에서** 일관된 성능 향상
- **상세 리뷰**: [OPQE 리뷰](opqe-review.md)

### 2.4 docTTTTTquery (2019) — 인덱스 측 역방향

- **arXiv**: 1904.08375
- **방법**: 검색 시점이 아니라 **인덱싱 시점**에 T5로 가상 쿼리 생성 → 문서에 append → BM25 인덱스 구축
- **장점**: 추론 비용 0, 인덱싱에만 비용
- **결과**: MS-MARCO MRR@10 0.184 → 0.277

### 2.5 GAR (ACL 2021)

- **arXiv**: 2009.08553
- **방법**: 3가지 보강 생성 (가능한 답변, 컨텍스트, 제목) → 각각 검색 → 결과 융합

### 2.6 CSQE — Corpus-Steered Query Expansion (EACL 2024)

- **arXiv**: 2402.18031
- **방법**: PRF + LLM 하이브리드. LLM이 초기 검색된 문서에서 핵심 문장 선택 → 코퍼스에 grounded된 확장
- **장점**: 환각 감소

### 2.7 GenQREnsemble (ECIR 2024)

- **arXiv**: 2404.03746
- **방법**: 여러 prompt로 paraphrase → ensemble
- **결과**: nDCG@10 +18%, MAP +24%

### 2.8 LLM-QE (2025)

- **arXiv**: 2502.17057
- **방법**: 검색기의 ranking 선호에 맞춰 expansion LLM을 **DPO로 정렬**
- **결과**: Contriever zero-shot 대비 +8% 이상

### 2.9 LLM 기반 QE의 실패 분석 (SIGIR 2025)

- **arXiv**: 2505.12694
- **발견**: LLM이 쿼리 지식이 없거나 쿼리가 매우 모호할 때 **오히려 성능 저하**
- **시사점**: 항상 expansion이 답은 아님

---

## 3. Query Decomposition (쿼리 분해)

복잡한 쿼리를 **여러 sub-question**으로 분해.

### 3.1 Self-Ask (2022)

- **arXiv**: 2210.03350
- **방법**: 모델이 명시적으로 follow-up 질문을 던지고 답 → 최종 답
- **벤치마크**: Bamboogle (Self-Ask 논문에서 만든 데이터셋)

### 3.2 Least-to-Most Prompting (2022)

- **arXiv**: 2205.10625
- **방법**: 복잡한 문제 → 순차적으로 더 간단한 sub-problem으로 분해, 이전 답 활용
- **결과**: SCAN 길이 분할에서 99% (CoT는 16%)

### 3.3 Decomposed Prompting / DecomP (ICLR 2023)

- **arXiv**: 2210.02406
- **방법**: 모듈식 — 각 sub-task를 전문 prompt-LLM 모듈에 위임. 재귀적 분해 가능
- **결과**: long-context multi-hop QA, ODQA에서 강함

### 3.4 RQ-RAG (2024)

- **arXiv**: 2404.00610
- **방법**: LLM을 end-to-end 학습 — 매 스텝 **rewrite / decompose / disambiguate / answer-directly** 중 선택
- **결과**: Llama2-7B에서 single-hop QA +1.9 평균

### 3.5 RA-ISF (ACL Findings 2024)

- **arXiv**: 2403.06840
- **방법**: 3-모듈 self-feedback 루프 (자기 지식 확인 → 패시지 관련성 → 질문 분해)
- **결과**: 5개 데이터셋 중 4개에서 SOTA

---

## 4. Query Routing / Adaptive Retrieval

**언제 검색할지, 어떤 전략을 쓸지** 결정.

### 4.1 Adaptive-RAG (NAACL 2024)

- **arXiv**: 2403.14403
- **방법**: 작은 LM 분류기가 쿼리 복잡도 예측 (A/B/C):
  - A: 검색 불필요 (LLM 파라메트릭 지식)
  - B: Single-hop 검색
  - C: Multi-hop 반복 검색
- **장점**: 가장 저렴한 충분한 전략으로 라우팅

### 4.2 Self-RAG (ICLR 2024 Oral)

- **arXiv**: 2310.11511
- **방법**: 어휘에 reflection token 추가 (Retrieve, IsRel, IsSup, IsUse). LLM이 스스로 검색 시점/품질 판단
- **결과**: 7B/13B가 ChatGPT, Llama2-chat-RAG 능가

### 4.3 SR-RAG / Self-Routing RAG (2025)

- **arXiv**: 2504.01018
- **방법**: 선택적 검색 + 지식 verbalization 결합. LLM 자체가 지식 소스 역할 가능

### 4.4 RAGRouter (2025)

- **arXiv**: 2505.23052
- **방법**: 여러 검색기/데이터스토어 간 라우팅 — 지식 표현 + RAG 능력 벡터 학습

### 4.5 Uncertainty 기반 적응 검색 (2025)

- **arXiv**: 2501.12835
- **발견**: Uncertainty 기반 검색 결정이 분류기 파이프라인보다 자주 우수
- **시사점**: "검색 안 하기"를 잘 하는 것이 강력한 추론과 상관

---

## 5. Multi-Query / Diversification

**여러 쿼리 변형**을 만들어 검색 → 결과 융합.

### 5.1 RAG-Fusion (2024)

- **arXiv**: 2402.03367
- **방법**: 원본 쿼리 + LLM 생성 4개 변형 → 독립 검색 → **Reciprocal Rank Fusion (RRF)**, k=60
- **결과**: vanilla RAG 대비 정확도 +8-10%, 포괄성 +30-40%
- **공식**:

$$\text{RRF}(d) = \sum_{q \in Q} \frac{1}{\text{rank}_q(d) + k}$$

### 5.2 LangChain MultiQueryRetriever

- **사용법**: LLM이 3개 paraphrase 생성 → 각각 검색 → 결과 deduplicate
- **코드**: `MultiQueryRetriever.from_llm(retriever=base, llm=llm)`

---

## 6. Step-Back Prompting (DeepMind, ICLR 2024)

- **arXiv**: 2310.06117
- **저자**: Zheng et al., Google DeepMind
- **방법**: 2단계
  1. 질문을 **상위 개념/원리로 추상화**
  2. 그 추상화 위에서 추론
- RAG 적용: step-back 질문을 검색 쿼리로 사용 → 더 좋은 배경 컨텍스트 확보

### 예시

```
원본 질문: "온도 1700K, 압력 50 atm에서 이상기체 X mol의 부피는?"
       ↓ Step Back
추상화: "이상기체 법칙은 무엇이며 어떻게 적용하나?"
       ↓ 검색 + 추론
원리(PV=nRT) 검색 → 적용 → 정답
```

**결과**: PaLM-2L에서 STEM, Knowledge QA, Multi-Hop +27%

---

## 7. Active / Iterative Query

**생성 중 동적으로** 재검색.

### 7.1 FLARE (EMNLP 2023)

- **arXiv**: 2305.06983
- **방법**: 생성 중 다음 문장 임시 예측 → 저신뢰 토큰 발견 시 그 문장을 쿼리로 재검색 → 재생성

### 7.2 IRCoT (ACL 2023)

- **arXiv**: 2212.10509
- **방법**: CoT 추론 단계와 검색을 교차 — 각 새 CoT 문장이 다음 검색 쿼리
- **결과**: GPT-3에서 검색 +21점, QA +15점

### 7.3 ReAct (ICLR 2023)

- **arXiv**: 2210.03629
- **방법**: Thought / Action / Observation 교차 생성

### 7.4 DRAGIN (ACL 2024)

- **arXiv**: 2403.10081
- **방법**: RIND 모듈이 토큰 불확실성 + 영향력 + 의미를 종합해 **언제 검색할지** 결정. QFS 모듈은 self-attention 가중치로 쿼리 생성

### 7.5 Search-R1 (2025)

- **arXiv**: 2503.09516
- **방법**: PPO/GRPO로 LLM이 추론과 검색을 교차하도록 RL 학습. Outcome-only reward
- **결과**: Qwen2.5-7B에서 RAG 대비 +41%
- **상세 리뷰**: [Search-R1 리뷰](search-r1-review.md)

### 7.6 R1-Searcher (2025)

- **arXiv**: 2503.05592
- **방법**: 2단계 outcome-supervised RL — Stage 1: 언제 검색할지, Stage 2: 어떻게 사용할지
- **결과**: GPT-4o-mini 능가 (multi-hop QA)

---

## 8. Conversational Query Rewriting

**다중 턴 대화**에서 맥락을 결합한 재작성.

### 8.1 CONQRR (EMNLP 2022)

- **arXiv**: 2112.08558
- **방법**: SCST 강화학습으로 대화 query rewriter 학습. 검색 reward와 정렬

### 8.2 ConvGQR (ACL 2023)

- **arXiv**: 2305.15645
- **방법**: Query rewriter PLM + answer generation PLM + knowledge infusion 결합

### 8.3 QReCC (NAACL 2021)

- **arXiv**: 2010.04898
- **데이터셋**: 14K 대화, 80K QA 쌍 — 대화형 쿼리 재작성 표준 벤치마크

### 8.4 ConvSearch-R1 (2025)

- **arXiv**: 2505.15776
- **방법**: 외부 rewrite supervision 없는 self-driven CQR. 2단계 RL (Self-Driven Policy Warm-Up + rank-incentive reward shaping)
- **결과**: TopiOCQA에서 3B로 +10%

**Coreference resolution만으로는 부족**: 생성형 재작성이 ellipsis와 토픽 드리프트도 처리.

---

## 9. Query Embedding 개선

쿼리를 **다르게 임베딩**.

### 9.1 INSTRUCTOR (ACL 2023)

- **arXiv**: 2212.09741
- **핵심**: 각 입력을 자연어 task instruction과 함께 임베딩. 330개 task instruction으로 contrastive 학습
- **결과**: 70-task MTEB SOTA, +3.4% 평균

### 9.2 E5의 비대칭 prefix

- `query: ...` / `passage: ...` 접두사 사용
- 같은 인코더로 비대칭 신호 주입

### 9.3 BGE-M3 (BAAI, 2024)

- **arXiv**: 2402.03216
- **특징**: Multi-Linguality (100+) + Multi-Granularity (8192 토큰) + Multi-Functionality (dense + sparse + ColBERT)

### 9.4 Promptagator (ICLR 2023)

- **arXiv**: 2209.11755
- **방법**: 8-shot LLM prompt로 task-specific 합성 쿼리 생성 → 도메인 특화 dual encoder 학습

---

## 10. RL 기반 쿼리 최적화 종합

| 방법 | arXiv | 보상 | 방향 |
|------|-------|------|------|
| **CONQRR** (2022) | 2112.08558 | 검색 recall | 대화형 재작성 |
| **RaFe** (2024) | 2405.14431 | Reranker 점수 | 일반 재작성, 라벨 무 |
| **OPQE** (2025) | 2510.17139 | NDCG/Hit | 가상 문서 생성 |
| **DeepRetrieval** (COLM 2025) | 2503.00223 | Recall@K | 쿼리 생성 |
| **Search-R1** (2025) | 2503.09516 | 정답 EM | 다중 턴 검색 에이전트 |
| **R1-Searcher** (2025) | 2503.05592 | 정답 정확도 | 언제+어떻게 검색 |
| **ConvSearch-R1** (2025) | 2505.15776 | Rank-incentive | 대화형 |
| **LLM-QE** (2025) | 2502.17057 | Rank + answer | 쿼리 확장 |

---

## 11. Query Caching / Speculative

### 11.1 GPTCache (Zilliz)

- **방법**: Semantic cache — 들어오는 쿼리를 임베딩 → 캐시된 쿼리들과 cosine 유사도 매칭 → threshold 이상이면 캐시된 답변 반환
- **결과**: 캐시 히트율 61.6-68.8%, API 호출 68.8% 감소
- **속도**: 캐시 히트 ~3-8ms vs 전체 파이프라인 500-2000ms

### 11.2 GPT Semantic Cache (2024)

- **arXiv**: 2411.05276
- **연구**: 지연/비용 분석

---

## 12. 종합 비교 — 증상별 처방

| 증상 | 추천 기법 |
|------|---------|
| 어휘 불일치 (BM25) | docTTTTTquery (인덱스 시) 또는 Query2Doc (쿼리 시) |
| 짧고 모호한 쿼리 | HyDE, Query2Doc, GAR |
| Multi-hop / 합성적 | Self-Ask, IRCoT, Least-to-Most, DecomP, RQ-RAG |
| 다양한 난이도 혼재 | Adaptive-RAG, Self-RAG |
| 대화형 / 다중 턴 | CONQRR, ConvGQR, ConvSearch-R1 |
| 개념/원리 질문 | **Step-Back Prompting** |
| 장문 생성 사실성 | FLARE, DRAGIN, Self-RAG |
| 다중 데이터스토어 | Router (LangChain RouterQueryEngine, RAGRouter) |
| 비용/지연 중요 | Semantic Cache (GPTCache) + Adaptive-RAG |
| 도메인 검색기, 라벨 없음 | GenQ / GPL / Promptagator |
| LLM이 환각 | CSQE (코퍼스 grounded) |
| RL 학습 + 라벨 없음 | RaFe (reranker reward), DeepRetrieval, OPQE |

---

## 13. 프로덕션 패턴

### LangChain

```python
# 1. Multi-Query Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever
retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

# 2. Self-Query Retriever (메타데이터 필터 자동 추출)
from langchain.retrievers.self_query.base import SelfQueryRetriever
retriever = SelfQueryRetriever.from_llm(llm, vectorstore, doc_info, metadata_info)

# 3. Ensemble Retriever (BM25 + Dense + RRF)
from langchain.retrievers import EnsembleRetriever
retriever = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])

# 4. HyDE
from langchain.chains import HypotheticalDocumentEmbedder
embeddings = HypotheticalDocumentEmbedder.from_llm(llm, base_embeddings, prompt_key="web_search")
```

### LlamaIndex

```python
# 1. HyDE
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
hyde = HyDEQueryTransform(include_original=True)

# 2. Step-Back Decompose
from llama_index.core.indices.query.query_transform import StepDecomposeQueryTransform

# 3. Sub-Question Engine (Self-Ask 스타일)
from llama_index.core.query_engine import SubQuestionQueryEngine

# 4. Router (인덱스 라우팅)
from llama_index.core.query_engine import RouterQueryEngine
```

### 2025 표준 스택

```
[저비용 경로]
  Paraphrase rewrite → Hybrid (BM25 + BGE-M3) → RRF → Reranker

[적응형]
  Adaptive-RAG 분류기 → no-retrieval / single / multi-hop

[대화형]
  Coreference-aware rewriter (CONQRR class) → 검색

[캐싱 레이어]
  GPTCache 또는 Redis Vector Cache (LLM/검색기 앞)
```

---

## 14. 2025-2026 트렌드 5가지

1. **Retrieval-as-reward RL의 성숙**: Search-R1, R1-Searcher, DeepRetrieval, OPQE, LLM-QE 모두 **golden query 라벨 없이** 학습
2. **적응형 / 자기인식 검색**: "언제 검색하지 **않을지**" 결정이 일관되게 우수
3. **하이브리드 expansion** (코퍼스 grounded + LLM): CSQE, ERRR이 long-tail에서 순수 LLM expansion 능가
4. **통합 임베딩 모델**: BGE-M3, INSTRUCTOR가 비대칭 prefix 트릭 대체
5. **Process-reward 모델**: 쿼리 최적화의 미해결 과제 (Tencent 서베이 arXiv:2412.17558)

---

## 15. 의사결정 흐름도

```
사용자 쿼리
    │
    ├─ 다중 턴 대화? ──Yes──→ CONQRR / ConvGQR
    │      No
    │
    ├─ 쿼리 복잡도 분류 (Adaptive-RAG)
    │      ├─ 단순 (LLM 지식으로 OK) → 검색 생략
    │      ├─ Single-hop → 일반 RAG
    │      └─ Multi-hop → 분해 (Self-Ask, IRCoT, RQ-RAG)
    │
    ├─ 어휘 불일치? ──Yes──→ Query2Doc, HyDE, OPQE
    │
    ├─ 개념적 질문? ──Yes──→ Step-Back Prompting
    │
    ├─ 다양성 필요? ──Yes──→ RAG-Fusion (Multi-Query + RRF)
    │
    ├─ 생성 중 사실성? ──Yes──→ FLARE, DRAGIN, Self-RAG
    │
    └─ RL로 학습 가능? ──Yes──→ RaFe, OPQE, Search-R1
```

---

## 16. 카테고리별 핵심 논문 1개씩

| 카테고리 | 대표 논문 | 한 줄 |
|---------|---------|------|
| Rewriting | **Query2Doc** (Microsoft, 2023) | LLM이 가상 답변 → 원본과 concat |
| Expansion | **HyDE** (CMU, 2023) | 가상 문서 임베딩으로 검색 |
| Decomposition | **Self-Ask** (Princeton, 2022) | Follow-up 질문으로 분해 |
| Routing | **Adaptive-RAG** (KAIST, 2024) | 복잡도 분류 → 전략 라우팅 |
| Multi-Query | **RAG-Fusion** (2024) | 변형 + RRF |
| Step-Back | **Step-Back** (DeepMind, 2024) | 추상화 후 추론 |
| Active | **FLARE** (CMU, 2023) | 저신뢰 시 재검색 |
| Conversational | **CONQRR** (2022) | RL 학습 대화 재작성 |
| Embedding | **INSTRUCTOR** (2023) | Instruction-tuned |
| RL | **OPQE** (2025) | HyDE + RL |

---

## 17. 관련 블로그 포스트

- [OPQE 상세 리뷰](opqe-review.md) — HyDE + RL
- [Search-R1 상세 리뷰](search-r1-review.md) — RL 검색 에이전트
- [CCS 상세 리뷰](cycle-consistent-search-review.md) — 정답 없는 RL
- [CRAG 상세 리뷰](crag-review.md) — 검색 후 교정
- [Self-RAG는 RAG 동향 서베이 참조](rag-survey-2026.md)
- [RAG 학습 합성 데이터 서베이](rag-synthetic-data-survey.md) — Promptagator, GPL
- [RAG End-to-end 학습 서베이](rag-end-to-end-training-survey.md)
- [RAG 검색 속도 최적화](rag-latency-optimization.md) — Caching 관련

---

## 참고 자료 (50+)

### Rewriting
- [Query2Doc (arXiv:2303.07678)](https://arxiv.org/abs/2303.07678)
- [Rewrite-Retrieve-Read (arXiv:2305.14283)](https://arxiv.org/abs/2305.14283)
- [RaFe (arXiv:2405.14431)](https://arxiv.org/abs/2405.14431)
- [DeepRetrieval (arXiv:2503.00223)](https://arxiv.org/abs/2503.00223)

### Expansion
- [HyDE (arXiv:2212.10496)](https://arxiv.org/abs/2212.10496)
- [OPQE (arXiv:2510.17139)](https://arxiv.org/abs/2510.17139)
- [docTTTTTquery (arXiv:1904.08375)](https://arxiv.org/abs/1904.08375)
- [GAR (arXiv:2009.08553)](https://arxiv.org/abs/2009.08553)
- [CSQE (arXiv:2402.18031)](https://arxiv.org/abs/2402.18031)
- [LLM-QE (arXiv:2502.17057)](https://arxiv.org/abs/2502.17057)

### Decomposition
- [Self-Ask (arXiv:2210.03350)](https://arxiv.org/abs/2210.03350)
- [Least-to-Most (arXiv:2205.10625)](https://arxiv.org/abs/2205.10625)
- [DecomP (arXiv:2210.02406)](https://arxiv.org/abs/2210.02406)
- [RQ-RAG (arXiv:2404.00610)](https://arxiv.org/abs/2404.00610)

### Routing
- [Adaptive-RAG (arXiv:2403.14403)](https://arxiv.org/abs/2403.14403)
- [Self-RAG (arXiv:2310.11511)](https://arxiv.org/abs/2310.11511)

### Multi-Query / Step-Back
- [RAG-Fusion (arXiv:2402.03367)](https://arxiv.org/abs/2402.03367)
- [Step-Back (arXiv:2310.06117)](https://arxiv.org/abs/2310.06117)

### Active
- [FLARE (arXiv:2305.06983)](https://arxiv.org/abs/2305.06983)
- [IRCoT (arXiv:2212.10509)](https://arxiv.org/abs/2212.10509)
- [ReAct (arXiv:2210.03629)](https://arxiv.org/abs/2210.03629)
- [DRAGIN (arXiv:2403.10081)](https://arxiv.org/abs/2403.10081)
- [Search-R1 (arXiv:2503.09516)](https://arxiv.org/abs/2503.09516)
- [R1-Searcher (arXiv:2503.05592)](https://arxiv.org/abs/2503.05592)

### Conversational
- [CONQRR (arXiv:2112.08558)](https://arxiv.org/abs/2112.08558)
- [ConvGQR (arXiv:2305.15645)](https://arxiv.org/abs/2305.15645)
- [ConvSearch-R1 (arXiv:2505.15776)](https://arxiv.org/abs/2505.15776)

### 임베딩
- [INSTRUCTOR (arXiv:2212.09741)](https://arxiv.org/abs/2212.09741)
- [BGE-M3 (arXiv:2402.03216)](https://arxiv.org/abs/2402.03216)
- [Promptagator (arXiv:2209.11755)](https://arxiv.org/abs/2209.11755)

### 서베이
- [Tencent Query Optimization Survey (arXiv:2412.17558)](https://arxiv.org/abs/2412.17558)
- [Query Expansion in PLM/LLM Era (arXiv:2509.07794)](https://arxiv.org/abs/2509.07794)
