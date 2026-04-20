---
title: "[서베이] RAG 학습용 합성 데이터 생성 논문 총정리 — E5-Mistral, Self-Instruct, InPars, Promptagator, Gecko, GPL, RAFT"
date: 2026-04-18
tags: ["서베이", "RAG", "합성데이터", "임베딩", "LLM"]
categories: ["ML/AI"]
summary: "도메인에 질의 라벨이 없을 때, 문서만으로 RAG 학습 데이터를 합성하는 11개 핵심 논문을 총정리한다. LLM 기반 Q&A 합성(E5-Mistral, Self-Instruct), 검색 특화 합성(InPars, Promptagator, Gecko), 생성기 특화(RAFT)까지."
math: true
toc: true
draft: false
---

## 왜 합성 데이터인가?

폐쇄 도메인 RAG 시스템을 만들 때 가장 큰 장벽:

> **문서는 있지만 (질의, 답변) 라벨이 없다.**

어노테이터를 고용하면 비싸고, 도메인이 자주 바뀌면 매번 라벨링해야 한다. 해결책은 **LLM을 사용해 학습 데이터 자체를 합성**하는 것.

이 글에서는 합성 데이터 생성 논문들을 다음 4가지 목적별로 정리한다:

1. **생성기(LLM) 학습용**: Self-Instruct, Evol-Instruct, RAFT
2. **검색기/임베딩 학습용**: E5-Mistral, InPars, Promptagator, Gecko, GPL, AugTriever
3. **인덱스 확장용**: Doc2Query, docTTTTTquery

---

## 1. 생성기 학습용 합성 데이터

### 1.1 Self-Instruct (ACL 2023)

- **저자**: Yizhong Wang et al. (UW, AI2)
- **핵심**: 175개의 seed task를 LLM이 스스로 확장 → 52K 인스트럭션 데이터

**부트스트랩 과정**:

```
[1] 175개 seed task (25 분류 + 150 개방형)
       ↓
[2] GPT-3가 in-context 예시 8개로 새 인스트럭션 생성
       ↓
[3] 분류/비분류 자동 판별
       ↓
[4] 인스턴스(입력+출력) 생성
       ↓
[5] ROUGE-L < 0.7 필터링 (중복 제거)
       ↓
최종 52K 인스트럭션, 82K 인스턴스
```

**결과**: GPT-3 + Self-Instruct 튜닝 → SuperNI에서 **+33점** 절대 개선. InstructGPT-001과 동등.

**의의**: Alpaca, WizardLM의 모태. 이후 모든 LLM instruction 데이터 생성의 원형.

### 1.2 Evol-Instruct / WizardLM (ICLR 2024)

- **저자**: Microsoft + Peking University
- **핵심**: 단순 인스트럭션을 **반복적으로 진화**시켜 복잡도 증가

**5가지 심화 연산자 (In-depth Evolving)**:

| 연산자 | 예시 |
|-------|------|
| **제약 추가** | "숫자를 정렬하라" → "오름차순으로, 중복 제거하며 정렬하라" |
| **심화** | "AI란?" → "AI의 철학적, 경제적, 윤리적 함의를 분석하라" |
| **구체화** | "동물을 설명하라" → "아프리카 사바나의 육식동물을 설명하라" |
| **추론 단계 증가** | 1-step → 3-step 문제 |
| **입력 복잡화** | 평문 → XML/JSON/표 |

**In-breadth Evolving**: 같은 도메인 내 **새로운 토픽** 생성 (다양성).

**Elimination Evolving**: 실패한 진화 제거 (정보 증가 없음, 응답 불가 등).

**결과**: WizardLM이 ChatGPT 대비 **고난도 과제**에서 선호됨. GPT-4 판정 기준 17/29 스킬에서 90%+ ChatGPT 능력.

### 1.3 RAFT (COLM 2024)

- **저자**: UC Berkeley (Gorilla 팀)
- **핵심**: 생성기를 RAG에 최적화하는 **방해 문서 혼합** 학습 데이터

**데이터 형식**:

```
P% (예: 80%):
  질의 + 정답 문서 D* + 방해 문서 D1, D2, D3
  → ##Reason: 문서에서 ##begin_quote##핵심 인용##end_quote##...
    ##Answer: 최종 답

(1-P)% (예: 20%):
  질의 + 방해 문서 D1, D2, D3 (정답 문서 없음!)
  → (파라메트릭 기억에서 답)
```

**핵심 통찰**: 항상 정답 문서를 주면 ($P = 100\%$) 모델이 검색에 과의존. **P = 60-80%**가 최적.

**결과**: LLaMA2-7B + RAFT가 HuggingFace API에서 GPT-3.5+RAG를 **+44.92%p** 능가.

상세 리뷰: [RAFT 논문 리뷰](raft-review.md)

---

## 2. 임베딩 모델 학습용 합성 데이터

### 2.1 E5-Mistral (ACL 2024)

- **저자**: Microsoft
- **arXiv**: 2401.00368
- **핵심**: **순전히 GPT-4/3.5 합성 데이터**만으로 MTEB SOTA 달성

**2단계 합성 프로세스**:

```
[1] Task Brainstorming (GPT-4)
    → 500K 이상의 작업 유형 생성
    → asymmetric retrieval (short-long, long-short 등)
    → symmetric (STS, bitext retrieval)

[2] Example Generation
    → 각 작업 유형에 대해 (task instruction, query, positive, hard negative) 생성
    → 93개 언어, ~500K 예시, ~180M 토큰
```

**결과**: Mistral-7B + 합성 데이터만 → MTEB **63.1**. 합성 + 라벨 데이터 → **66.6 (SOTA)**. LoRA + 32×V100에서 **< 1,000 스텝**에 학습.

**의의**: "LLM 합성 데이터가 실제 라벨을 대체 가능하다"를 입증.

### 2.2 InPars (SIGIR 2022)

- **저자**: NeuralMind, Zeta Alpha
- **핵심**: GPT-3에 **few-shot으로 (질의, 문서) 예시**를 주고 → 타겟 문서에서 합성 질의 생성

**GBQ (Guided by Bad Questions) 프롬프트**:

```
Example 1: [Good question]
Example 2: [Bad question - too generic]
Example 3: [Good question]
...
Document: [타겟 문서]
Generated Question:
```

"나쁜 질문 예시"를 포함하여 일반적/모호한 질문 생성을 억제.

**결과**: BM25 + monoT5 리랭커를 InPars 합성 데이터로 학습 → BEIR에서 **BM25 대비 +9 nDCG@10**.

### 2.3 InPars-v2 (2023)

- **개선**:
  - GPT-J 6B (오픈소스)로 OpenAI Curie 대체
  - **monoT5-3B 리랭커로 필터링**: 100K 합성 질의 중 상위 **10K만** 선별
- **결과**: BEIR에서 **v1 대비 +6 nDCG@10 평균 개선**.

### 2.4 Promptagator (ICLR 2023)

- **저자**: Google Research
- **핵심**: **단 8개 예시**로 각 BEIR 작업별 리트리버 학습

**프로세스**:

```
[1] 각 작업마다 <=8개 (질의, 문서) 예시 + 작업 설명 작성
       ↓
[2] FLAN 137B가 각 코퍼스 문서에서 합성 질의 생성
       ↓
[3] Consistency filtering (라운드트립):
    - 초기 리트리버 학습
    - 합성 (q, d) 쌍 중 d가 q의 top-K에 포함되는 것만 유지
       ↓
[4] Dual encoder from scratch 학습 + 리랭커
```

**Consistency Filtering**의 효과:
- 너무 일반적인 질의 제거 (많은 문서와 매칭)
- 환각된 질의 제거 (소스 문서와 무관)

**결과**: BEIR 11개 작업에서 ColBERT-v2 대비 **+1.2 nDCG@10**, 리랭커 포함 시 **+5.0**.

### 2.5 Gecko (2024)

- **저자**: Google DeepMind
- **arXiv**: 2403.20327
- **핵심**: **FRet** (Few-shot Prompted Retrieval dataset) — LLM이 positive/negative **모두** 직접 선정

**혁신**: 기존 방법은 "소스 문서가 positive"라고 가정. Gecko는 LLM이 **진짜 positive를 재라벨링**.

```
[1] 문서에서 LLM이 (task, query) 생성
       ↓
[2] Seed 임베딩 모델로 query에 대한 top-N 후보 검색
       ↓
[3] LLM이 top-N을 재랭킹:
    - 최고점 문서 = positive (소스 문서가 아닐 수 있음!)
    - 하위 점수 문서 = hard negatives
```

**결과**: Gecko-768d가 MTEB **66.31** — OpenAI text-embedding-3-large (7배 큼)를 능가.

**핵심 통찰**: LLM이 재라벨링한 positive가 원본 소스 문서를 positive로 쓰는 것보다 **더 좋다**.

### 2.6 GPL (NAACL 2022)

- **저자**: UKP Lab (TU Darmstadt)
- **핵심**: **Cross-encoder로 pseudo-label** → MarginMSE로 dense retriever 학습

**3단계 파이프라인**:

```
[1] Query Generation
    docT5query로 문서당 3개 합성 질의
       ↓
[2] Negative Mining
    TAS-B로 top-50 검색
       ↓
[3] Pseudo-Labeling
    Cross-encoder(MS MARCO)가 (q, pos)와 (q, neg) 점수화
    → margin = score(pos) - score(neg)를 supervision으로 사용
       ↓
[4] MarginMSE 학습
    Student retriever가 CE의 margin을 예측하도록
```

**결과**: 6개 BEIR 도메인 데이터셋에서 zero-shot TAS-B 대비 **+9.3 nDCG@10**.

**의의**: 도메인 적응의 표준 방법. 지도 학습에 근접한 품질을 비지도로 달성.

### 2.7 AugTriever (Salesforce, 2022)

- **핵심**: 합성 LLM 없이 **문서 자체에서** 가상 질의 생성

**2가지 전략**:

| 전략 | 방법 |
|------|------|
| **QExt** | 문서에서 salient span 추출 (TF-IDF, PLM 기반 키워드) |
| **TQGen** | 기존 요약/제목 모델로 가상 질의 생성 (retrieval supervision 불필요) |

**결과**: Hybrid 전략이 Contriever 수준 성능 — **완전 비지도, LLM API 비용 0**.

---

## 3. 인덱스 확장용

### 3.1 Doc2Query / docTTTTTquery (2019)

- **저자**: Nogueira et al. (NYU, Waterloo)
- **핵심**: 문서에서 가상 질의를 생성하고 문서에 **append하여 BM25 인덱싱**

**프로세스**:

```
문서 D → T5가 예측하는 가상 질의 40-80개
       → D + queries를 BM25 인덱스에 저장

질의 시: 일반 BM25로 검색
```

**결과**: MS MARCO MRR@10:
- BM25: 0.184
- doc2query: 0.218
- **docTTTTTquery (T5 + 40 samples): 0.277** (**+9점**)

**의의**: 어휘 불일치(vocabulary mismatch) 해결. 추가 인프라 불필요.

**한계**: ~33%의 환각된 단어(관련 없음). **Doc2Query-- (2023)**에서 relevance 모델로 필터링.

---

## 4. 종합 비교표

| 논문 | 형식 | 대상 모델 | 필터링 방법 |
|------|------|---------|-----------|
| **Self-Instruct** | (instruction, input, output) | 생성기 | ROUGE-L 유사도 |
| **Evol-Instruct** | (complex instruction, response) | 생성기 | 진화 실패 감지 |
| **RAFT** | (question, oracle+distractors, CoT) | 생성기 | P% 샘플링 |
| **E5-Mistral** | (task, query, pos, hard-neg) | 임베딩 | LLM 직접 생성 |
| **InPars** | (synth query, doc) | 리트리버/리랭커 | GPT logprob |
| **InPars-v2** | 동일 + 필터링 | 리트리버 | **monoT5-3B 리랭커** |
| **Promptagator** | (task-specific query, doc) | 리트리버 + 리랭커 | **Consistency filter** |
| **Gecko (FRet)** | (task, query, LLM-pos, LLM-neg) | 임베딩 | **LLM 재랭킹** |
| **GPL** | (synth q, pos, neg, CE margin) | 리트리버 | Cross-encoder |
| **AugTriever** | (span/summary, source doc) | 리트리버 | 없음 (비지도) |
| **Doc2Query** | (doc → queries, append to index) | BM25 확장 | 없음 |

---

## 5. 공통 기법 및 교훈

### 5.1 필터링은 필수

저품질 합성 데이터는 오히려 성능을 **떨어뜨린다**. 모든 성공적인 방법이 일종의 필터링을 사용:

- **ROUGE 유사도** (Self-Instruct) — 중복 제거
- **로그 확률** (InPars) — 모델 확신도
- **리랭커 점수** (InPars-v2) — 별도 품질 판단기
- **Consistency filter** (Promptagator) — 라운드트립 검증
- **LLM 재랭킹** (Gecko) — LLM이 직접 판단
- **Cross-encoder margin** (GPL) — 상대적 관련성

### 5.2 Hard Negative가 핵심

임베딩 학습에서 **쉬운 음성**(랜덤)은 거의 쓸모없다. 모든 성공 논문이 hard negative 전략:

- BM25 top-K에서 오답 선택 (InPars)
- Dense 리트리버 top-K 활용 (GPL, Gecko)
- LLM이 직접 hard negative 선정 (E5-Mistral, Gecko)
- 방해 문서로 명시적 포함 (RAFT)

### 5.3 Few-shot 예시가 제어 신호

Self-Instruct 8개, Promptagator 8개, InPars few-shot — **소수의 예시가 생성 품질을 크게 좌우**한다. "나쁜 예시"를 함께 주는 것(InPars의 GBQ)도 효과적.

### 5.4 Teacher → Student Distillation

작은 모델이 큰 모델의 품질에 근접하는 핵심 레시피:
- GPL: Cross-encoder → dense retriever
- Gecko: LLM → 1.2B 임베딩 모델
- E5-Mistral: GPT-4 → Mistral-7B

---

## 6. 폐쇄 도메인 실무 추천

### 단계별 전략

```
[1] 문서만 있음
       ↓ (Doc2Query로 가상 질의 생성, 즉시 BM25 개선)
[2] BM25 인덱스 확장
       ↓ (InPars-v2로 합성 (q, pos, neg) 생성)
[3] 리트리버 학습 데이터
       ↓ (E5-Mistral 스타일로 GPT-4 합성 + 리트리버 FT)
[4] 고품질 리트리버
       ↓ (RAFT로 LLM 생성기 FT)
[5] 도메인 RAG 완성
```

### 비용 추정 (문서 1,000개 기준)

| 단계 | 도구 | 예상 비용 |
|------|------|---------|
| Doc2Query | T5 (로컬) | 무료 |
| InPars 합성 | GPT-4 | ~$30-100 |
| E5-Mistral 스타일 | GPT-4 | ~$100-300 |
| RAFT 데이터 생성 | GPT-4 | ~$50-200 |
| 학습 (QLoRA) | RTX 4090 1장 | 전기세 |

**총 합계**: **$200-600** 정도면 중급 도메인 RAG 완성 가능.

---

## 7. 관련 블로그 포스트

- [RAFT 상세 리뷰](raft-review.md) — 방해 문서 혼합 학습
- [RAG 동향 총정리](rag-survey-2026.md) — 전체 RAG 기법
- [RAG 검색 속도 최적화](rag-latency-optimization.md)
