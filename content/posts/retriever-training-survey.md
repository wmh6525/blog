---
title: "[서베이] 검색기(임베딩) 학습 논문 총정리 — DPR부터 NV-Embed까지 15년 진화사"
date: 2026-04-19
tags: ["서베이", "RAG", "검색", "임베딩", "contrastive learning"]
categories: ["ML/AI"]
summary: "RAG의 검색 품질을 결정하는 임베딩 모델 학습 논문 15편을 총정리한다. DPR, SimCSE, Contriever, RocketQA, ANCE, BGE, E5, GTE, ColBERTv2, Margin-MSE, SPLADE, Matryoshka, Jina, NV-Embed까지."
math: true
toc: true
draft: false
---

## 왜 검색기 학습인가?

RAG 정확도의 **절반 이상**은 검색기가 결정한다:
- 관련 문서가 top-k에 없으면 → LLM이 아무리 좋아도 답 못 만듦
- 관련 문서가 있어도 top-1에 노이즈 있으면 → 할루시네이션 유발

이 글은 **dense retrieval**의 핵심 논문 15편을 **연대순 + 기법별**로 정리한다.

---

## 1. 기반 논문: Dense Retrieval의 탄생 (2019-2020)

### 1.1 DPR (Dense Passage Retrieval)

- **논문**: Karpukhin et al., EMNLP 2020 (arXiv: 2004.04906)
- **소속**: Facebook AI Research
- **아키텍처**: Dual-encoder (질의용 BERT + 문서용 BERT, 768-dim CLS)

**손실 함수 (InfoNCE)**:

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(q, p^+))}{\exp(\text{sim}(q, p^+)) + \sum_j \exp(\text{sim}(q, p_j^-))}$$

**Negative 전략**:
- **In-batch negatives**: 배치 내 다른 질의의 positive 문서들을 negative로 사용 → $B^2$ 쌍 생성
- **BM25 hard negative 1개**: BM25 top 결과 중 정답 미포함 문서

**결과**: Natural Questions Top-20 **78.4%** (BM25 59.1%)

**의의**: "**In-batch + BM25 hard negative**" 레시피가 이후 모든 dense retrieval의 표준.

### 1.2 SimCSE (Princeton, EMNLP 2021)

- **저자**: Tianyu Gao, Xingcheng Yao, Danqi Chen
- **핵심**: 같은 문장을 **두 번** 인코더에 통과시키면서 **서로 다른 dropout mask**를 적용 → positive pair

**직관**: Dropout 자체가 "최소한의 데이터 증강"이 된다. 추가 증강 기법 불필요.

**결과**: STS 벤치마크 Spearman **76.3** (비지도), **83.8** (supervised, NLI contradiction을 hard negative로)

**의의**: 데이터 증강의 단순화. 이후 모든 임베딩 모델의 pretraining 단계에 영향.

---

## 2. Hard Negative Mining의 혁명 (2020-2021)

### 2.1 ANCE (Microsoft, ICLR 2021)

- **논문**: arXiv: 2007.00808
- **핵심**: "배치 내 negative는 **너무 쉽다**. 전체 corpus에서 **ANN 기반 hard negative**를 뽑자."

**비동기 파이프라인**:

```
┌─────────────────────────┐         ┌─────────────────────┐
│   Inferencer GPU        │────────▶│    Trainer GPU      │
│  (10K step마다 인덱스    │  hard   │  (계속 학습)        │
│   재구축, hard negative  │  neg    │                     │
│   mining)               │         │                     │
└─────────────────────────┘         └─────────────────────┘
```

**결과**: MS MARCO MRR@10 **33.0** (BM25 24.0)

**의의**: **ANN 기반 hard negative mining**이 표준 기법으로 자리잡음.

### 2.2 RocketQA (Baidu, NAACL 2021) — Hard Negative의 완성

- **논문**: arXiv: 2010.08191
- **3가지 혁신**:

**1. Cross-batch negatives**: 여러 GPU 간 negative 공유 → 배치 크기 대비 효과적 negative 수 증가

**2. Denoised Hard Negatives** (**가장 중요**):
> Hard negative 중에는 **라벨이 없는 진짜 positive**가 섞여 있다 (False negative 문제).

해결: **Cross-encoder**가 hard negative를 재평가하여 "진짜 상관없는 것"만 선별.

```
BM25/ANN top-K → hard neg 후보
     ↓
Cross-encoder 점수화
     ↓
높은 점수 (관련 있음) → 제거 (false negative)
낮은 점수 (진짜 무관) → 학습에 사용
```

**3. Data augmentation**: Cross-encoder가 라벨 없는 질의에 자동 라벨 → 학습 데이터 확장

**결과**: MS MARCO MRR@10 **37.0**

**의의**: "**Cross-encoder로 hard negative를 정제**"하는 패턴을 확립. 이후 E5, BGE, NV-Embed 모두 이 패턴을 계승.

### 2.3 RocketQAv2 (EMNLP 2021)

- **핵심**: Retriever + Reranker **공동 학습**
- **Dynamic Listwise Distillation**: Retriever의 top-k를 Reranker가 실시간 평가 → KL divergence로 retriever 업데이트. Reranker는 retriever의 새 후보로 계속 재학습.

**의의**: 정적 hard negative list가 아닌 **동적** 샘플링. ColBERTv2, SPLADE v3가 계승.

---

## 3. 비지도 사전학습 (2021-2022)

### 3.1 Contriever (Meta, TMLR 2022)

- **논문**: arXiv: 2112.09118
- **핵심**: **완전 비지도 pretraining** — 같은 문서의 두 랜덤 스팬을 positive pair로

**Independent Cropping**:

```
문서: [A B C D E F G H I J]
        ↓ 랜덤 스팬 2개 추출
Positive pair: [C D E F] ↔ [G H I]
```

**MoCo momentum contrast**: 큰 negative queue (65K)를 느리게 업데이트되는 momentum encoder로 유지 → 배치 크기 제약 없이 많은 negative 활용.

**결과**: BEIR R@100에서 BM25 능가 (11/15). 비지도임에도!

**의의**: "라벨 없이도 가능하다"를 입증. 이후 E5, BGE, GTE의 **Stage 1** 사전학습.

---

## 4. 대규모 약지도 학습 시대 (2022-2023)

### 4.1 E5 (Microsoft, 2022)

- **논문**: arXiv: 2212.03533
- **2단계 학습**:

**Stage 1 — Weakly-supervised Contrastive Pretraining**:
- **CCPairs** 270M 쌍 자동 수집:
  - Reddit post-comment 쌍
  - StackExchange Q-A
  - Common Crawl
  - Wikipedia
  - 과학 논문 (title-abstract)
- **Consistency filter**: 1.3B 쌍 → 초기 모델로 점수화 → 일관성 있는 270M만 유지

**Stage 2 — Supervised Fine-tuning**:
- NLI + MS MARCO + NQ
- Hard negative + **cross-encoder distillation**

**비대칭 prefix 트릭**:

```
입력: "query: What is photosynthesis?"  ← 질의
입력: "passage: Photosynthesis is..."   ← 문서
```

같은 인코더를 쓰지만 prefix로 비대칭 signal 주입.

**결과**: E5-base가 BEIR에서 **BM25를 라벨 없이 능가** (최초)

### 4.2 BGE / C-Pack (BAAI, SIGIR 2024)

- **논문**: arXiv: 2309.07597
- **3-Stage Pipeline** (업계 표준이 됨):

```
Stage 1: Plain-text Pretraining (RetroMAE)
    ↓ 100M+ 무라벨 문서
    ↓
Stage 2: General-purpose Contrastive
    ↓ 100M+ 약지도 쌍
    ↓ In-batch negatives만, huge batch
    ↓
Stage 3: Task-specific Fine-tuning
    ↓ ~1M 라벨 쌍
    ↓ ANN hard negatives + in-batch
```

**결과**: bge-large-en-v1.5 MTEB **64.23**

**생태계**: 학습 코드, 데이터, C-MTEB 벤치마크까지 **전부 오픈소스**. 실무 표준.

### 4.3 GTE (Alibaba, 2023)

- **논문**: arXiv: 2308.03281
- **혁신**: **Symmetric negatives** — 배치 내 다른 질의**도** negative로, 다른 문서**도** negative로 → 사실상 negative 2배

**결과**: gte-base (110M 파라미터) MTEB **62.4** — OpenAI text-embedding-ada-002 능가

**의의**: **효율성**이 가장 큰 기여. 작은 모델로 더 좋은 성능.

---

## 5. Late Interaction: ColBERT 계열

### 5.1 ColBERTv2 (Stanford, NAACL 2022)

- **논문**: arXiv: 2112.01488
- **아키텍처**: **Late interaction** — 각 토큰이 임베딩을 가지고 MaxSim으로 매칭

$$\text{score}(q, d) = \sum_{i \in q} \max_{j \in d} \langle E_q[i], E_d[j] \rangle$$

**학습 — Denoised Distillation**:

```
[1] ColBERTv1으로 top-k 검색
      ↓
[2] MiniLM cross-encoder로 재평가
      ↓
[3] 64-way tuple (1 positive + 63 hard negatives)
      ↓
[4] KL-divergence loss로 ColBERTv2 distillation
```

**Residual Compression** (결정적 기여):
- 각 토큰 벡터 = (centroid index + 1-2 bit residual)
- MS MARCO 인덱스: **154GB → 16-25GB (6-10배 축소)**

**결과**: MS MARCO MRR@10 **39.7** (단일 retriever 최고)

**의의**: Late interaction의 품질을 유지하면서 저장 문제 해결. ColPali, Jina v4로 계승.

### 5.2 Margin-MSE Distillation (TU Wien, 2020)

- **논문**: Hofstätter et al., arXiv: 2010.02666
- **핵심**: Cross-encoder teacher의 **절대 점수가 아닌 margin**을 student가 모방

$$\mathcal{L} = \text{MSE}\left(M_s(q, p^+) - M_s(q, p^-),\ M_t(q, p^+) - M_t(q, p^-)\right)$$

**왜 margin인가?**
- Cross-encoder 점수 범위: 예를 들어 [-5, 10]
- Dual-encoder (cosine) 점수 범위: [-1, 1]
- **절대값은 맞출 수 없지만, 상대적 차이는 맞출 수 있다**

**의의**: 이후 모든 teacher-student distillation의 표준 손실. E5, TAS-B, SPLADE v2, Contriever-supervised가 모두 사용.

---

## 6. Sparse Retrieval의 부활: SPLADE

### 6.1 SPLADE v2 (Naver, 2021)

- **논문**: arXiv: 2109.10086
- **아키텍처**: BERT의 **MLM head를 활용** — 각 토큰이 어휘 크기의 sparse 벡터 생성

$$w_j = \max_i \log(1 + \text{ReLU}(w_{ij}))$$

**FLOPS 정규화**: 검색 비용(역인덱스 posting list 길이)이 균형 잡히도록 명시적 regularization.

**효과**:
- Sparse (역인덱스 호환) + neural (학습된 가중치 + 어휘 확장)
- "**learn**"이 "학습", "연구", "조사"도 활성화 (BM25는 불가능)

### 6.2 SPLADE v3 (2024)

- **개선**:
  - Hard negative 100개/질의 (50 top + 50 random)
  - **KL + Margin-MSE 결합** 손실 (λ_KL=1, λ_MSE=0.05)
  - Ensemble cross-encoder teacher

**결과**: 41/44 벤치마크에서 통계적으로 유의한 개선. Cross-encoder 수준에 근접.

---

## 7. 현대 LLM 기반 임베딩 (2024-)

### 7.1 E5-Mistral (Microsoft, 2024)

- **핵심**: Mistral-7B를 임베더로 변환. **순수 합성 데이터**로 MTEB SOTA.
- 상세: [합성 데이터 서베이](rag-synthetic-data-survey.md) 참조

### 7.2 NV-Embed (NVIDIA, ICLR 2025)

- **논문**: arXiv: 2405.17428
- **3가지 혁신**:

**1. Latent Attention Pooling**:
- Mean pooling (희석됨), last-token (최근 편향) 대신
- 학습된 512개 latent 벡터에 cross-attention → MLP → mean pool

**2. Causal Mask 제거**:
- 임베딩은 autoregressive 불필요 → bidirectional attention 허용

**3. 2-Stage Contrastive Instruction Tuning**:
- Stage 1: 검색 데이터, in-batch + hard negatives
- Stage 2: 비검색 (분류, 클러스터링, STS), **in-batch 비활성화** (같은 클래스는 정당히 비슷해야 함)

**결과**: NV-Embed-v2 MTEB **72.31** (2024.08 #1)

**의의**: "**Decoder-only LLM을 임베더로**" 패러다임 확립.

### 7.3 Jina v3 / v4 (2024-2025)

**Jina v3**:
- XLM-RoBERTa + **작업별 LoRA 5개**:
  - `retrieval.query`, `retrieval.passage`
  - `text-matching`
  - `classification`
  - `separation`
- 8192 토큰, 89개 언어

**Jina v4 (2025)**:
- 3.8B 멀티모달 (텍스트+이미지)
- 단일 벡터 + ColBERT 스타일 multi-vector **동시 지원**
- 시각적 문서(표, 차트, 다이어그램)에 특화

**의의**: **작업별 LoRA** 모듈러성. 추론 시 필요한 어댑터만 로드.

---

## 8. Matryoshka Representation Learning (NeurIPS 2022)

- **논문**: Kusupati et al., arXiv: 2205.13147
- **핵심**: 하나의 임베딩이 **여러 차원에서** 독립적으로 작동

**학습 방식**:

$$\mathcal{L} = \sum_{m \in \lbrace 8, 16, 32, 64, \ldots, 2048 \rbrace} \mathcal{L}(f(x)[:m])$$

**추가 비용**: 사실상 0 (같은 forward pass를 다른 truncation으로 평가)

**효과**:
- 14× 작은 임베딩으로 동일 정확도
- Adaptive retrieval: low-d로 후보 축소 → full-d로 재랭킹

**채택**: OpenAI text-embedding-3, Nomic, Jina v3/v4, Snowflake Arctic, gte-v1.5. **2023-2024 표준**.

---

## 9. Hard Negative Mining 진화사 (종합)

| 시기 | 방법 | Negative 출처 | 해결 문제 |
|------|------|-------------|----------|
| 2020 | **DPR** | In-batch + 1 BM25 | 쉬운 randoms로는 학습 안 됨 |
| 2020 | **ANCE** | ANN 전체 corpus, 10K step마다 재구축 | Train-test 분포 격차 |
| 2020 | **RocketQA** | Cross-batch + **CE-denoised** | **False negatives** (라벨 없는 positive) |
| 2021 | **RocketQAv2** | Dynamic listwise distillation | 정적 list는 낡음 |
| 2021 | **ColBERTv2** | CE distillation (Margin-MSE/KL) | 라벨 대신 CE 순위 모방 |
| 2022 | **E5** | 약지도 쌍 + hard neg + CE distillation | 라벨 부족 |
| 2023 | **BGE/GTE** | 3-stage: unsup → weak sup → ANN hard neg | 전체 파이프라인 표준화 |
| 2024 | **SPLADE v3** | 100 hard neg (50 top + 50 sampled) | Recall vs Precision 균형 |
| 2024 | **NV-Embed** | Hard neg + **positive 점수 임계값 필터링** | LLM 스케일에서 false neg |

### 보편 레시피 (2024 표준)

```
1. 약지도 데이터로 in-batch negative로 warm-up
       ↓
2. 현재 모델로 ANN hard negative mining
       ↓
3. Cross-encoder로 **denoise** (false negative 제거)
       ↓
4. Margin-MSE + KL로 student에 distillation
       ↓
5. Matryoshka + 작업별 prompt/LoRA
```

---

## 10. 폐쇄 도메인 실무 추천

### 단계별 접근

```
[Level 0] 베이스라인
  - BGE-m3 오픈소스 모델 그대로 사용
  - BM25와 앙상블 (RRF)

[Level 1] 도메인 Fine-tuning
  - Doc2Query로 합성 질의 생성
  - BGE에 in-batch + BM25 hard negative FT

[Level 2] Hard Negative Mining
  - Stage 1 모델로 ANN top-50 → BGE-reranker로 denoise
  - Margin-MSE로 다시 FT

[Level 3] 최대 정확도
  - ColBERTv2로 교체 (저장 공간 감수)
  - 또는 Matryoshka로 차원 유연성 확보
```

### 모델 선택 가이드

| 상황 | 추천 모델 |
|------|---------|
| 일반 영어 RAG | **BGE-large-en-v1.5** |
| 한국어/다국어 | **BGE-m3** 또는 **Jina v3** |
| 높은 정확도 필요, 저장 공간 여유 | **ColBERTv2** |
| LLM 통합 선호 | **NV-Embed-v2** 또는 **E5-Mistral** |
| 모바일/엣지 | **Matryoshka** 적용 모델 (GTE-base) |

---

## 11. 관련 블로그 포스트

- [RAG 학습용 합성 데이터 서베이](rag-synthetic-data-survey.md)
- [도메인 최적화 LLM for RAG](domain-optimized-llm-for-rag.md)
- [RAG 검색 속도 최적화](rag-latency-optimization.md) — Seismic, CAGRA
