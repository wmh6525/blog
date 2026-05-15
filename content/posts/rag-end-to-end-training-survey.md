---
title: "[서베이] RAG End-to-end 공동 학습 논문 총정리 — ORQA, REALM, RAG, RETRO, Atlas, REPLUG, RA-DIT, Self-RAG"
date: 2026-04-20
tags: ["서베이", "RAG", "LLM", "검색", "공동학습"]
categories: ["ML/AI"]
summary: "검색기와 생성기를 함께 학습하는 end-to-end RAG 논문 12편을 총정리한다. ORQA(2019)의 latent retrieval부터 Self-RAG(2024)의 reflection tokens까지, 검색-생성 결합도의 진화."
math: true
toc: true
draft: false
---

## 왜 End-to-end 공동 학습인가?

RAG 파이프라인을 "분리된 컴포넌트"로 보면 한계가 있다:

```
검색기 (고정) → 문서 → 생성기 (고정) → 답변
      ↑ 생성기가 원하는 것을 모름    ↑ 검색기가 가져온 것을 제대로 못 씀
```

**End-to-end 학습**은 이 간극을 메운다:
- 생성기가 **"이런 문서가 도움된다"** 를 검색기에 신호로 전달
- 검색기가 **"생성기에 유용한 문서"** 를 찾도록 학습
- 혹은 **생성기가 검색 자체를 판단**하도록 학습

이 글은 **2019년 ORQA부터 2024년 RA-DIT/Self-RAG까지** 12편의 핵심 논문을 정리한다.

---

## 1. 결합도 스펙트럼으로 본 12편

**느슨한 결합 ← ──────────────────────── → 완전 통합**

```
kNN-LM (frozen LM+retrieval)
    ↓
FLARE (inference-time active retrieval)
    ↓
REPLUG (LM frozen, retriever만 학습)
    ↓
ORQA (marginal likelihood)
    ↓
RAG-Lewis (query encoder만 학습)
    ↓
Atlas / RA-DIT (두 단계 FT)
    ↓
REALM (pretraining부터 통합)
    ↓
RETRO / InstructRetro (아키텍처 자체가 retrieval-aware)
```

---

## 2. Latent Retrieval 시대 (2019-2020)

### 2.1 ORQA (Google, ACL 2019)

- **논문**: "Latent Retrieval for Weakly Supervised Open Domain QA"
- **저자**: Kenton Lee, Ming-Wei Chang, Kristina Toutanova
- **핵심**: **QA 쌍만으로** 검색기+독해기 공동 학습 — 문서는 latent variable

**Training Objective**:

$$P(y | x) = \sum_z P(y | x, z) \cdot P(z | x)$$

- $x$: 질의
- $z$: retrieved passage (latent)
- $y$: answer span
- 정답 marginal log-likelihood 최대화 → gradient가 retriever까지 역전파

**Cold-start 문제**: Retrieval이 너무 랜덤하면 학습 불가

**해결 — ICT (Inverse Cloze Task)**:

```
문서: [A B C D E F G H I J]
       ↓ 한 문장 제거
학습: C를 query로 → [A B D E F G H I J]를 positive로 검색
```

**결과**: NQ에서 BM25 대비 EM **+19점**. 최초로 "학습된 dense retrieval > BM25" 입증.

### 2.2 REALM (Google, ICML 2020)

- **논문**: arXiv: 2002.08909
- **핵심**: **사전학습(pretraining)** 단계부터 retrieval 통합

**MLM + Retrieval**:

$$P(y | x) = \sum_z P(y | x, z) \cdot P(z | x)$$

- $x$: 마스킹된 문장
- $z$: retrieved Wikipedia doc
- $y$: 마스킹된 토큰

**"유용한 문서를 retrieve할 때 MLM loss가 줄어든다"** → retriever 업데이트.

**핵심 트릭 — Asynchronous MIPS Index Refresh**:

```
학습은 계속 진행
        ↓
500 step마다 별도 GPU에서:
  - 현재 document encoder로 전체 corpus 재인코딩
  - FAISS 인덱스 재구축
  - 학습 GPU에 교체
```

**Salient Span Masking**: 랜덤 마스킹 대신 **named entities, 날짜**만 마스킹 → world knowledge 학습에 집중.

**결과**: NQ-Open에서 T5-11B 능가 (REALM은 **30배 작음**).

**의의**: Retrieval-aware pretraining의 시작점.

### 2.3 RAG (Lewis et al., NeurIPS 2020)

- **저자**: Patrick Lewis et al., Facebook AI Research
- **핵심**: DPR + BART의 공동 fine-tuning. **"RAG"라는 용어를 만든 논문.**

**두 변형**:

**RAG-Sequence**: 하나의 문서로 전체 답변 생성

$$P(y | x) = \sum_z P(z | x) \cdot \prod_i P(y_i | x, z, y_{\lt i})$$

**RAG-Token**: 토큰마다 다른 문서 marginalize

$$P(y_i | x, y_{\lt i}) = \sum_z P(z | x) \cdot P(y_i | x, z, y_{\lt i})$$

**Retriever 상태**:
- **Query encoder: 학습** (seq2seq loss가 역전파)
- **Document encoder: frozen** (재인덱싱 비용 회피)

**결과**: NQ **44.5 EM**, TriviaQA **56.1 EM** — Open-domain QA SOTA.

**의의**: Generation task에 retrieval 도입. 지금의 RAG 패러다임 정립.

---

## 3. Retrieval-Aware 아키텍처 (2020-2023)

### 3.1 kNN-LM (ICLR 2020)

- **논문**: Khandelwal et al., "Generalization through Memorization"
- **핵심**: **학습 없이** inference time에 retrieval로 LM 보강

$$P_{\text{final}}(y | x) = \lambda \cdot P_{\text{kNN}}(y | x) + (1 - \lambda) \cdot P_{\text{LM}}(y | x)$$

**Datastore 구축**:
```
모든 학습 context의 (hidden_state, next_token) 쌍 저장
→ FAISS 인덱스로 변환
```

**추론 시**:
```
현재 context의 hidden_state
       ↓ FAISS로 k-NN 검색
       ↓ 이웃들의 next_token 분포 계산
       ↓ LM 분포와 interpolation
```

**결과**: Wikitext-103 perplexity **18.65 → 15.79** (추가 학습 0).

**의의**: **"기억이 학습보다 쉽다"**. Datastore 교체만으로 domain adaptation 가능.

**한계**: 매 토큰마다 kNN 검색 → 느림. 메모리 수백 GB.

### 3.2 RETRO (DeepMind, ICML 2022)

- **논문**: arXiv: 2112.04426
- **핵심**: **아키텍처에 retrieval 내장** — Chunked Cross-Attention (CCA)

**구조**:

```
입력 텍스트를 chunk (~64 tokens) 단위로 분할
       ↓
각 chunk마다 frozen BERT로 nearest neighbor 검색
       ↓
Neighbors를 Transformer encoder로 처리
       ↓
Decoder가 CCA로 neighbor 참조하며 생성
```

**BERT retriever는 완전 frozen**: 2T 토큰 DB 재인덱싱 비용 회피.

**결과**: 7.5B 파라미터 RETRO가 **GPT-3 (175B)**, **Jurassic-1 (178B)**과 comparable — **25배 작음**.

**의의**: "검색 DB 크기를 늘리면 같은 파라미터로 더 나아진다"를 입증.

### 3.3 InstructRetro (NVIDIA, ICLR 2024)

- **논문**: arXiv: 2310.07713
- **핵심**: RETRO를 **48B 규모**로 스케일링 (최대 retrieval-pretrained LLM)

**2단계**:
1. 43B GPT를 **100B 토큰 추가** retrieval-augmented pretraining (1.2T 토큰 DB)
2. Instruction tuning

**놀라운 발견 — Encoder Ablation**:
> Instruction tuning 후, **RETRO encoder를 제거**하고 decoder만 써도 comparable 성능!

→ RETRO decoder가 retrieval 활용 방식을 **내재화**했다는 증거.

**결과**: 8개 short QA에서 GPT 대비 **+7%**, 4개 long QA에서 **+10%**, 3개 요약에서 **+16%**.

**추가 학습 비용**: **+2.58%** GPU 시간만.

---

## 4. Few-shot 시대 (2022-2023)

### 4.1 Atlas (Meta, JMLR 2023)

- **논문**: arXiv: 2208.03299
- **아키텍처**: Contriever + T5-based **Fusion-in-Decoder (FiD)**

**FiD**:
```
각 retrieved passage를 개별 encoder로 인코딩
       ↓
Decoder에서 모두 concat하여 cross-attention
```

**4가지 Retriever 학습 objective 비교**:

| Objective | 설명 |
|-----------|------|
| **ADist** | Decoder attention → retriever 분포 KL |
| **EMDR²** | EM 알고리즘, retrieved docs를 latent로 |
| **PDist** (채택) | 각 문서가 LM perplexity를 얼마나 개선하는지 distill |
| **LOOP** | Leave-one-out PDist |

**PDist가 최종 선택**:

$$P_{\text{LM}}(z | x, y) \propto \text{perplexity}(y | x, z)^{-1}$$

Retriever가 "LM이 선호하는 문서"를 학습.

**결과**: NQ 64-shot에서 **42%+**, PaLM 540B을 3% 능가 (**파라미터 50배 작음** — Atlas 11B).

### 4.2 REPLUG (Meta, NAACL 2024)

- **논문**: arXiv: 2301.12652
- **핵심**: **LM은 완전 frozen (black-box)** — GPT-3/Codex도 가능

**REPLUG LSR (LM-Supervised Retrieval)**:

$$\mathcal{L} = \text{KL}(P_{\text{LM}}(z | x, y) \| P_R(z | x))$$

- $P_R$: 리트리버의 검색 확률
- $P_{\text{LM}}$: 각 문서가 얼마나 perplexity 개선하는지

"**LM이 좋아하는 문서**"를 찾도록 리트리버만 학습.

**Ensemble 추론**:
```
top-K 문서 각각 prepend → LM을 K번 호출
       ↓
retrieval weight로 output 분포 가중평균
```

**결과**: GPT-3 175B에서 perplexity **6.3%** 개선, Codex MMLU **+5.1%**.

**의의**: **Black-box LLM API에도 retrieval 결합 가능**. LLM을 fine-tune 할 필요 없음.

**한계**: LM을 K번 호출 → 추론 비용 K배.

---

## 5. Instruction Tuning 시대 (2023-)

### 5.1 RA-DIT (Meta, ICLR 2024)

- **논문**: arXiv: 2310.01352
- **핵심**: 기존 LLM에 retrieval 능력을 **retrofit**

**Dual Fine-tuning**:

**Stage 1 — LM-ft**: LLaMA를 retrieval-augmented instruction data로 SFT
```
입력: instruction + [retrieved context] → output
```
→ "검색 결과를 활용하는 법"을 학습

**Stage 2 — R-ft**: DRAGON+ retriever를 LM perplexity distillation으로 학습 (REPLUG LSR 방식)

```
Retriever가 LM이 선호하는 문서를 찾도록
```

**결과**: LLaMA 65B + DRAGON+에서 zero-shot **+8.9%**, 5-shot **+1.4%** 평균 개선.

**의의**: **검색 pretraining 없이** 기존 LLM을 retrieval-ready로 만드는 실용적 레시피. 두 단계 순차 학습.

### 5.2 Self-RAG (UW + IBM, ICLR 2024)

- **논문**: arXiv: 2310.11511
- **핵심**: LM이 **reflection tokens**를 생성해 검색/품질을 스스로 판단

**4가지 Reflection Token**:

| 토큰 | 값 | 의미 |
|------|-----|------|
| **Retrieve** | yes/no/continue | 지금 검색 필요? |
| **ISREL** | Relevant/Irrelevant | 검색된 passage 관련? |
| **ISSUP** | Fully/Partially/No support | 생성된 segment가 passage에 뒷받침됨? |
| **ISUSE** | 1-5 | 최종 답 유용성 |

**학습 파이프라인**:

```
[1] Critic 모델 학습
    GPT-4가 reflection token 라벨 생성
    → Critic C (7B)를 supervised FT
          ↓
[2] Augmented Dataset
    Critic이 학습 corpus의 각 예시에 reflection token 삽입
          ↓
[3] Generator 학습
    표준 next-token prediction으로 reflection token 포함한 sequence 학습
    (추론 시 critic 불필요!)
```

**추론 — Adaptive Decoding**:
```
LM이 "Retrieve=yes" 생성
       ↓
Retriever 호출 → K개 passages
       ↓
각 passage에 대해 병렬로 후보 생성
       ↓
ISREL/ISSUP/ISUSE 확률로 tree-decoding soft scoring
       ↓
최선 선택
```

**결과**: Self-RAG 7B/13B가 **ChatGPT, Llama2-chat, Alpaca 능가**. ASQA citation precision **+29.56%p**.

**의의**: **검색 시점을 LM이 결정**. 인용 가능성(citation) 개선.

---

## 6. Active Retrieval: 동적 검색 시점 결정

### 6.1 FLARE (CMU, EMNLP 2023)

- **논문**: arXiv: 2305.06983
- **핵심**: **학습 없이**, 생성 중 low-confidence 감지 시 재검색

**알고리즘**:

```
다음 문장 임시 생성
       ↓
문장 내 token logprob 확인
       ↓
min(logprob) < threshold?
       ├─ Yes → 임시 문장을 query로 재검색 → 문장 재생성
       └─ No → 그대로 사용, 다음 문장으로
```

**Forward-looking Retrieval**: 미래 생성물을 예측해 query로 사용하는 아이디어.

**결과**: 2WikiMultihopQA, StrategyQA 등 long-form에서 single retrieval 대비 개선.

**의의**: **학습 없이 black-box LM에 적응적 검색** 추가. GPT-4 API에 바로 적용 가능.

### 6.2 Toolformer (Meta, NeurIPS 2023)

- **논문**: arXiv: 2302.04761
- **핵심**: LM이 **스스로** API 호출 삽입을 학습 (self-supervised)

**4단계 파이프라인**:

```
[1] Sample
    GPT-J가 여러 위치에 후보 API call 삽입
          ↓
[2] Execute
    실제 API 호출 (search, QA, calculator 등)
          ↓
[3] Filter
    API 결과 포함 시 미래 토큰 LM loss가
    실질적으로 감소하는 경우만 유지
          ↓
[4] Fine-tune
    필터링된 augmented dataset으로 표준 CE loss FT
```

**결과**: GPT-J 6.7B Toolformer가 **GPT-3 175B보다 zero-shot 우수**. LAMA factual completion **+11.7~18.6%p**.

**의의**: 인간 어노테이션 없이 tool use 학습. Self-RAG의 정신적 조상.

---

## 7. 종합 비교표

| 논문 | 연도 | Retriever 상태 | 학습 신호 | 핵심 |
|------|------|-------------|---------|------|
| ORQA | 2019 | 학습 | Marginal QA NLL + ICT | Latent retrieval의 시작 |
| kNN-LM | 2020 | Frozen | 없음 | Inference-time interpolation |
| REALM | 2020 | 학습 (비동기) | MLM | Retrieval-aware pretraining |
| RAG-Lewis | 2020 | Query enc만 | Seq2seq NLL | DPR+BART joint FT |
| RETRO | 2021 | Frozen BERT | LM loss | 2T 토큰 DB + CCA |
| Atlas | 2022 | 학습 | PDist (perplexity distill) | Few-shot RAG + FiD |
| FLARE | 2023 | Frozen | 없음 | Forward-looking retrieval |
| REPLUG | 2023 | 학습 (LSR) | KL(P_LM \| P_R) | Black-box LM distillation |
| Toolformer | 2023 | Frozen tools | Self-supervised filtering | Tool use 자기학습 |
| InstructRetro | 2023 | Frozen | LM + IT | 48B 스케일링 |
| Self-RAG | 2023 | Frozen | GPT-4→Critic→Generator | Reflection tokens |
| RA-DIT | 2023 | 학습 | Dual FT (LM + LSR) | LLM retrofit |

---

## 8. 진화의 흐름

### 8.1 Supervision Signal의 진화

```
QA Label (ORQA)
   → MLM (REALM)
      → Seq2seq marginal (RAG)
         → LM Perplexity Distillation (Atlas PDist, REPLUG LSR, RA-DIT)
            → GPT-4 → Critic Distillation (Self-RAG)
```

**핵심 원칙**: "**LM이 원하는 문서가 좋은 문서**" — perplexity가 supervision signal로 수렴.

### 8.2 Document Encoder Freezing

대부분 **document encoder는 frozen** 또는 비동기 업데이트:
- RAG-Lewis: Query encoder만 학습
- REALM: 500 step마다 비동기 재인덱싱
- Atlas: 주기적 재인덱싱
- RETRO, REPLUG: 완전 frozen

**이유**: 대규모 corpus 재인덱싱 비용 (수십~수백 GB FAISS 재구축).

### 8.3 결합도 vs 유연성 트레이드오프

| 전략 | 장점 | 단점 |
|------|------|------|
| **Frozen LM** (REPLUG) | Black-box API 호환 | LM이 retrieval-aware 아님 |
| **Frozen Retriever** (RETRO, Self-RAG) | 검색 인프라 재사용 | Retrieval 품질 제한 |
| **둘 다 학습** (REALM) | 최대 통합 | 인프라 복잡, 재인덱싱 비용 |
| **순차 FT** (RA-DIT) | 실용적 타협 | Joint optimization 아님 |

---

## 9. 폐쇄 도메인 실무 추천

### 시나리오별 권장

**A. 문서+질의 쌍 있음, LLM 튜닝 가능**
```
1. BGE에 contrastive FT (Hard negative)
2. RAFT 또는 RA-DIT 스타일로 LLM FT
3. 최종적으로 Self-RAG 스타일로 reflection 주입
```

**B. LLM이 Black-box (GPT-4 API 사용)**
```
1. Retriever만 REPLUG LSR 스타일로 학습
2. FLARE로 active retrieval 추가 (학습 없이 구현 가능)
```

**C. 오픈엔드 과제, 정답 불명확**
```
1. CCS 스타일로 정답 없이 RL (이전 포스트 참조)
2. 또는 Self-RAG의 ISSUP/ISUSE 점수로 품질 추정
```

---

## 10. 관련 블로그 포스트

- [RAG 학습용 합성 데이터 서베이](rag-synthetic-data-survey.md)
- [검색기 학습 논문 총정리](retriever-training-survey.md)
- [RAFT 상세 리뷰](raft-review.md)
- [CRAG 상세 리뷰](crag-review.md)
- [Search-R1 상세 리뷰](search-r1-review.md)
- [CCS 상세 리뷰](cycle-consistent-search-review.md)
