---
title: "논문을 읽기 전에: 선수 개념과 논문 지도"
date: 2026-10-13
summary: "논문 리뷰를 이해하기 위한 원문 읽기 순서, 익힐 개념, 먼저 볼 부분을 정리했다."
toc: true
draft: false
---

어떤 논문을 읽어야 할지는 현재 이해하려는 방법에 따라 다르다. 각 글의 첫머리에 선수 개념과 원문 링크를 붙였다. 여기서는 여러 글에서 다시 등장하는 논문을 찾아볼 수 있다.

[전체 학습 순서]({{< relref "learn" >}}) · [AI 기초 설명]({{< relref "ai-basics" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

## 선수 논문은 이렇게 읽기

1. 먼저 해당 글의 쉬운 도입과 용어 설명을 읽는다. 확률·행렬·손실처럼 기초 개념이 낯설면 원문부터 읽기보다 그 뜻을 먼저 익힌다.
2. 선수 논문에서는 초록과 문제 설정을 읽고, 구조 그림에서 입력·저장 정보·출력에 표시한다.
3. 각 글의 표에 적힌 개념과 읽을 부분을 확인한다. 첫 독서에서는 필요한 부분을 목표로 읽고, 유도나 구현을 다룰 때 범위를 넓힌다.
4. 현재 논문으로 돌아와 이전 방법에서 무엇을 유지하고 무엇을 바꿨는지 자기 말로 설명한다.

표의 순서는 편집한 학습 순서다. 출판 연대순이나 모든 직접 인용 관계를 뜻하지 않는다. 각 글에서 ‘개념 준비’, ‘사용한 구성 요소’, ‘비교하는 방법’, ‘구현 대상’을 구분했다. 원문 링크는 arXiv·저널·원문 저장소의 제목과 자료를 확인해 연결했다.

## 대표적인 읽기 연결

| 이해하려는 글 | 먼저 읽을 흐름 | 먼저 이해할 개념 |
|---|---|---|
| Mamba | HiPPO → S4 → Mamba | 상태, 긴 과거의 압축, 고정 전이와 입력 의존 갱신 |
| Mamba-2 | Transformer의 Attention + Mamba → Mamba-2 | 재귀 계산과 구조화 행렬 계산의 대응 |
| H-Net | 바이트·패치 개념 → BLT + Mamba-2 → H-Net | 미리 정한 경계와 학습되는 경계 |
| DeBERTa | Transformer → BERT → ELECTRA의 판별 학습 | 문맥, 위치 표현, 마스크 예측과 대체 토큰 판별 |
| DPO | 강화학습 개념 → PPO → InstructGPT → DPO | 정책, 선호 쌍, 보상 모델, 참조 정책 |
| SDFT | 증류 → InstructGPT의 SFT/RLHF 흐름 → SDFT | 교사와 학생, 입력 문맥 차이, 획득과 망각 |
| LLaDA | BERT의 마스킹 → D3PM → MDLM → LLaDA | 학습 때 가리는 과정과 생성 때 복원하는 과정 |
| SONAR | 문장 임베딩 → LASER + NLLB → SONAR | 다국어 공유 표현과 문장 벡터의 병목 |

## 글별 선수 논문 찾기

| 현재 글 | 선수 논문·개념 준비의 읽기 순서 |
|---|---|
| [\[Book\] Build a Large Language Model (Ch.2) — 텍스트 데이터 다루기]({{< relref "posts/build-llm-ch2-working-with-text-data" >}}) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) → [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) |
| [\[논문 리뷰\] DeBERTa v1/v2/v3 — 분리된 어텐션과 향상된 마스크 디코더로 BERT를 넘어서다]({{< relref "posts/deberta-review" >}}) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) → [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) → [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) |
| [\[서베이\] 최신 RAG 동향 총정리 (2026) — 모델, 워크플로우, 프로덕션 패턴]({{< relref "posts/rag-survey-2026" >}}) | [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) → [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) |
| [\[서베이\] RAG의 쿼리 성능을 높이는 모든 방법 — Rewriting, Expansion, Decomposition, Routing부터 RL까지]({{< relref "posts/rag-query-optimization-survey" >}}) | [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) → [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) → [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496) |
| [\[서베이\] 검색기(임베딩) 학습 논문 총정리 — DPR부터 NV-Embed까지 15년 진화사]({{< relref "posts/retriever-training-survey" >}}) | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) → [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) → [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832) |
| [\[서베이\] RAG 학습용 합성 데이터 생성 논문 총정리 — E5-Mistral, Self-Instruct, InPars, Promptagator, Gecko, GPL, RAFT]({{< relref "posts/rag-synthetic-data-survey" >}}) | [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) → [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) → [RAFT: Adapting Language Model to Domain Specific RAG](https://arxiv.org/abs/2403.10131) |
| [\[논문 리뷰\] CRAG — 검색 결과가 잘못되었을 때 스스로 교정하는 RAG]({{< relref "posts/crag-review" >}}) | [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) → [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) |
| [\[논문 리뷰\] LightRAG — 그래프 기반 RAG를 가볍고 빠르게, GraphRAG의 대안]({{< relref "posts/lightrag-review" >}}) | [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) → [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130) |
| [\[논문 리뷰\] RAFT — '오픈북 시험 준비'처럼 LLM을 도메인 RAG에 최적화하기]({{< relref "posts/raft-review" >}}) | [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) → [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) |
| [\[연구분석\] 도메인 특화 RAG용 LLM 구축 완전 가이드 — 경량 기법부터 풀 파이프라인까지]({{< relref "posts/domain-optimized-llm-for-rag" >}}) | [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) → [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) → [RAFT: Adapting Language Model to Domain Specific RAG](https://arxiv.org/abs/2403.10131) |
| [\[서베이\] RAG End-to-end 공동 학습 논문 총정리 — ORQA, REALM, RAG, RETRO, Atlas, REPLUG, RA-DIT, Self-RAG]({{< relref "posts/rag-end-to-end-training-survey" >}}) | [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) → [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) |
| [\[논문 리뷰\] Search-R1 — 강화학습으로 LLM에게 '언제, 무엇을 검색할지' 스스로 배우게 하기]({{< relref "posts/search-r1-review" >}}) | [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) → [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) → [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) |
| [\[논문 리뷰\] Cycle-Consistent Search — 정답 라벨 없이 검색 에이전트를 학습시키는 법]({{< relref "posts/cycle-consistent-search-review" >}}) | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) → [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) → [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516) |
| [\[논문 리뷰\] OPQE — HyDE에 강화학습을 결합한 On-Policy Pseudo-document Query Expansion]({{< relref "posts/opqe-review" >}}) | [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) → [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496) → [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) |
| [\[서베이\] RAG 검색 속도를 획기적으로 줄이는 방법들 — KV 캐시, 서브밀리초 인덱스, 예측적 프리페칭]({{< relref "posts/rag-latency-optimization" >}}) | [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) → [Attention Is All You Need](https://arxiv.org/abs/1706.03762) → [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832) |
| [\[논문 리뷰\] TurboRAG — KV 캐시 사전 계산으로 RAG 추론 9.4배 가속]({{< relref "posts/turborag-review" >}}) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) → [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) |
| [\[논문 리뷰\] CacheBlend (EuroSys 2025 Best Paper) — 비접두사 KV 캐시 재사용으로 RAG 5배 가속]({{< relref "posts/cacheblend-review" >}}) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) → [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) |
| [\[논문 리뷰\] CAGRA — GPU에서 ANN 검색을 10-77배 가속하는 NVIDIA의 그래프 인덱스]({{< relref "posts/cagra-review" >}}) | [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320) → [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) |
| [\[논문 리뷰\] Seismic (SIGIR 2024 준최우수) — 학습된 희소 검색을 서브밀리초로 가속]({{< relref "posts/seismic-review" >}}) | [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) → [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720) |
| [\[입문\] 강화학습 완전 정복 — 게임 배우는 아이부터 PPO·DPO까지 그림으로 이해하기]({{< relref "posts/reinforcement-learning-beginner-guide" >}}) | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) → [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) |
| [\[논문 리뷰\] DPO — RLHF에서 강화학습을 제거하다, '당신의 LM은 몰래 보상모델이다']({{< relref "posts/dpo-review" >}}) | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) → [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) |
| [\[리뷰\] SDFT — 새로 배워도 잊지 않는 파인튜닝: Self-Distillation Enables Continual Learning]({{< relref "posts/sdft-self-distillation-review" >}}) | [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) → [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) |
| [\[서베이\] 저망각 파인튜닝 연구 동향 (2026) — 온폴리시·희소 갱신·학습률·측정]({{< relref "posts/low-forgetting-finetuning-survey-2026" >}}) | [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) → [Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897) → [Continual Learning via Sparse Memory Finetuning](https://arxiv.org/abs/2510.15103) |
| [\[리뷰\] 메모리 레이어로 잊지 않고 배우기 — Memory Layers at Scale부터 Sparse Memory Finetuning, Fast-weight PKM까지]({{< relref "posts/memory-layers-sparse-finetuning-review" >}}) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) → [Large Memory Layers with Product Keys](https://arxiv.org/abs/1907.05242) |
| [\[서베이\] 도메인 학습을 더 잘 되게 하는 방법론 (2026) — 증강·리플레이·학습률·병합·자기증류·평가]({{< relref "posts/domain-adaptation-methods-survey-2026" >}}) | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) → [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) → [Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897) |
| [\[주간 논문\] 2026-10-01 — '온폴리시라서 덜 잊는다'가 흔들린다]({{< relref "posts/weekly-papers-2026-10-01" >}}) | [Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897) → [Memory Layers at Scale](https://arxiv.org/abs/2412.09764) → [Continual Learning via Sparse Memory Finetuning](https://arxiv.org/abs/2510.15103) |
| [\[주간 논문\] 2026-10-05 — 잘 붙인 SFT가 온폴리시 자기증류를 따라잡았다]({{< relref "posts/weekly-papers-2026-10-05" >}}) | [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) → [Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897) → [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) |
| [\[주간 논문\] 2026-10-12 — 온폴리시 증류에도 '신선한 롤아웃'은 필요 없었다]({{< relref "posts/weekly-papers-2026-10-12" >}}) | [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) → [Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897) → [Continual Learning via Sparse Memory Finetuning](https://arxiv.org/abs/2510.15103) |
| [\[개념 정리\] Bayesian Surprise — 놀라움의 수학적 정의와 뇌의 주의 메커니즘]({{< relref "posts/bayesian-surprise" >}}) | [A Mathematical Theory of Communication](https://onlinelibrary.wiley.com/doi/10.1002/j.1538-7305.1948.tb00917.x) |
| [\[개념 정리\] State Space Model(SSM) — 수식으로 이해하는 Mamba의 수학적 기반]({{< relref "posts/ssm-explained" >}}) | [HiPPO: Recurrent Memory with Optimal Polynomial Projections](https://arxiv.org/abs/2008.07669) → [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396) |
| [\[논문 리뷰\] HiPPO: 장거리 의존성 문제의 수학적 해결 — SSM과 Mamba의 이론적 기반]({{< relref "posts/hippo-review" >}}) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| [\[논문 리뷰\] Mamba: 선택적 상태 공간 모델로 Transformer에 도전하다]({{< relref "posts/mamba-ssm-review" >}}) | [HiPPO: Recurrent Memory with Optimal Polynomial Projections](https://arxiv.org/abs/2008.07669) → [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396) |
| [\[논문 리뷰\] Mamba-2: SSM과 Attention의 이중성을 밝히다]({{< relref "posts/mamba2-review" >}}) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) → [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) |
| [\[논문 리뷰\] Mamba-3: SSM 원리로 추론 효율성의 Pareto 프론티어를 진전시키다]({{< relref "posts/mamba3-review" >}}) | [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) → [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) |
| [\[서베이\] Mamba/SSM 계열 모델 완전 정리 — HiPPO에서 Nemotron-H까지]({{< relref "posts/mamba-family-complete-survey" >}}) | [HiPPO: Recurrent Memory with Optimal Polynomial Projections](https://arxiv.org/abs/2008.07669) → [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396) → [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) → [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) |
| [\[코드 분석\] Mamba — state-spaces/mamba 소스코드 완전 분석]({{< relref "posts/mamba-code-analysis" >}}) | [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) → [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) |
| [\[연구 분석\] Selective SSM만으로는 왜 거대 언어모델이 될 수 없는가?]({{< relref "posts/why-pure-ssm-cannot-be-llm" >}}) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) → [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396) → [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) |
| [\[논문 리뷰\] Hymba: SSM과 Attention을 헤드 수준에서 융합한 하이브리드 아키텍처]({{< relref "posts/hymba-review" >}}) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) → [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) → [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) |
| [\[서베이\] 해마를 어떻게 구현할 것인가 — attention 옆에 기억을 따로 두는 방법들]({{< relref "posts/hippocampal-memory-attention-survey" >}}) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) → [Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217) → [Large Memory Layers with Product Keys](https://arxiv.org/abs/1907.05242) |
| [SancMamba: 바이트 수준 계층적 시퀀스 압축 언어 모델 설계기 (1)]({{< relref "posts/sanc-mamba-week1" >}}) | [The Forward-Forward Algorithm: Some Preliminary Investigations](https://arxiv.org/abs/2212.13345) → [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) → [Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2412.09871) |
| [Hymba-SANC: 유동 청킹 기반 개념 토큰 언어 모델 구현 가이드]({{< relref "posts/hymba-sanc-implementation-guide" >}}) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) → [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) → [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](https://arxiv.org/abs/2507.07955) |
| [\[서베이\] 오토인코더에서 VQ-VAE까지 — 이산 표현 학습의 계보]({{< relref "posts/vqvae-survey" >}}) | [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) |
| [\[논문 리뷰\] Hourglass Transformer — 계층적 구조로 시퀀스 모델의 효율성을 높이다]({{< relref "posts/hourglass-transformer-review" >}}) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| [\[서베이\] BLT와 바이트 수준 언어 모델의 계보 — 토크나이저 없는 LLM을 향해]({{< relref "posts/blt-byte-level-lm-survey" >}}) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) → [Hierarchical Transformers Are More Efficient Language Models](https://arxiv.org/abs/2110.13711) → [MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers](https://arxiv.org/abs/2305.07185) |
| [\[논문 리뷰\] H-Net: 완전 E2E 동적 청킹으로 토크나이저를 제거하다]({{< relref "posts/hnet-review" >}}) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) → [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) → [Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2412.09871) |
| [\[논문 리뷰\] MBLM — 5M 바이트 컨텍스트를 단일 GPU로 처리하는 계층적 바이트 LLM]({{< relref "posts/mblm-review" >}}) | [MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers](https://arxiv.org/abs/2305.07185) → [Hierarchical Transformers Are More Efficient Language Models](https://arxiv.org/abs/2110.13711) → [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) |
| [\[논문 리뷰\] BLT-D — Byte Latent Transformer에 블록 디퓨전을 결합해 추론을 92% 가속]({{< relref "posts/blt-d-review" >}}) | [Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2412.09871) → [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006) → [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524) |
| [\[논문 리뷰\] SONAR — 200개 언어 + 음성을 하나의 1024차원 공간에 통합한 다국어 임베딩]({{< relref "posts/sonar-review" >}}) | [Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond](https://arxiv.org/abs/1812.10464) → [No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) |
| [\[서베이\] 연속 의미 공간에서 예측하는 언어 모델 — Discrete Token을 넘어서 (LCM, Coconut, CALM, Diffusion LM, JEPA)]({{< relref "posts/continuous-vector-language-models" >}}) | [SONAR: Sentence-Level Multimodal and Language-Agnostic Representations](https://arxiv.org/abs/2308.11466) → [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) → [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) |
| [\[심화 리뷰\] CALM + Tencent WeChat AI 시리즈 — Scoring Rules · Patch Training · EAR · CALM 모든 수식 완전 해부]({{< relref "posts/calm-tencent-series-review" >}}) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) → [Language Generation with Strictly Proper Scoring Rules](https://arxiv.org/abs/2405.18906) → [Beyond Next Token Prediction: Patch-Level Training for Large Language Models](https://arxiv.org/abs/2407.12665) |
| [\[논문 리뷰\] Hourglass Diffusion Transformers — 메가픽셀에서 DiT의 99% 연산을 절약하다]({{< relref "posts/hourglass-dit-review" >}}) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) → [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) → [Hierarchical Transformers Are More Efficient Language Models](https://arxiv.org/abs/2110.13711) |
| [\[서베이\] Diffusion Language Model 완전 정복 — Diffusion-LM부터 LLaDA, Mercury, GIDD까지]({{< relref "posts/diffusion-language-models-survey" >}}) | [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) → [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006) → [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) |
| [\[논문 리뷰\] SEDD 심화 — Score Entropy 손실의 수식 유도와 학습/추론 완전 해부]({{< relref "posts/sedd-deep-review" >}}) | [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) → [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006) |
| [\[논문 리뷰\] MDLM 심화 — '마스킹 디퓨전 = 가중 MLM' 정리의 완전 유도와 학습/추론 해부]({{< relref "posts/mdlm-deep-review" >}}) | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) → [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006) → [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834) |
| [\[논문 리뷰\] LLaDA — 8B 마스킹 디퓨전 LLM, 데이터·하이퍼파라미터·학습/추론 완전 해부]({{< relref "posts/llada-review" >}}) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) → [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) → [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524) |
| [\[논문 리뷰\] Mercury 심화 — 최초의 상용 Diffusion LLM, 학습/추론과 벤치마크 완전 해부]({{< relref "posts/mercury-deep-review" >}}) | [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006) → [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834) → [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524) |
| [역전파 없이 학습할 수 있을까? — Forward-Forward부터 EBT까지]({{< relref "posts/no-bp-methods-survey" >}}) | [The Forward-Forward Algorithm: Some Preliminary Investigations](https://arxiv.org/abs/2212.13345) → [Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation](https://arxiv.org/abs/1602.05179) |
| [\[논문 리뷰\] Forward-Forward Algorithm: 역전파 없이 두 번의 Forward Pass로 학습하기]({{< relref "posts/forward-forward-review" >}}) | [Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation](https://arxiv.org/abs/1602.05179) |
| [\[논문 리뷰\] Prospective Configuration: 역전파를 넘어서는 학습 — 먼저 상상하고, 그다음 굳힌다]({{< relref "posts/prospective-configuration-review" >}}) | [Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation](https://arxiv.org/abs/1602.05179) |
| [\[논문 리뷰\] Meta-PCN: Meta Prediction Error를 활용한 안정적이고 확장 가능한 Deep Predictive Coding Networks]({{< relref "posts/meta-pcn-review" >}}) | [Inferring neural activity before plasticity as a foundation for learning beyond backpropagation](https://doi.org/10.1038/s41593-023-01514-1) → [Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation](https://arxiv.org/abs/1602.05179) |
| [\[서베이\] 역전파 없이 언어 모델을 학습하는 연구 동향과 사례]({{< relref "posts/no-bp-lm-survey" >}}) | [The Forward-Forward Algorithm: Some Preliminary Investigations](https://arxiv.org/abs/2212.13345) → [Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation](https://arxiv.org/abs/1602.05179) → [Inferring neural activity before plasticity as a foundation for learning beyond backpropagation](https://doi.org/10.1038/s41593-023-01514-1) |
| [\[코드 분석\] Prospective Configuration — PCLayer와 PCTrainer로 구현하는 비역전파 학습]({{< relref "posts/prospective-config-code-analysis" >}}) | [Inferring neural activity before plasticity as a foundation for learning beyond backpropagation](https://doi.org/10.1038/s41593-023-01514-1) |
| [\[코드 분석\] DSF & EqProp — 비역전파 RNN 학습의 두 가지 접근]({{< relref "posts/dsf-eqprop-code-analysis" >}}) | [Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation](https://arxiv.org/abs/1602.05179) → [Inferring neural activity before plasticity as a foundation for learning beyond backpropagation](https://doi.org/10.1038/s41593-023-01514-1) |
| [\[심층 분석\] Scientist Agents — AI가 과학자가 되는 길, Sakana부터 Google AI co-scientist까지]({{< relref "posts/scientist-agents-survey" >}}) | [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) |
| [\[딥리서치 보고서\] AI Scientist 방법론 — 2025.06~2026.06 1년간 논문 종합]({{< relref "posts/scientist-agents-2025-2026-report" >}}) | [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) → [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/abs/2408.06292) |
| [\[설계론\] Scientist Agents가 같아 보이지만 다른 20가지 — 차별화를 만드는 설계 차원]({{< relref "posts/scientist-agents-design-dimensions" >}}) | [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) → [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/abs/2408.06292) |
| [\[리뷰\] Cairn — 범용 문제 해결 엔진 (54/54 Tencent 해킹 챌린지 우승 시스템)]({{< relref "posts/cairn-general-problem-solver" >}}) | [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) → [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/abs/2408.06292) |
| [\[서베이\] Context Condensation 연구 동향 (2026) — 에이전트 컴팩션·RL 학습·안전성·표현 압축·이론]({{< relref "posts/context-condensation-survey-2026" >}}) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) → [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) |
| [\[리서치\] LLM Harness 최적화와 오류 — 평가·에이전트 하네스 논문 30+ 종합]({{< relref "posts/llm-harness-optimization-errors" >}}) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) → [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) → [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) |
| [\[리뷰\] The Last AI Built by Humans — 재귀적 자기개선(RSI)의 자율성 5단계 로드맵]({{< relref "posts/recursive-self-improvement-survey" >}}) | [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) → [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/abs/2408.06292) |
| [\[논문 리뷰\] VIDraft 논문 6편 묶어 읽기 — 인과 누설 감사·Aether·Darwin 병합·FINAL Bench]({{< relref "posts/vidraft-papers-review" >}}) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) → [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) |
| [\[서베이\] 인공지능과 실제 지능 — '기계와 뇌는 같은 것을 하는가'를 다룬 연구 40편]({{< relref "posts/ai-and-natural-intelligence-survey" >}}) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) → [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) → [Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217) |

## 여러 글에서 다시 사용하는 논문

### A Mathematical Theory of Communication {#shannon}

[원문](https://onlinelibrary.wiley.com/doi/10.1002/j.1538-7305.1948.tb00917.x)

**익힐 개념:** 정보량·엔트로피·확률과 로그.

**먼저 볼 부분:** 개별 사건의 정보량과 분포의 불확실성을 구분하는 설명.

### Language Generation with Strictly Proper Scoring Rules {#scoring}

[원문](https://arxiv.org/abs/2405.18906)

**익힐 개념:** 확률 예측 평가·proper scoring rule.

**먼저 볼 부분:** 예측 분포를 평가하는 학습 목표와 로그 점수의 관계.

[블로그에서 설명 읽기]({{< relref "posts/calm-tencent-series-review" >}})

### Beyond Next Token Prediction: Patch-Level Training for Large Language Models {#patch-training}

[원문](https://arxiv.org/abs/2407.12665)

**익힐 개념:** 토큰 묶음·패치 수준 학습.

**먼저 볼 부분:** 예측 단위를 다음 토큰에서 묶음으로 바꾸는 학습 구성.

[블로그에서 설명 읽기]({{< relref "posts/calm-tencent-series-review" >}})

### Attention Is All You Need {#attention}

[원문](https://arxiv.org/abs/1706.03762)

**익힐 개념:** Attention·위치 정보·인코더/디코더.

**먼저 볼 부분:** 현재 토큰이 다른 위치의 정보를 읽는 구조 그림과 Attention 설명.

### BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding {#bert}

[원문](https://arxiv.org/abs/1810.04805)

**익힐 개념:** 양방향 문맥·마스크 예측·사전학습.

**먼저 볼 부분:** 입력 표현과 가려진 토큰을 예측하는 사전학습 목표.

### ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators {#electra}

[원문](https://arxiv.org/abs/2003.10555)

**익힐 개념:** 대체 토큰 판별·생성기/판별기.

**먼저 볼 부분:** 토큰을 생성해 바꾸는 단계와 바뀐 토큰을 판별하는 단계.

### HiPPO: Recurrent Memory with Optimal Polynomial Projections {#hippo}

[원문](https://arxiv.org/abs/2008.07669)

**익힐 개념:** 상태·과거 압축·다항식 투영.

**먼저 볼 부분:** 기억을 계수로 저장하고 새 입력으로 갱신하는 문제 설정.

[블로그에서 설명 읽기]({{< relref "posts/hippo-review" >}})

### Efficiently Modeling Long Sequences with Structured State Spaces {#s4}

[원문](https://arxiv.org/abs/2111.00396)

**익힐 개념:** SSM·이산화·재귀와 컨볼루션.

**먼저 볼 부분:** 상태 공간식과 긴 시퀀스를 효율적으로 계산하는 구조.

### Mamba: Linear-Time Sequence Modeling with Selective State Spaces {#mamba}

[원문](https://arxiv.org/abs/2312.00752)

**익힐 개념:** 입력 의존 상태 갱신·선택 메커니즘.

**먼저 볼 부분:** 입력에 따라 바뀌는 값과 selective scan의 역할.

[블로그에서 설명 읽기]({{< relref "posts/mamba-ssm-review" >}})

### Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality {#mamba2}

[원문](https://arxiv.org/abs/2405.21060)

**익힐 개념:** 구조화된 Attention·상태 공간 이중성·청크 계산.

**먼저 볼 부분:** 재귀와 행렬 표현의 대응, 청크 안과 청크 사이 계산.

[블로그에서 설명 읽기]({{< relref "posts/mamba2-review" >}})

### Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks {#rag}

[원문](https://arxiv.org/abs/2005.11401)

**익힐 개념:** 검색기·생성기·외부 지식.

**먼저 볼 부분:** 질문에서 문서를 검색하고 문서와 함께 답을 만드는 구성.

[블로그에서 설명 읽기]({{< relref "posts/rag-survey-2026" >}})

### Dense Passage Retrieval for Open-Domain Question Answering {#dpr}

[원문](https://arxiv.org/abs/2004.04906)

**익힐 개념:** 질문/문서 인코더·대조 학습·negative.

**먼저 볼 부분:** 관련 문서와 오답 문서를 비교하는 학습 데이터 구성.

[블로그에서 설명 읽기]({{< relref "posts/retriever-training-survey" >}})

### ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT {#colbert}

[원문](https://arxiv.org/abs/2004.12832)

**익힐 개념:** 토큰별 표현·late interaction.

**먼저 볼 부분:** 전체 문장 벡터 하나 대신 토큰 표현들을 비교하는 구조.

[블로그에서 설명 읽기]({{< relref "posts/retriever-training-survey" >}})

### SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking {#splade}

[원문](https://arxiv.org/abs/2107.05720)

**익힐 개념:** 학습된 희소 표현·어휘 확장·역색인.

**먼저 볼 부분:** 모델 출력이 희소한 검색 표현이 되는 과정.

### Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs {#hnsw}

[원문](https://arxiv.org/abs/1603.09320)

**익힐 개념:** 근사 이웃 검색·그래프 탐색.

**먼저 볼 부분:** 이웃 연결을 따라 후보를 찾는 기본 탐색 흐름.

### Precise Zero-Shot Dense Retrieval without Relevance Labels {#hyde}

[원문](https://arxiv.org/abs/2212.10496)

**익힐 개념:** 가상 문서·임베딩 검색.

**먼저 볼 부분:** 가상 문서를 생성하고 실제 문서 검색에 사용하는 순서.

[블로그에서 설명 읽기]({{< relref "posts/rag-query-optimization-survey" >}})

### From Local to Global: A Graph RAG Approach to Query-Focused Summarization {#graphrag}

[원문](https://arxiv.org/abs/2404.16130)

**익힐 개념:** 엔터티·관계·커뮤니티 요약.

**먼저 볼 부분:** 그래프 구성과 질문에 맞춘 요약이 연결되는 구조.

### RAFT: Adapting Language Model to Domain Specific RAG {#raft}

[원문](https://arxiv.org/abs/2403.10131)

**익힐 개념:** 정답/방해 문서·근거를 읽는 SFT.

**먼저 볼 부분:** 문서를 함께 넣는 학습 예시와 답변 근거.

[블로그에서 설명 읽기]({{< relref "posts/raft-review" >}})

### Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning {#searchr1}

[원문](https://arxiv.org/abs/2503.09516)

**익힐 개념:** 검색 행동·롤아웃·검색 결과 마스킹.

**먼저 볼 부분:** 모델 행동과 도구 응답이 번갈아 나타나는 학습 경로.

[블로그에서 설명 읽기]({{< relref "posts/search-r1-review" >}})

### Proximal Policy Optimization Algorithms {#ppo}

[원문](https://arxiv.org/abs/1707.06347)

**익힐 개념:** 정책·보상·정책 갱신.

**먼저 볼 부분:** 행동의 확률을 갱신하는 목적과 큰 변경을 제한하는 장치.

[블로그에서 설명 읽기]({{< relref "posts/reinforcement-learning-beginner-guide" >}})

### Training language models to follow instructions with human feedback {#instructgpt}

[원문](https://arxiv.org/abs/2203.02155)

**익힐 개념:** SFT·선호 데이터·보상 모델·RLHF.

**먼저 볼 부분:** 지도 학습, 보상 학습, 정책 학습을 나눈 구성도.

[블로그에서 설명 읽기]({{< relref "posts/reinforcement-learning-beginner-guide" >}})

### Distilling the Knowledge in a Neural Network {#distill}

[원문](https://arxiv.org/abs/1503.02531)

**익힐 개념:** 교사/학생·확률 분포·증류.

**먼저 볼 부분:** 정답 하나와 교사 예측 분포를 학습하는 차이.

### Self-Distillation Enables Continual Learning {#sdft}

[원문](https://arxiv.org/abs/2601.19897)

**익힐 개념:** 자기증류·학생 생성 답·기존 능력 보존.

**먼저 볼 부분:** 힌트를 본 교사와 학생의 문맥 차이, 새 과제와 기존 과제 평가.

[블로그에서 설명 읽기]({{< relref "posts/sdft-self-distillation-review" >}})

### Large Memory Layers with Product Keys {#pkm}

[원문](https://arxiv.org/abs/1907.05242)

**익힐 개념:** 키/값·희소 조회·메모리 슬롯.

**먼저 볼 부분:** 쿼리가 저장 칸을 고르고 값을 읽는 구조.

[블로그에서 설명 읽기]({{< relref "posts/memory-layers-sparse-finetuning-review" >}})

### Memory Layers at Scale {#memory-scale}

[원문](https://arxiv.org/abs/2412.09764)

**익힐 개념:** 메모리 레이어·활성 용량과 총 용량.

**먼저 볼 부분:** FFN을 메모리 조회로 바꾸는 구조와 평가 목적.

[블로그에서 설명 읽기]({{< relref "posts/memory-layers-sparse-finetuning-review" >}})

### Continual Learning via Sparse Memory Finetuning {#smf}

[원문](https://arxiv.org/abs/2510.15103)

**익힐 개념:** 갱신 슬롯 선택·획득과 망각의 비교.

**먼저 볼 부분:** 어느 저장 칸을 고치는지와 순차 학습의 평가.

[블로그에서 설명 읽기]({{< relref "posts/memory-layers-sparse-finetuning-review" >}})

### Auto-Encoding Variational Bayes {#vae}

[원문](https://arxiv.org/abs/1312.6114)

**익힐 개념:** 잠재 분포·인코더/디코더·재구성.

**먼저 볼 부분:** 입력에서 잠재 표현을 만들고 복원하는 흐름.

[블로그에서 설명 읽기]({{< relref "posts/vqvae-survey" >}})

### Hierarchical Transformers Are More Efficient Language Models {#hourglass}

[원문](https://arxiv.org/abs/2110.13711)

**익힐 개념:** 다운/업샘플링·스킵 연결·인과 정렬.

**먼저 볼 부분:** 축소된 시퀀스를 처리하고 원래 길이로 복원하는 그림.

[블로그에서 설명 읽기]({{< relref "posts/hourglass-transformer-review" >}})

### MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers {#megabyte}

[원문](https://arxiv.org/abs/2305.07185)

**익힐 개념:** 바이트 패치·글로벌/로컬 모델.

**먼저 볼 부분:** 패치 사이 모델과 패치 안 모델의 역할 구분.

[블로그에서 설명 읽기]({{< relref "posts/blt-byte-level-lm-survey" >}})

### Byte Latent Transformer: Patches Scale Better Than Tokens {#blt}

[원문](https://arxiv.org/abs/2412.09871)

**익힐 개념:** 엔트로피 패칭·바이트 인코더/디코더.

**먼저 볼 부분:** 경계를 정하고 묶음을 백본에 넘기는 흐름.

[블로그에서 설명 읽기]({{< relref "posts/blt-byte-level-lm-survey" >}})

### Dynamic Chunking for End-to-End Hierarchical Sequence Modeling {#hnet}

[원문](https://arxiv.org/abs/2507.07955)

**익힐 개념:** 학습된 경계·계층적 압축·E2E.

**먼저 볼 부분:** 경계 선택이 전체 학습에 연결되는 구조.

[블로그에서 설명 읽기]({{< relref "posts/hnet-review" >}})

### No Language Left Behind: Scaling Human-Centered Machine Translation {#nllb}

[원문](https://arxiv.org/abs/2207.04672)

**익힐 개념:** 다국어 번역·인코더/디코더.

**먼저 볼 부분:** 다른 언어 사이 정보를 전달하는 번역 모델 구성.

### Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond {#laser}

[원문](https://arxiv.org/abs/1812.10464)

**익힐 개념:** 문장 임베딩·다국어 공유 공간.

**먼저 볼 부분:** 여러 언어의 문장을 공유 표현으로 만드는 목적.

### SONAR: Sentence-Level Multimodal and Language-Agnostic Representations {#sonar}

[원문](https://arxiv.org/abs/2308.11466)

**익힐 개념:** 문장 벡터·복원·텍스트와 음성의 정렬.

**먼저 볼 부분:** 문장 정보를 하나의 벡터에 모으는 병목 구조.

[블로그에서 설명 읽기]({{< relref "posts/sonar-review" >}})

### Denoising Diffusion Probabilistic Models {#ddpm}

[원문](https://arxiv.org/abs/2006.11239)

**익힐 개념:** 노이즈 추가·역방향 복원·확산 단계.

**먼저 볼 부분:** 훼손과 복원을 나눈 생성 과정 그림.

### Structured Denoising Diffusion Models in Discrete State-Spaces {#d3pm}

[원문](https://arxiv.org/abs/2107.03006)

**익힐 개념:** 이산 상태 전이·마스킹·역확산.

**먼저 볼 부분:** 연속 노이즈 대신 토큰 상태를 바꾸는 과정.

[블로그에서 설명 읽기]({{< relref "posts/diffusion-language-models-survey" >}})

### Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution {#sedd}

[원문](https://arxiv.org/abs/2310.16834)

**익힐 개념:** concrete score·확률 비율·score entropy.

**먼저 볼 부분:** 비교하는 상태와 학습하는 비율의 정의.

[블로그에서 설명 읽기]({{< relref "posts/sedd-deep-review" >}})

### Simple and Effective Masked Diffusion Language Models {#mdlm}

[원문](https://arxiv.org/abs/2406.07524)

**익힐 개념:** 마스킹 손실·복원·출력 제약.

**먼저 볼 부분:** 가린 토큰에 집중하는 학습과 샘플링 규칙.

[블로그에서 설명 읽기]({{< relref "posts/mdlm-deep-review" >}})

### The Forward-Forward Algorithm: Some Preliminary Investigations {#ff}

[원문](https://arxiv.org/abs/2212.13345)

**익힐 개념:** positive/negative 입력·층별 goodness.

**먼저 볼 부분:** 두 forward pass와 층별 학습 목표.

[블로그에서 설명 읽기]({{< relref "posts/forward-forward-review" >}})

### Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation {#eqprop}

[원문](https://arxiv.org/abs/1602.05179)

**익힐 개념:** 에너지·평형 상태·두 단계 학습.

**먼저 볼 부분:** 자유 상태와 목표로 살짝 유도한 상태의 비교.

### Inferring neural activity before plasticity as a foundation for learning beyond backpropagation {#prospective}

[원문](https://doi.org/10.1038/s41593-023-01514-1)

**익힐 개념:** 상태 최적화·가중치 학습·예측 오차.

**먼저 볼 부분:** 활성 상태를 먼저 조정하고 가중치를 나중에 바꾸는 순서.

[블로그에서 설명 읽기]({{< relref "posts/prospective-configuration-review" >}})

### Hopfield Networks is All You Need {#hopfield}

[원문](https://arxiv.org/abs/2008.02217)

**익힐 개념:** 연상 기억·패턴 완성·Attention 대응.

**먼저 볼 부분:** 단서에서 기억을 읽는 갱신과 Attention의 관계.

### ReAct: Synergizing Reasoning and Acting in Language Models {#react}

[원문](https://arxiv.org/abs/2210.03629)

**익힐 개념:** 생각·행동·관측의 반복.

**먼저 볼 부분:** 추론 문장, 도구 호출, 도구 결과의 순서.

### The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery {#ai-scientist}

[원문](https://arxiv.org/abs/2408.06292)

**익힐 개념:** 가설·실험·결과·논문·평가.

**먼저 볼 부분:** 연구 루프의 단계와 결과를 검증하는 범위.

[블로그에서 설명 읽기]({{< relref "posts/scientist-agents-survey" >}})
