---
title: "AI 처음부터 공부하기: 이 블로그의 학습 순서"
date: 2026-10-13
summary: "AI를 몰라도 시작할 수 있는 기초 수업과 관심사별 읽기 경로. 기존 글 85편을 선수 지식과 학습 목표에 따라 연결했다."
toc: true
draft: false
---

이 블로그에는 연구 노트와 논문 리뷰가 함께 있다. 날짜순으로 읽으면 처음 만나는 용어가 한꺼번에 나올 수 있다. **기초 설명 → 쉬운 흐름 이해 → 관심 분야의 개별 방법 → 비교·수식·코드** 순서로 읽어 보자. 아래는 이 블로그의 85편을 공부하기 위한 읽기 경로다.

## 아무것도 모른다면 여기서 시작하기

1. [AI 기초 설명]({{< relref "ai-basics" >}}): 수학과 코딩 없이 학습·추론·토큰·벡터·검색의 의미를 익힌다.
2. [텍스트 데이터 다루기]({{< relref "posts/build-llm-ch2-working-with-text-data" >}}): 문장 → 토큰 → 번호 → 벡터의 순서를 설명으로 먼저 읽는다.
3. [RAG 전체 흐름]({{< relref "posts/rag-survey-2026" >}}): 질문 → 문서 검색 → 자료를 참고한 답변의 경로를 본다. 제품과 벤치마크 표는 뒤에 읽어도 된다.
4. [강화학습 입문]({{< relref "posts/reinforcement-learning-beginner-guide" >}}): 앞부분의 행동·환경·보상 예시부터 읽는다. PPO·DPO의 세부 수식은 첫 독서의 목표가 아니다.
5. 아래에서 관심 경로 하나를 고른다. 전부 읽거나 논문 이름을 외우는 것을 목표로 삼지 않아도 된다.

[용어 사전]({{< relref "ai-glossary" >}})은 필요한 항목만 찾아보면 된다. Python을 모르면 코드보다 입력·중간 처리·출력의 설명부터 읽자.

## 한 번 읽고 넘어가는 대신 이렇게 공부하기

| 독서 | 할 일 | 다음 단계로 가기 전 확인 |
|---|---|---|
| 첫 번째 | 쉬운 도입과 문제·방법·한계 읽기 | 무엇이 문제이고 어느 단계를 바꿨는지 말할 수 있다 |
| 두 번째 | 그림과 비교표로 입력부터 출력까지 따라가기 | 어떤 정보를 저장하고 읽는지 설명할 수 있다 |
| 세 번째 | 필요한 수식이나 코드를 한 구간씩 보기 | 기호·변수와 앞에서 설명한 역할을 연결할 수 있다 |
| 복습 | 마지막 확인 질문에 자기 말로 답하기 | 결과의 조건과 아직 모르는 부분을 구분할 수 있다 |

막히면 다음 글로 급히 넘어가지 말고, 현재 글의 “먼저 읽기” 링크나 용어 사전으로 돌아가자. 코드 분석 글은 Python과 사용한 라이브러리를 익힌 뒤 읽는 심화 단계다. SSM·확률·학습 목적의 수식은 벡터·행렬·확률·미분을 배운 뒤 다시 보면 더 잘 연결된다.

## 관심에 맞는 경로 고르기

| 하고 싶은 일 | 출발 경로 |
|---|---|
| 글을 모델이 읽는 형태로 바꾸는 과정을 알고 싶다 | [텍스트를 모델 입력으로 바꾸기](#text) |
| 데이터로 예측하는 실습을 해 보고 싶다 | [직접 데이터로 예측해 보기](#data) |
| 내 문서에서 근거를 찾아 답하게 하고 싶다 | [자료를 찾아 답하는 AI](#rag) |
| 답변을 개선하고 새 능력을 가르치고 싶다 | [좋은 답을 배우고 기존 능력 지키기](#learning) |
| 긴 문맥을 읽고 기억하는 구조가 궁금하다 | [문장을 읽으며 기억하는 모델](#memory) |
| 토큰 대신 작은 단위를 묶는 구조가 궁금하다 | [작은 단위를 묶어 표현하기](#representation) |
| 빈칸을 복원하며 글을 만드는 방법이 궁금하다 | [여러 빈칸을 복원하며 글 만들기](#diffusion) |
| 학습의 계산 방법 자체를 바꾸는 연구가 궁금하다 | [역전파를 바꾸면 무엇이 달라질까?](#no-bp) |
| AI가 도구를 쓰고 실험하는 과정을 알고 싶다 | [도구를 쓰고 실험하는 AI](#agents) |
| 실제로 앱을 만들고 운영하는 흐름을 보고 싶다 | [앱을 만들고 운영하는 사례](#engineering) |

## 텍스트를 모델 입력으로 바꾸기 {#text}

기초 설명을 읽은 뒤 시작한다. 코드를 실행하려면 Python의 변수·리스트·함수도 알아야 한다.

앞쪽은 개념과 전체 흐름, 뒤쪽은 비교·응용·구현을 읽는 순서다. 각 글의 첫머리에도 사전 용어와 읽기 안내가 있다.

1. [\[Book\] Build a Large Language Model (Ch.2) — 텍스트 데이터 다루기]({{< relref "posts/build-llm-ch2-working-with-text-data" >}}) — 글이 토큰·번호·벡터로 바뀌는 순서를 설명하기.
2. [\[논문 리뷰\] DeBERTa v1/v2/v3 — 분리된 어텐션과 향상된 마스크 디코더로 BERT를 넘어서다]({{< relref "posts/deberta-review" >}}) — 단어의 내용과 위치를 따로 다루는 이유 이해하기.

## 직접 데이터로 예측해 보기 {#data}

텍스트 경로와 별개로 선택할 수 있다. 개념부터 읽고 Python 코드는 나중에 따라 해도 된다.

앞쪽은 개념과 전체 흐름, 뒤쪽은 비교·응용·구현을 읽는 순서다. 각 글의 첫머리에도 사전 용어와 읽기 안내가 있다.

1. [\[시험 대비\] 회귀/분류 연습 데이터셋 총정리 — sklearn, seaborn, UCI, Kaggle 즉시 사용 가이드]({{< relref "posts/data-analysis-datasets-guide" >}}) — 예측 목표에 맞는 연습 데이터 고르기.
2. [\[시험 대비\] 데이터 분석 회귀 문제 완전 가이드 — 전처리, 라이브러리, 모델, 평가]({{< relref "posts/data-analysis-regression-guide" >}}) — 수치 예측의 전처리·학습·평가 연결하기.
3. [\[실전 연습\] 회귀 문제 시뮬레이션 — California Housing에 결측치/이상치 주입 후 풀이]({{< relref "posts/data-analysis-regression-practice" >}}) — 문제 있는 데이터가 어떻게 정리되는지 따라가기.
4. [\[시험 대비\] 데이터 분석 분류 문제 완전 가이드 — 전처리, 라이브러리, 모델, 평가]({{< relref "posts/data-analysis-classification-guide" >}}) — 분류와 불균형에서 평가 기준이 달라지는 이유 이해하기.
5. [\[실전 연습\] 분류 문제 시뮬레이션 — Titanic에 결측치/이상치/불균형 강화 후 풀이]({{< relref "posts/data-analysis-classification-practice" >}}) — Titanic 예제로 분류 판단과 평가를 구분하기.

## 자료를 찾아 답하는 AI {#rag}

기초 설명의 RAG 부분부터 시작한다. 학습을 다루는 뒷부분은 강화학습 경로를 함께 참고한다.

앞쪽은 개념과 전체 흐름, 뒤쪽은 비교·응용·구현을 읽는 순서다. 각 글의 첫머리에도 사전 용어와 읽기 안내가 있다.

1. [\[서베이\] 최신 RAG 동향 총정리 (2026) — 모델, 워크플로우, 프로덕션 패턴]({{< relref "posts/rag-survey-2026" >}}) — 질문·검색·자료·답변의 전체 흐름 잡기.
2. [\[서베이\] RAG의 쿼리 성능을 높이는 모든 방법 — Rewriting, Expansion, Decomposition, Routing부터 RL까지]({{< relref "posts/rag-query-optimization-survey" >}}) — 사용자 질문을 검색하기 좋은 형태로 바꾸기.
3. [\[서베이\] 검색기(임베딩) 학습 논문 총정리 — DPR부터 NV-Embed까지 15년 진화사]({{< relref "posts/retriever-training-survey" >}}) — 관련 문서와 비슷하지만 틀린 문서를 구분하는 학습 이해하기.
4. [\[서베이\] RAG 학습용 합성 데이터 생성 논문 총정리 — E5-Mistral, Self-Instruct, InPars, Promptagator, Gecko, GPL, RAFT]({{< relref "posts/rag-synthetic-data-survey" >}}) — 문서에서 학습용 질문과 답을 만드는 과정 이해하기.
5. [\[논문 리뷰\] CRAG — 검색 결과가 잘못되었을 때 스스로 교정하는 RAG]({{< relref "posts/crag-review" >}}) — 검색 결과를 점검하고 보완하는 분기 이해하기.
6. [\[논문 리뷰\] LightRAG — 그래프 기반 RAG를 가볍고 빠르게, GraphRAG의 대안]({{< relref "posts/lightrag-review" >}}) — 문서의 대상과 관계를 함께 검색하는 방법 이해하기.
7. [\[논문 리뷰\] RAFT — '오픈북 시험 준비'처럼 LLM을 도메인 RAG에 최적화하기]({{< relref "posts/raft-review" >}}) — 정답 문서와 방해 문서를 같이 주는 학습 이해하기.
8. [\[연구분석\] 도메인 특화 RAG용 LLM 구축 완전 가이드 — 경량 기법부터 풀 파이프라인까지]({{< relref "posts/domain-optimized-llm-for-rag" >}}) — 검색 문제와 답변 문제에 맞는 학습 방법 구분하기.
9. [\[서베이\] RAG End-to-end 공동 학습 논문 총정리 — ORQA, REALM, RAG, RETRO, Atlas, REPLUG, RA-DIT, Self-RAG]({{< relref "posts/rag-end-to-end-training-survey" >}}) — 검색기와 답변기의 학습 신호가 연결되는 방식 이해하기.
10. [\[논문 리뷰\] Search-R1 — 강화학습으로 LLM에게 '언제, 무엇을 검색할지' 스스로 배우게 하기]({{< relref "posts/search-r1-review" >}}) — 검색 행동을 보상으로 학습하는 과정 따라가기.
11. [\[논문 리뷰\] Cycle-Consistent Search — 정답 라벨 없이 검색 에이전트를 학습시키는 법]({{< relref "posts/cycle-consistent-search-review" >}}) — 검색 결과로 질문을 복원하는 간접 평가 이해하기.
12. [\[논문 리뷰\] OPQE — HyDE에 강화학습을 결합한 On-Policy Pseudo-document Query Expansion]({{< relref "posts/opqe-review" >}}) — 가상 문서의 목적이 실제 검색을 돕는 데 있음을 설명하기.
13. [\[서베이\] RAG 검색 속도를 획기적으로 줄이는 방법들 — KV 캐시, 서브밀리초 인덱스, 예측적 프리페칭]({{< relref "posts/rag-latency-optimization" >}}) — 검색·입력 처리·생성 중 느린 단계 찾기.
14. [\[논문 리뷰\] TurboRAG — KV 캐시 사전 계산으로 RAG 추론 9.4배 가속]({{< relref "posts/turborag-review" >}}) — 문서 계산을 미리 해 두는 방법 이해하기.
15. [\[논문 리뷰\] CacheBlend (EuroSys 2025 Best Paper) — 비접두사 KV 캐시 재사용으로 RAG 5배 가속]({{< relref "posts/cacheblend-review" >}}) — 저장한 계산을 재사용할 때 문맥 차이를 반영하는 이유 이해하기.
16. [\[논문 리뷰\] CAGRA — GPU에서 ANN 검색을 10-77배 가속하는 NVIDIA의 그래프 인덱스]({{< relref "posts/cagra-review" >}}) — 문서 검색을 GPU에서 병렬로 수행하는 설계 이해하기.
17. [\[논문 리뷰\] Seismic (SIGIR 2024 준최우수) — 학습된 희소 검색을 서브밀리초로 가속]({{< relref "posts/seismic-review" >}}) — 검색 후보를 묶고 걸러 계산을 줄이는 흐름 이해하기.

## 좋은 답을 배우고 기존 능력 지키기 {#learning}

강화학습 입문 글의 앞부분부터 시작한다. 모델을 갱신하는 이야기와 지금 답을 만드는 이야기를 구분한다.

앞쪽은 개념과 전체 흐름, 뒤쪽은 비교·응용·구현을 읽는 순서다. 각 글의 첫머리에도 사전 용어와 읽기 안내가 있다.

1. [\[입문\] 강화학습 완전 정복 — 게임 배우는 아이부터 PPO·DPO까지 그림으로 이해하기]({{< relref "posts/reinforcement-learning-beginner-guide" >}}) — 행동·결과·보상·다음 행동의 관계 이해하기.
2. [\[논문 리뷰\] DPO — RLHF에서 강화학습을 제거하다, '당신의 LM은 몰래 보상모델이다']({{< relref "posts/dpo-review" >}}) — 두 답의 선호 비교로 모델을 조정하는 원리 이해하기.
3. [\[리뷰\] SDFT — 새로 배워도 잊지 않는 파인튜닝: Self-Distillation Enables Continual Learning]({{< relref "posts/sdft-self-distillation-review" >}}) — 예시를 본 모델이 보지 않은 모델을 가르치는 과정 이해하기.
4. [\[서베이\] 저망각 파인튜닝 연구 동향 (2026) — 온폴리시·희소 갱신·학습률·측정]({{< relref "posts/low-forgetting-finetuning-survey-2026" >}}) — 덜 배우는 것과 덜 잊는 것을 구분해 비교하기.
5. [\[리뷰\] 메모리 레이어로 잊지 않고 배우기 — Memory Layers at Scale부터 Sparse Memory Finetuning, Fast-weight PKM까지]({{< relref "posts/memory-layers-sparse-finetuning-review" >}}) — 일부 저장 칸을 읽고 고치는 방식 이해하기.
6. [\[서베이\] 도메인 학습을 더 잘 되게 하는 방법론 (2026) — 증강·리플레이·학습률·병합·자기증류·평가]({{< relref "posts/domain-adaptation-methods-survey-2026" >}}) — 분야 지식·답변 사용법·일반 능력을 구분하기.
7. [\[주간 논문\] 2026-10-01 — '온폴리시라서 덜 잊는다'가 흔들린다]({{< relref "posts/weekly-papers-2026-10-01" >}}) — 온폴리시 여부와 학습률의 효과 구분하기.
8. [\[주간 논문\] 2026-10-05 — 잘 붙인 SFT가 온폴리시 자기증류를 따라잡았다]({{< relref "posts/weekly-papers-2026-10-05" >}}) — 같은 조건의 비교 실험이 필요한 이유 이해하기.
9. [\[주간 논문\] 2026-10-12 — 온폴리시 증류에도 '신선한 롤아웃'은 필요 없었다]({{< relref "posts/weekly-papers-2026-10-12" >}}) — 만든 답변의 시점·분포·학습 방향을 나눠 읽기.
10. [\[주간 논문\] 2026-10-19 — 다시 재 보니, 자기증류의 이득은 힌트가 아니라 '생각 모드'에서 왔다]({{< relref "posts/weekly-papers-2026-10-19" >}}) — 보고된 이득을 평가 설계·학습량·업데이트 위치로 나눠 읽기.

## 문장을 읽으며 기억하는 모델 {#memory}

텍스트 경로를 먼저 읽는다. SSM은 직관부터 보고 행렬·미분 수식은 두 번째 독서에서 다룬다.

앞쪽은 개념과 전체 흐름, 뒤쪽은 비교·응용·구현을 읽는 순서다. 각 글의 첫머리에도 사전 용어와 읽기 안내가 있다.

1. [\[개념 정리\] Bayesian Surprise — 놀라움의 수학적 정의와 뇌의 주의 메커니즘]({{< relref "posts/bayesian-surprise" >}}) — 드문 사건과 믿음의 변화를 구분하기.
2. [\[개념 정리\] State Space Model(SSM) — 수식으로 이해하는 Mamba의 수학적 기반]({{< relref "posts/ssm-explained" >}}) — 새 입력과 과거 요약으로 상태가 갱신되는 흐름 이해하기.
3. [\[논문 리뷰\] HiPPO: 장거리 의존성 문제의 수학적 해결 — SSM과 Mamba의 이론적 기반]({{< relref "posts/hippo-review" >}}) — 긴 과거를 정해진 크기의 기억에 담는 문제 이해하기.
4. [\[논문 리뷰\] Mamba: 선택적 상태 공간 모델로 Transformer에 도전하다]({{< relref "posts/mamba-ssm-review" >}}) — 입력에 따라 기억 갱신이 달라지는 이유 이해하기.
5. [\[논문 리뷰\] Mamba-2: SSM과 Attention의 이중성을 밝히다]({{< relref "posts/mamba2-review" >}}) — 같은 계산을 순서대로 하거나 묶어서 하는 관점 이해하기.
6. [\[논문 리뷰\] Mamba-3: SSM 원리로 추론 효율성의 Pareto 프론티어를 진전시키다]({{< relref "posts/mamba3-review" >}}) — 상태 표현·갱신 근사·입출력 변경을 구분하기.
7. [\[서베이\] Mamba/SSM 계열 모델 완전 정리 — HiPPO에서 Nemotron-H까지]({{< relref "posts/mamba-family-complete-survey" >}}) — 모델 이름을 기억 저장·조회 방식으로 분류하기.
8. [\[코드 분석\] Mamba — state-spaces/mamba 소스코드 완전 분석]({{< relref "posts/mamba-code-analysis" >}}) — 수식의 상태 갱신과 구현의 함수 호출 대응시키기.
9. [\[연구 분석\] Selective SSM만으로는 왜 거대 언어모델이 될 수 없는가?]({{< relref "posts/why-pure-ssm-cannot-be-llm" >}}) — 요약 기억과 정확한 과거 조회가 다른 요구임을 이해하기.
10. [\[논문 리뷰\] Hymba: SSM과 Attention을 헤드 수준에서 융합한 하이브리드 아키텍처]({{< relref "posts/hymba-review" >}}) — 요약 기억과 Attention을 함께 쓰는 구조 이해하기.
11. [\[서베이\] 해마를 어떻게 구현할 것인가 — attention 옆에 기억을 따로 두는 방법들]({{< relref "posts/hippocampal-memory-attention-survey" >}}) — 비슷한 경험을 구분해 저장하고 다시 찾는 기능 나누기.
12. [SancMamba: 바이트 수준 계층적 시퀀스 압축 언어 모델 설계기 (1)]({{< relref "posts/sanc-mamba-week1" >}}) — 문자 경계와 의미 경계가 같은지 확인하는 실험 이해하기.
13. [Hymba-SANC: 유동 청킹 기반 개념 토큰 언어 모델 구현 가이드]({{< relref "posts/hymba-sanc-implementation-guide" >}}) — 청킹·기억·상위 예측을 모듈로 연결하는 설계 따라가기.

## 작은 단위를 묶어 표현하기 {#representation}

텍스트 경로의 토큰과 임베딩을 먼저 익힌다. 모델을 묶는 원리와 계산량에 집중한다.

앞쪽은 개념과 전체 흐름, 뒤쪽은 비교·응용·구현을 읽는 순서다. 각 글의 첫머리에도 사전 용어와 읽기 안내가 있다.

1. [\[서베이\] 오토인코더에서 VQ-VAE까지 — 이산 표현 학습의 계보]({{< relref "posts/vqvae-survey" >}}) — 입력을 압축했다 복원하는 구조 이해하기.
2. [\[논문 리뷰\] Hourglass Transformer — 계층적 구조로 시퀀스 모델의 효율성을 높이다]({{< relref "posts/hourglass-transformer-review" >}}) — 짧게 묶어서 처리한 뒤 원래 길이로 복원하는 흐름 이해하기.
3. [\[서베이\] BLT와 바이트 수준 언어 모델의 계보 — 토크나이저 없는 LLM을 향해]({{< relref "posts/blt-byte-level-lm-survey" >}}) — 바이트 입력이 길어져 묶음 처리가 필요한 이유 이해하기.
4. [\[논문 리뷰\] H-Net: 완전 E2E 동적 청킹으로 토크나이저를 제거하다]({{< relref "posts/hnet-review" >}}) — 묶음의 경계 자체를 학습하는 방법 이해하기.
5. [\[논문 리뷰\] MBLM — 5M 바이트 컨텍스트를 단일 GPU로 처리하는 계층적 바이트 LLM]({{< relref "posts/mblm-review" >}}) — 작은 묶음을 큰 묶음으로 쌓는 계층 구조 이해하기.
6. [\[논문 리뷰\] BLT-D — Byte Latent Transformer에 블록 디퓨전을 결합해 추론을 92% 가속]({{< relref "posts/blt-d-review" >}}) — 묶은 입력 처리와 바이트 출력 생성의 비용 구분하기.
7. [\[논문 리뷰\] SONAR — 200개 언어 + 음성을 하나의 1024차원 공간에 통합한 다국어 임베딩]({{< relref "posts/sonar-review" >}}) — 서로 다른 언어의 문장을 같은 벡터 공간에 두는 목적 이해하기.
8. [\[서베이\] 연속 의미 공간에서 예측하는 언어 모델 — Discrete Token을 넘어서 (LCM, Coconut, CALM, Diffusion LM, JEPA)]({{< relref "posts/continuous-vector-language-models" >}}) — 다음 글 조각 대신 중간 벡터를 예측한다는 뜻 이해하기.
9. [\[심화 리뷰\] CALM + Tencent WeChat AI 시리즈 — Scoring Rules · Patch Training · EAR · CALM 모든 수식 완전 해부]({{< relref "posts/calm-tencent-series-review" >}}) — 묶음 표현·예측 목표·출력 복원의 관계 이해하기.
10. [\[논문 리뷰\] Hourglass Diffusion Transformers — 메가픽셀에서 DiT의 99% 연산을 절약하다]({{< relref "posts/hourglass-dit-review" >}}) — 이미지의 해상도를 바꾸며 계산하는 구조 이해하기.

## 여러 빈칸을 복원하며 글 만들기 {#diffusion}

텍스트 경로를 먼저 읽는다. 바이트·이미지 확산의 응용은 표현 경로에도 연결돼 있다.

앞쪽은 개념과 전체 흐름, 뒤쪽은 비교·응용·구현을 읽는 순서다. 각 글의 첫머리에도 사전 용어와 읽기 안내가 있다.

1. [\[서베이\] Diffusion Language Model 완전 정복 — Diffusion-LM부터 LLaDA, Mercury, GIDD까지]({{< relref "posts/diffusion-language-models-survey" >}}) — 순차 생성과 반복 복원 생성의 차이 이해하기.
2. [\[논문 리뷰\] SEDD 심화 — Score Entropy 손실의 수식 유도와 학습/추론 완전 해부]({{< relref "posts/sedd-deep-review" >}}) — 이산 상태 사이 확률 비율을 학습한다는 뜻 이해하기.
3. [\[논문 리뷰\] MDLM 심화 — '마스킹 디퓨전 = 가중 MLM' 정리의 완전 유도와 학습/추론 해부]({{< relref "posts/mdlm-deep-review" >}}) — 가린 정도가 다른 문장을 복원하며 학습하는 흐름 이해하기.
4. [\[논문 리뷰\] LLaDA — 8B 마스킹 디퓨전 LLM, 데이터·하이퍼파라미터·학습/추론 완전 해부]({{< relref "posts/llada-review" >}}) — 양쪽 문맥을 보고 마스크를 채우는 과정 이해하기.
5. [\[논문 리뷰\] Mercury 심화 — 최초의 상용 Diffusion LLM, 학습/추론과 벤치마크 완전 해부]({{< relref "posts/mercury-deep-review" >}}) — 병렬 생성의 속도를 조건과 반복 횟수로 해석하기.

## 역전파를 바꾸면 무엇이 달라질까? {#no-bp}

기초 설명의 손실·파라미터를 익히고 용어 사전에서 역전파를 먼저 읽는다. 코드 분석은 마지막에 둔다.

앞쪽은 개념과 전체 흐름, 뒤쪽은 비교·응용·구현을 읽는 순서다. 각 글의 첫머리에도 사전 용어와 읽기 안내가 있다.

1. [역전파 없이 학습할 수 있을까? — Forward-Forward부터 EBT까지]({{< relref "posts/no-bp-methods-survey" >}}) — 학습 신호를 얻는 대안들을 구분하기.
2. [\[논문 리뷰\] Forward-Forward Algorithm: 역전파 없이 두 번의 Forward Pass로 학습하기]({{< relref "posts/forward-forward-review" >}}) — 좋은 입력과 나쁜 입력을 층마다 구분하는 학습 이해하기.
3. [\[논문 리뷰\] Prospective Configuration: 역전파를 넘어서는 학습 — 먼저 상상하고, 그다음 굳힌다]({{< relref "posts/prospective-configuration-review" >}}) — 현재 상태를 먼저 조정하고 학습하는 순서 이해하기.
4. [\[논문 리뷰\] Meta-PCN: Meta Prediction Error를 활용한 안정적이고 확장 가능한 Deep Predictive Coding Networks]({{< relref "posts/meta-pcn-review" >}}) — 층마다 예측 오차가 다를 때 생기는 문제 이해하기.
5. [\[서베이\] 역전파 없이 언어 모델을 학습하는 연구 동향과 사례]({{< relref "posts/no-bp-lm-survey" >}}) — 역전파 대신 무엇을 계산하고 저장하는지 비교하기.
6. [\[코드 분석\] Prospective Configuration — PCLayer와 PCTrainer로 구현하는 비역전파 학습]({{< relref "posts/prospective-config-code-analysis" >}}) — 상태 조정과 가중치 갱신의 코드 루프 구분하기.
7. [\[코드 분석\] DSF & EqProp — 비역전파 RNN 학습의 두 가지 접근]({{< relref "posts/dsf-eqprop-code-analysis" >}}) — 고정 피드백과 평형 상태 비교의 구현 차이 이해하기.

## 도구를 쓰고 실험하는 AI {#agents}

기초 설명과 RAG의 첫 글을 먼저 읽으면 좋다. 자동화한 기능과 실제 검증된 결과를 구분한다.

앞쪽은 개념과 전체 흐름, 뒤쪽은 비교·응용·구현을 읽는 순서다. 각 글의 첫머리에도 사전 용어와 읽기 안내가 있다.

1. [\[심층 분석\] Scientist Agents — AI가 과학자가 되는 길, Sakana부터 Google AI co-scientist까지]({{< relref "posts/scientist-agents-survey" >}}) — 가설·실험·검증·기록의 연구 루프 이해하기.
2. [\[딥리서치 보고서\] AI Scientist 방법론 — 2025.06~2026.06 1년간 논문 종합]({{< relref "posts/scientist-agents-2025-2026-report" >}}) — 계산 결과와 실제 실험 검증을 나눠 읽기.
3. [\[설계론\] Scientist Agents가 같아 보이지만 다른 20가지 — 차별화를 만드는 설계 차원]({{< relref "posts/scientist-agents-design-dimensions" >}}) — 누가 결정하고 어디에 기록하는지로 설계를 비교하기.
4. [\[리뷰\] Cairn — 범용 문제 해결 엔진 (54/54 Tencent 해킹 챌린지 우승 시스템)]({{< relref "posts/cairn-general-problem-solver" >}}) — 공유 기록을 통해 다음 행동을 조율하는 구조 이해하기.
5. [\[서베이\] Context Condensation 연구 동향 (2026) — 에이전트 컴팩션·RL 학습·안전성·표현 압축·이론]({{< relref "posts/context-condensation-survey-2026" >}}) — 긴 작업 기록에서 무엇을 줄이고 남기는지 이해하기.
6. [\[리서치\] LLM Harness 최적화와 오류 — 평가·에이전트 하네스 논문 30+ 종합]({{< relref "posts/llm-harness-optimization-errors" >}}) — 모델과 실행·채점 조건의 영향을 구분하기.
7. [\[리뷰\] The Last AI Built by Humans — 재귀적 자기개선(RSI)의 자율성 5단계 로드맵]({{< relref "posts/recursive-self-improvement-survey" >}}) — 자동 수정과 실제 개선의 증거 구분하기.
8. [\[논문 리뷰\] VIDraft 논문 6편 묶어 읽기 — 인과 누설 감사·Aether·Darwin 병합·FINAL Bench]({{< relref "posts/vidraft-papers-review" >}}) — 논문 설명·공개 코드·평가 결과를 대조하는 읽기 연습.
9. [\[서베이\] 인공지능과 실제 지능 — '기계와 뇌는 같은 것을 하는가'를 다룬 연구 40편]({{< relref "posts/ai-and-natural-intelligence-survey" >}}) — 비슷한 결과와 같은 작동 원리를 구분하기.

## 앱을 만들고 운영하는 사례 {#engineering}

회사 궁합과 홈런 더비 제작기는 AI 지식 없이 읽어도 된다. 사진·점수 계산에서 서비스 운영으로 이어지는 과정과, 게임 입력·보상·서버 검증을 연결하는 과정을 살펴본다.

앞쪽은 개념과 전체 흐름, 뒤쪽은 비교·응용·구현을 읽는 순서다. 각 글의 첫머리에도 사전 용어와 읽기 안내가 있다.

1. [회사 궁합 앱 제작기 (1): 아이디어에서 첫 버전까지]({{< relref "posts/check-fit-company-1" >}}) — 입력 정보가 점수와 화면 결과로 바뀌는 흐름 이해하기.
2. [회사 궁합 앱 제작기 (2): 내 맥북에서 서비스해 보기]({{< relref "posts/check-fit-company-2" >}}) — 내 컴퓨터가 하는 일과 방문자의 기기가 하는 일 구분하기.
3. [회사 궁합 앱 제작기 (3): Supabase와 Vercel로 진짜 서비스 만들기]({{< relref "posts/check-fit-company-3" >}}) — 화면·데이터 저장·접근 권한의 역할 나누기.
4. [회사 궁합 앱 제작기 (4): 메일 없이 로그인하기]({{< relref "posts/check-fit-company-4" >}}) — 임시 계정의 저장·가입·기록 이전 구분하기.
5. [\[영상분석\] 8년간의 엣지 플랫폼 구축과 그 종료 — Atlassian Envoy 플랫폼의 아키텍처·진화·유지보수에 관한 사례 연구]({{< relref "posts/atlassian-edge-platform-8years" >}}) — 요청 처리와 설정 변경의 수명주기 구분하기.
6. [\[대회\] 공공조달 법령위반 판정 — 추출·결정 분리, 법정 상수, 시간 예산 설계 (DACON 236754)]({{< relref "posts/dacon-procurement-law-violation" >}}) — 자료 추출과 위반 판정의 역할을 나누는 사례 이해하기.
7. [\[리서치\] 기업은 지식과 콘텐츠를 어떻게 관리하나 — KMS·CMS의 제품·가격·실제 운영·지식 내재화·데이터 축적 (2026)]({{< relref "posts/kms-cms-enterprise-guide-2026" >}}) — 지식이 기록·검색·체득되는 과정과 그 비용 구조 이해하기.

8. [홈런 더비 제작기: 탭 한 번으로 치는 도트 야구 게임 만들기]({{< relref "posts/homerun-derby" >}}) — 타격 규칙·도트 그래픽·성장 요소와 서버 기록 검증 연결하기.

## 주간 논문과 최신 사례는 언제 읽을까?

주간 논문은 기초 수업보다 관련 분야를 복습하는 글에 가깝다. 학습·증류·망각 경로의 앞선 글을 읽은 뒤 날짜순으로 따라가면 논쟁의 변화를 보기 쉽다. 영상 분석과 대회 기록은 실제 시스템에서 어떤 선택을 했는지 살펴보는 사례다.

한 경로를 마쳤다면 모델 이름보다 질문을 정리해 보자. “어디가 느린가?”, “무엇을 기억해야 하나?”, “학습 뒤 무엇을 잃었나?”, “실제로 검증됐나?”처럼 다음 글을 읽는 기준을 갖추면 된다.

## 지금 읽는 논문의 배경이 막힌다면

각 글의 “선수 개념과 읽을 논문” 표를 먼저 보자. 읽을 순서, 익힐 개념, 원문에서 먼저 볼 부분을 함께 적었다. [선수 논문 전체 지도]({{< relref "paper-prerequisites" >}})에서는 글별 목록과 공통 기반 논문을 찾을 수 있다. 이미 알고 있는 개념은 넘어가고, 현재 글에서 이해되지 않는 연결부터 보완하면 된다.
