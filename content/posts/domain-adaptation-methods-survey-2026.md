---
title: "[서베이] 도메인 학습을 더 잘 되게 하는 방법론 (2026) — 증강·리플레이·학습률·병합·자기증류·평가"
date: 2026-10-06
tags: ["연구노트", "서베이", "도메인적응", "파인튜닝", "연속사전학습", "지식주입", "합성데이터", "LoRA", "CatastrophicForgetting", "LLM", "2026"]
categories: ["ML/AI"]
summary: "LLM에 의료·법률·금융·사내 문서 같은 도메인을 학습시킬 때 실제로 효과가 확인된 방법을 데이터·학습 레시피·사후학습·평가 네 갈래로 정리했다. 작은 코퍼스는 다양하게 증강해야 들어가고(EntiGraph 39.5→56.2), 일반 데이터를 1~25% 섞어야 덜 잊으며, 학습률은 분포 이동 크기와 토큰 예산에 따라 정반대 처방이 나온다. instruct 모델에 바로 CPT하면 대화 능력이 무너지므로(MT-Bench 7.89→약 1.0) base에서 학습하고 병합으로 되돌린다. SFT는 지식이 아니라 사용법을 가르치고, 새 사실을 SFT로 넣으면 환각이 는다. 도메인 RL은 대체로 이미 있는 지식을 끌어낸다. 그리고 공정하게 평가하면 공개 의료 특화 모델의 대부분이 자기 base를 이기지 못했다. 2023~2026년 논문 약 80편 기준."
math: true
toc: true
draft: false
---

## Executive Summary

"우리 도메인 데이터로 더 학습시키면 더 잘하겠지"는 생각보다 자주 틀린다. 원문을 그대로 이어 학습하면 지식은 **저장되지만 꺼내 쓰지 못하고**, instruct 모델의 대화 능력은 무너지고, 공정하게 재면 base 모델보다 나아진 게 없는 경우가 많다. 2023~2026년 연구는 **무엇을 바꾸면 도메인 학습이 실제로 되는가**를 꽤 구체적으로 밝혀 왔다. 레버는 여섯 개로 정리된다.

| 레버 | 핵심 처방 | 대표 근거 |
|---|---|---|
| **① 데이터 증강** | 작은 코퍼스는 원문 그대로가 아니라 **다양한 형태로 수백 배 불려서** 넣는다. 문서와 QA를 함께 | Physics of LMs 3.1, EntiGraph, Active Reading, SPA, Synthetic Mixed Training |
| **② 리플레이·혼합** | 일반 데이터를 **약한 이동이면 1~5%, 강한 이동이면 약 25%** 섞는다. 리플레이는 목표 성능도 올린다 | Ibrahim et al., Bethune et al., Kotha & Liang, D-CPT·CMR 법칙 |
| **③ 학습률·스케줄** | 분포 이동 크기와 토큰 예산에 따라 정한다. 큰 이동·큰 예산은 재워밍, 작은 예산은 낮게. 끝은 고품질 데이터로 어닐링 | Ibrahim et al., NVIDIA Reuse, ChipNeMo, Stability Gap, Llama 3·OLMo 2 |
| **④ 시작점·병합** | instruct가 아니라 **base에서 학습**하고, 차이 벡터를 더하거나 병합해 대화 능력을 되돌린다 | FinDAP, Instruction Residual, Chat Vector, Shadow-FT, Grafting |
| **⑤ 사후학습** | SFT는 **사용법**을 가르치는 단계다. 새 사실은 앞 단계에서, 또는 "문서를 본 자신"을 교사로 증류해서 넣는다. RL은 끌어내기 도구 | Gekhman et al., Prompt Distillation, SDFT, Yue et al., Interplay |
| **⑥ 평가** | base와 1:1로, 모델별로 튜닝한 프롬프트와 통계 검정으로 잰다. 일반 능력 유지를 함께 보고한다 | Jeong et al., Contamination 연구, HealthBench |

한 문장 결론:

> 도메인 학습이 실패하는 가장 흔한 이유는 방법론이 아니라 **"같은 사실을 한 가지 형태로 한 번만 보여주고, 일반 데이터 없이, 잘못된 체크포인트에서, 잘못 재는 것"**이다. 그리고 그 전에 **"이 도메인이 정말 base 모델에 없는가"**를 먼저 확인해야 한다.

이 글은 모델 **파라미터**에 도메인을 넣는 방법에 집중한다. 검색으로 붙이는 쪽은 [도메인 특화 RAG LLM 구축 가이드](../domain-optimized-llm-for-rag/)에서 다뤘다. 망각 자체의 메커니즘은 [저망각 파인튜닝 서베이](../low-forgetting-finetuning-survey-2026/)에 있다.

---

## 0. 먼저 물을 것: 정말 도메인 학습이 필요한가

가장 불편한 결과부터 보자.

**Medical Adaptation of LLMs and VLMs: Are We Making Progress?** (Jeong, Garg, Lipton, Oberst — CMU·JHU, EMNLP 2024)
- 공개 의료 특화 LLM 7종(Meditron, BioMistral, OpenBioLLM 등)을 **각자의 base 모델과 1:1로** 비교했다. 프롬프트는 모델마다 따로 최적화하고 통계 검정을 했다.
- 3-shot 의료 QA에서 의료 모델은 **12.1%만 유의하게 이겼다.** 49.8%는 비겼고 38.2%는 졌다.
- 프롬프트를 의료 모델에만 맞춰 최적화하면 승률이 **70%를 넘었다.** 그동안 보고된 "도메인 이득"의 상당 부분이 평가 방식에서 나왔다는 뜻이다.
- 확장판(2411.08870)에서 Meditron-7B는 3-shot 비교의 76.9%에서 졌다.

비슷한 신호가 여럿 있다.
- BloombergGPT(50B, 금융 데이터로 처음부터 학습)가 나온 직후 GPT-4가 금융 텍스트 분석에서 대등하거나 더 나았다는 보고가 나왔다.
- Llama 3에서 도메인 데이터 어닐링은 8B의 GSM8K를 24.0%, MATH를 6.4% 올렸지만 **405B에서는 효과가 미미했다.**
- 최신 대형 모델은 벤치마크 사실의 95~98%를 이미 내부에 갖고 있고, 병목은 저장이 아니라 **꺼내 쓰기(recall)**라는 분석도 있다(2602.14080).

**정리.** 도메인 학습의 이득이 가장 확실한 경우는 **웹에 거의 없는 데이터**다. 사내 문서, 칩 설계 데이터(ChipNeMo), 학습 마감 이후의 사건, 웹과 거리가 먼 전문 분야가 그렇다. 공개 의료 논문처럼 이미 사전학습에 많이 들어간 분야라면, 먼저 **가장 강한 범용 base + 좋은 프롬프트 + RAG**와 비교하고 시작하는 게 맞다.

---

## 1. 왜 "그냥 더 학습"은 잘 안 되나

도메인 학습에서 넣으려는 것은 세 가지다.
- **지식**: 사실, 개념, 용어 ("이 약의 금기 사항은 X다")
- **형식·사용법**: 그 분야의 답변 방식, 지시 따르기
- **추론**: 그 분야에서 문제를 푸는 절차

이 셋은 학습 방식이 다르다. 그리고 지식 쪽에 잘 알려진 함정이 몇 가지 있다.

**저장은 되는데 꺼내지 못한다 — Physics of Language Models 3.1** (Allen-Zhu & Li, 2309.14316)
- 가상 인물 10만 명의 전기를 학습시킨 통제 실험이다.
- 인물마다 전기를 **한 가지 문장**으로만 보여주면, 모델은 전기를 외우지만 QA로 물으면 **약 9.7%**만 맞혔다.
- 인물마다 표현이 다른 전기를 **5개** 보여주면 **96.6%**가 됐다. 문장 순서만 섞어도 4.4%에서 약 70%로 올랐다.
- 탐침(probing)으로 보니, 증강이 있을 때는 속성이 **인물 이름 토큰 위에** 거의 선형으로 저장됐다. 증강이 없으면 여러 토큰에 흩어져 꺼낼 수 없었다.
- 일부 인물만 증강해도 증강하지 않은 인물의 정답률이 4.4%에서 86.8%로 올랐다("유명인이 소수를 돕는다"). 모델이 "지식을 꺼낼 수 있는 형태로 저장하는 법"을 배운다는 뜻이다.

**perplexity가 낮아도 답은 못 한다 — Instruction-tuned LMs are Better Knowledge Learners** (Jiang et al., 2402.12847)
- 문서의 perplexity를 아주 낮게 떨어뜨려도 그 문서에 대한 QA 정답률은 낮게 머물렀다("perplexity curse").

**역방향은 따로 배워야 한다 — Reversal Curse** (Berglund et al., 2309.12288)
- "A는 B다"로 배운 모델은 "B는 무엇인가"에 잘 답하지 못한다. GPT-4는 "톰 크루즈의 어머니는?"에 79% 답했지만 반대 방향 질문에는 33%만 답했다.

**초반에는 오히려 나빠진다 — Stability Gap** (Guo et al., 2406.14833)
- 의료 CPT를 시작하면 도메인 perplexity는 꾸준히 좋아지는데, 의료 과제 정확도와 일반 정확도는 **먼저 떨어졌다가 V자로 회복**했다. 초반에 지시 따르기 능력이 흔들리기 때문으로 본다.
- 실무적으로는 **초반 지표 하락을 보고 학습을 일찍 멈추지 말라**는 뜻이다.

이 네 가지가 아래 처방들의 공통 배경이다.

---

## 2. 데이터 레버

### 2.1 작은 코퍼스는 다양하게 증강해서 넣는다

도메인 문서가 수백~수천 개뿐이라면 원문을 반복 학습하는 것은 거의 효과가 없다. 대신 **같은 내용을 여러 형태로 다시 쓴 합성 데이터**를 크게 만든다.

**Synthetic Continued Pretraining (EntiGraph)** (Yang, Band, Li, Candès, Hashimoto — Stanford, 2409.07431)
- 문서에서 엔티티를 뽑고, 엔티티 쌍·삼중쌍 사이의 관계를 GPT-4-Turbo가 분석하게 해서 다양한 글을 만든다. 다양성을 "지식 그래프의 조합"으로 확보하는 방식이다.
- QuALITY(문서 265개, 1.3M 토큰)를 455M 토큰으로 약 **350배** 불렸다. Llama 3 8B, 닫힌 책 객관식:

| 방법 | 정확도 |
|---|---|
| base | 39.49 |
| 원문 CPT | base 수준 또는 약간 아래 |
| 패러프레이즈 CPT | 약 43 (잘 늘지 않아 38M 토큰에서 중단) |
| **EntiGraph CPT (455M)** | **56.22** |
| GPT-4 (닫힌 책) | 51.30 |
| base + RAG | 60.35 |
| EntiGraph + RAG | 62.60 |

- 정확도는 합성 토큰 수에 **로그 선형**으로 늘었다. RAG와 함께 쓰면 효과가 더해졌다.
- 학습 설정: 배치의 10%를 일반 데이터(RedPajama)로 리플레이, 2 epoch, 최대 학습률 5e-6.

**Learning Facts at Scale with Active Reading** (Lin et al. — Meta FAIR, 2508.09494)
- 모델이 문서마다 **스스로 공부 전략**(패러프레이즈, 개념 연결, 능동 회상, 비유 등)을 정하고, 그 전략대로 학습 데이터를 만든다.
- Llama-3.1-8B, SimpleQA의 Wikipedia 기반 부분집합: 원문 반복 학습 15.9% → 패러프레이즈 25.7% → 합성 QA 47.9% → **Active Reading 66%**(반복 학습 대비 +313%). FinanceBench는 26%(+160%)였다.
- 패러프레이즈와 합성 QA는 데이터를 늘려도 정체했지만 Active Reading은 약 40억 단어까지 계속 올랐다. 저자들은 다양성 차이로 설명한다.
- 규모를 키우자 **사전학습 수준의 학습률**(1e-5가 아니라 3e-4)과 **증강 데이터:사전학습 데이터 1:1** 혼합이 필요했다. 이렇게 1T 토큰으로 학습한 WikiExpert-8B는 SimpleQA 23.5%로 Llama 3.1 405B(17.1%)를 넘었다.

**SPA: A Simple but Tough-to-Beat Baseline for Knowledge Injection** (Tang, Wang, Wang, Lyu — 칭화대, 2603.22213)
- 학습과학에 근거한 **고정 프롬프트 7개**(핵심 개념, 마인드맵, 함의, 심화 QA, 사례 연구, 토론, 선생님식 설명)로 생성량만 키운다.
- QuALITY 455M 토큰에서 57.03으로 EntiGraph(56.22)와 Active Reading 재현치(51.75)를 넘었다. 생성기로 gpt-oss-120b를 써서 GPT-4-Turbo보다 약 50배 쌌다.
- RL로 학습시킨 증강기(SEAL)는 데이터가 늘면 **다양성이 무너져** 일찍 포화했다고 보고한다.
- 다만 Active Reading과의 순위는 각 그룹의 프롬프트 튜닝에 좌우된다. 제3자 비교는 아직 없다.

**Synthetic Mixed Training + Focal Rewriting** (Han et al. — Stanford·MIT·UW, 2603.23562)
- 합성 문서만, 또는 합성 QA만 늘리면 **RAG 아래에서 정체**했다. 생성기를 8B에서 70B로 키워도 QA 생성에서는 0.1% 차이였다.
- 해법: 합성 QA와 합성 문서를 **1:1**로 섞는다. 문서를 생성할 때 특정 질문에 초점을 맞추게 해서("{q}에 초점을 맞춰 다시 써라") 다양성을 높인다. FineWeb 10% 리플레이를 유지한다.
- Llama 3.1 8B Instruct에서 6개 설정 중 5개에서 RAG를 넘었다. QuALITY는 RAG 65.3 → 68.2 → (+RAG) 69.7, LongHealth는 RAG 70.0 → 71.2 → (+RAG) 80.3이었다.

**다양성이 핵심이라는 다른 근거들**
- **Fine-Tuning or Retrieval?** (Ovadia et al. — Microsoft, 2312.05934): 시사 문제 지식 주입에서 패러프레이즈 개수를 1개에서 10개로 늘리면 정확도가 약 0.50에서 0.59로 단조 증가했다. 그래도 RAG(0.875)에는 크게 못 미쳤다.
- **Injecting New Knowledge via SFT** (Mecklenburg et al. — Microsoft, 2404.00213): 문서당 N 토큰을 만드는 방식보다 **사실 하나당 QA를 고르게** 만드는 방식이 더 고르게 주입됐다.
- **New News** (Park, Zhang, Tanaka, 2505.01812): 모델이 직접 만든 Self-QA가 가장 좋았다. 단, 합성 QA를 학습할 때 **원문을 컨텍스트에 붙이면 학습이 크게 망가졌다**("contextual shadowing"). 답이 컨텍스트에 있으니 가중치에 저장할 필요가 없어지는 것이다.
- **Demystifying Synthetic Data in LLM Pre-training** (Meta, 2510.01631): 사전학습 규모에서는 재작성 데이터만으로는 빨라지지 않았고, **약 1/3 재작성 + 2/3 원문**일 때 같은 손실에 5~10배 빨리 도달했다. 약 8B보다 큰 생성기는 더 나은 데이터를 만들지 못했다.

**주의할 점.**
- 생성기의 환각이 그대로 가중치에 들어간다. 이득의 일부는 강한 생성기로부터의 **증류**일 수 있다.
- **놀라운 사실**은 엉뚱한 문맥으로 번진다. "How new data permeates LLM knowledge"(Google DeepMind, 2504.09522)는 학습 전 확률이 낮은 사실일수록 무관한 문맥에 새어 나오는 "priming"이 심하다고 보고했다. 중간 개념을 거쳐 설명하는 "징검다리" 증강과, 파라미터 업데이트 상위 8%를 버리는 방법이 이를 50~95% 줄였다.

### 2.2 문서만이 아니라 "질문받을 방식"도 함께 보여준다

- **Pre-Instruction-Tuning** (2402.12847): QA 쌍을 먼저 학습하고, 그다음 문서와 QA를 함께 학습했다. Llama-2 7B의 Wiki2023 정답률이 30.3%에서 **48.1%**로 올랐다.
- **AdaptLLM — Reading Comprehension** (Cheng, Huang, Wei — Microsoft, 2309.09530): 원문 CPT는 지식은 넣지만 **프롬프트 응답 능력을 해친다.** 원문을 요약, 단어→문장, 추론, 패러프레이즈 같은 "독해 문제" 형식으로 바꾸고 일반 지시 데이터를 섞었다. LLaMA-7B 금융 63.4로 원문 CPT(57.6)를 넘었고, BloombergGPT-50B(62.5)와 비슷했다.
- **Instruction Pre-Training** (Microsoft·칭화대, 2406.14491): 원문에 근거한 지시-응답 쌍을 대량 합성해 CPT 코퍼스에 넣었다. Llama3-8B의 금융·바이오 CPT가 여러 과제에서 Llama3-70B와 대등하거나 나았다. 다만 합성 응답의 정답률은 약 70~77%로 잡음도 함께 들어간다.

**공통 메시지.** 지식을 넣는 단계에서부터 "이 지식이 어떻게 질문될지"를 함께 보여줘야 꺼내 쓸 수 있는 형태로 저장된다.

### 2.3 일반 데이터를 섞는다 (리플레이)

도메인 데이터만으로 학습하면 일반 능력을 잃는다. 해법은 원래 사전학습과 비슷한 일반 데이터를 섞는 **리플레이**다. 비율은 분포 이동의 크기에 따라 다르다.

**Simple and Scalable Strategies to Continually Pre-train LLMs** (Ibrahim et al. — Mila·EleutherAI, 2403.08763)
- 405M·10B 모델을 Pile로 사전학습한 뒤 SlimPajama(약한 이동) 또는 독일어(강한 이동)로 이어 학습했다.

| Pile → 독일어 (강한 이동) | Pile 손실 | 독일어 손실 | 평균 |
|---|---|---|---|
| 리플레이 없음 | 3.56 | 1.11 | 2.34 |
| 1% 리플레이 | 2.83 | 1.12 | 1.97 |
| 25% 리플레이 | 2.33 | 1.16 | 1.75 |

- 약한 이동은 **5%**, 강한 이동은 **25%**를 골랐다. 학습률 재워밍·재감쇠와 함께 쓰면 **두 데이터를 합쳐 처음부터 다시 학습한 것과** 검증 손실이 거의 같았다.

**실제 도메인 모델들의 선택**: Meditron 1%(리플레이로 평균 +1.6%), SaulLM 약 2%, Arcee의 SEC 공시 CPT 약 1.4%, EntiGraph·Active Reading·Synthetic Mixed Training 10%, 대규모 Active Reading 50%.

**리플레이는 망각 방지만이 아니다.**
- **Scaling Laws for Forgetting with Pretraining Data Injection** (Apple, 2502.06042): 파인튜닝 데이터에 사전학습 데이터를 **1%만** 넣어도 사전학습 분포에 대한 망각이 막혔다. 목표 도메인 손실은 거의 변하지 않았다.
- **Replaying pre-training data improves fine-tuning** (Kotha & Liang — Stanford, 2603.04964): 일반 데이터를 섞으면 **목표 과제 성능도 올랐다.** 데이터 효율이 파인튜닝에서 최대 1.87배, mid-training에서 2.06배였다. 8B Llama 3에서는 웹 내비게이션 +4.5%, 바스크어 QA +2%였다. 목표 데이터가 사전학습에 없었을수록 효과가 컸다.
- 원래 사전학습 데이터를 구할 수 없다면, 모델이 **스스로 생성한 텍스트**를 리플레이로 써도 망각이 거의 사라졌다는 보고가 있다(2605.26097).

**비율을 미리 예측하는 법칙.** D-CPT Law(2406.01375)와 CMR Scaling Law(2407.17467)는 작은 실험으로 "도메인 비율 대 일반 손실" 곡선을 맞춰 최적 비율을 예측한다. CMR에서는 일반 손실 허용폭 0.05 기준으로 금융 도메인 비율이 모델이 클수록 커졌다(460M 29.8% → 3.1B 47.8%). 단, 두 법칙 모두 **손실 기준**이고 4B 이하 모델에서 맞춘 것이라 다운스트림 성능까지 보장하지는 않는다.

**정직한 정리.** 권장 비율이 연구마다 1%에서 70%(일반 데이터 기준)까지 퍼져 있다. "손실 허용폭"과 "다운스트림 점수" 등 기준이 다르기 때문이다. 출발점은 **약한 이동 1~5%, 소규모 합성 CPT 10%, 언어 수준의 강한 이동 25% 이상**으로 잡고, 짧은 실험으로 조정하는 것이 현실적이다.

### 2.4 데이터를 고르고, 양보다 질과 반복

- **Stability Gap 논문**(2406.14833)의 처방이 가장 구체적이다. OpenLLaMA-3B 의료 CPT에서 50B 토큰을 한 번 도는 것보다, **KenLM perplexity로 고른 최고 품질 5B 토큰을 약 4 epoch** 도는 쪽이 나았다. 원래 사전학습의 데이터 구성비를 따라 도메인 데이터로 바꿔 넣는 것도 도움이 됐다.

| OpenLLaMA-3B 의료 평균 | 점수 |
|---|---|
| base | 36.2 |
| 50B 전체 1회 | 37.4 |
| 재워밍·재감쇠 | 37.7 |
| 10% 리플레이 | 37.5 |
| **제안 레시피 (20B 예산)** | **40.7** |

- 시점 이후 웹 데이터로 이어 학습한 연구(2609.23916)에서도 큐레이션한 6B 토큰이 40B 토큰과 맞먹었다.
- **목표 데이터와의 유사도로 고르기**: DSIR(2302.03169)은 해시 n-gram 중요도 가중치로 전문가 큐레이션과 비슷한 성능을 냈다. LESS(2402.04333)는 그래디언트 특징으로 고른 5% 데이터가 전체 데이터를 이기는 경우가 있었다.
- **도메인 전용 품질 필터**: 프랑스어 의료 인코더 연구(2606.22079)에서는 "교육적 품질" 같은 범용 필터보다 의료 용어 밀도 필터가 나았다.
- **작은 실험 한 번을 믿지 말 것**: 데이터 출처의 가치 순위가 계산 규모에 따라 뒤집혔다(2507.22250). 출처마다 작은 스케일링 곡선을 그려 보는 편이 안전하다.

### 2.5 사전학습을 직접 할 수 있다면

- **The Finetuner's Fallacy** (DatologyAI, 2603.16177): 작은 도메인 데이터를 파인튜닝까지 아껴두지 말고 **사전학습에 1~5% 비율로 10~50번 반복해 섞으면** 같은 도메인 점수에 최대 1.75배 적은 토큰으로 도달했다. 웹과 거리가 먼 도메인에서는 이렇게 학습한 1B가 일반 3B를 이겼다. 이후 파인튜닝에서도 덜 잊었다.
- 단, 너무 많이 반복하면 **기억 붕괴**가 온다. 사실을 사전학습에 수천 번 이상 넣으면 보존이 급격히 나빠졌고, 큰 모델일수록 더 낮은 빈도에서 무너졌다(Knowledge Infusion Scaling Law, 2509.19371).

---

## 3. 학습 레시피 레버

### 3.1 학습률: 정반대 처방이 공존한다

이 부분은 연구마다 처방이 정반대로 보인다. 하지만 **분포 이동 크기와 토큰 예산**으로 나누면 대부분 정리된다.

| 상황 | 처방 | 근거 |
|---|---|---|
| 큰 예산(≥100B 토큰) + 강한 이동 | 원래 최대 학습률의 0.5~1배로 **재워밍**, 코사인으로 0.1배까지 감쇠, 워밍업 약 1% | Ibrahim et al.(2403.08763), Meditron(2311.16079) |
| 약한 이동, 일반 능력 향상 목적 | 사전학습의 **최소** 학습률에서 워밍업 없이 시작, η_min/100까지 감쇠 | NVIDIA Reuse, Don't Retrain(2407.07263) |
| 작은 예산(수~수십 B 토큰) + 강한 모델 보존 | **낮게.** ChipNeMo는 상수 5e-6, 워밍업·스케줄러 없음 | ChipNeMo(2311.00176) |
| 큰 모델 | 작은 모델보다 낮게. 3B에서 3e-4는 해로웠고 3e-5가 나았다 | Stability Gap(2406.14833) |

- **재워밍의 비용**: 학습률을 다시 올리는 순간 손실이 튄다. 같은 데이터로 재워밍해도 튄다(Gupta et al., 2308.04014). 워밍업 길이 자체는 크게 중요하지 않았다.
- **NVIDIA의 반대 결과**: 8T 토큰으로 학습한 15B 모델을 300B 토큰 더 학습할 때는 **어떤 워밍업도 성능을 떨어뜨렸다.** 평균 정확도가 48.9에서 56.1로 올랐다. Ibrahim et al.과 결론이 다른 것은 설정이 다르기 때문이다. NVIDIA는 약한 이동으로 일반 능력을 높이려 했고, Ibrahim et al.은 큰 분포 이동을 다뤘다.
- **ChipNeMo의 경고**: CodeLlama 수준의 큰 학습률을 쓰자 손실이 튀고 **코딩을 뺀 모든 도메인·학술 벤치마크가 크게 나빠졌다.**
- **예측 법칙**: Learning Dynamics in Continual Pre-Training(2505.07796, ICML 2025)은 CPT 손실 곡선을 "분포 이동 항"과 "학습률 감쇠 항"으로 나눠, 임의의 스케줄·최대 학습률·리플레이 비율에서 손실을 예측했다. 분포 이동이 약하거나 사전학습이 부족했던 경우가 아니면, **일반 손실은 아무리 오래 학습해도 base보다 낮아지지 않았다.**

**어닐링과 mid-training.** 학습 마지막에 학습률을 0으로 내리면서 고품질 도메인 데이터 비중을 높이는 방식이 표준이 됐다.
- Llama 3: 도메인 데이터 업샘플링 어닐링으로 8B의 GSM8K +24.0%, MATH +6.4%. 405B에서는 미미했다. 50% 학습된 8B를 40B 토큰 동안 어닐링(새 데이터 30%)해 **데이터셋의 가치를 평가하는** 용도로도 썼다.
- OLMo 2: 학습 FLOPs의 5~10%를 mid-training에 쓰고, 데이터 순서를 바꾼 어닐링 여러 개를 **평균(souping)**했다. 평균이 개별 실행보다 항상 나았다. 7B GSM8K 24.1 → 67.5.
- "Does your data spark joy?"(Databricks, 2406.03476): 학습 마지막 **10~20%** 구간에서 도메인 데이터를 업샘플링하는 것이 가장 좋았다.
- 주의: mid-training 이득의 일부는 도메인 데이터가 아니라 **학습률을 내린 것 자체**에서 온다.

### 3.2 어디서 시작하나: base에서 배우고 병합으로 되돌린다

**instruct 모델에 바로 CPT하면 대화 능력이 무너진다.**
- **FinDAP** (Salesforce, 2501.04961): Llama-3-8B-Instruct에 금융 CPT만 하자 MT-Bench가 **7.89에서 약 1.0**으로 떨어졌다. 도메인 데이터든 일반 데이터든 마찬가지였다. 망각 크기는 CPT > 지시 튜닝(IT) > 선호 정렬 순이었다. 해법은 **CPT와 IT를 한 번에 함께 학습**하는 것이었다(순차보다 나았다). 최종 모델의 CFA 점수는 34.4에서 55.6으로 올랐고 MMLU는 48.1에서 47.4로 유지됐다.
- **Balancing CPT and Instruction Fine-Tuning** (Jindal et al., 2410.10739): instruct 모델에 1B 토큰을 CPT하자 IFEval이 약 10점 떨어졌다.

**대신 base에서 학습하고 차이 벡터를 더한다.**
- **Instruction Residual**(2410.10739): CPT한 base에 "instruct − base" 가중치 차이를 더하면 원래 instruct 모델보다 약 4~5점 나았다. 같은 계열 안에서는 다른 버전의 차이도 통했다(Llama-3.1 차이를 Llama-3 base에 더함). 단, 1.5B 이하에서는 잘 통하지 않았다.
- **Chat Vector**(2310.04799): 새 언어로 CPT한 base에 "chat − base"를 더해 SFT 없이 지시 따르기와 안전성을 되살렸다.
- **Shadow-FT**(2505.12716): base와 instruct의 가중치 차이는 2% 미만이다. base를 파인튜닝하고 그 변화량을 instruct에 더하는 방식이 instruct를 직접 파인튜닝하는 것보다 나았다(Qwen3·Llama 3, 벤치마크 19개).
- **TIES 병합**(Arcee, 2406.14971): Llama-3-70B-Instruct에 SEC 공시를 CPT한 뒤 원래 instruct와 층별 가중치로 병합해 일반 능력을 상당 부분 회복했다. 소거 실험이 없어 근거는 약하다.

**더 이른 체크포인트가 더 잘 배운다** (2025~2026년에 모이는 흐름)
- **Off-Policy Merging Beats On-Policy Self-Distillation** (CMU, 2610.05872): 새 데이터의 SFT를 사후학습 모델이 아니라 **더 이른 체크포인트(donor)**에서 하고, 그 변화량을 λ배로 줄여 사후학습 모델에 더한다("grafting"). 가상 사실 1,125개 주입에서 거의 완벽히 주입하면서 기존 능력도 지켰다. OLMo-3에서는 사전학습이 끝나기 전 체크포인트가 가장 좋은 donor였다. 자세한 내용은 [지난 주간 논문](../weekly-papers-2026-10-05/)에 있다.
- **Overtrained Language Models Are Harder to Fine-Tune** (2503.19206): 3T 토큰까지 학습한 OLMo-1B가 2.3T 체크포인트보다 지시 튜닝 후 2% 넘게 나빴다. 오래 학습할수록 파라미터가 민감해지기 때문이다.
- Learning Dynamics(2505.07796)도 덜 감쇠된 체크포인트가 도메인 손실을 더 낮춘다며, 모델 공개자들에게 **감쇠 전 체크포인트도 공개하라**고 권했다.
- 한계: 대부분 작은 모델 결과이고, 중간 체크포인트를 공개하는 회사가 드물다.

### 3.3 LoRA냐 전체 파인튜닝이냐

**LoRA Learns Less and Forgets Less** (Biderman et al. — Columbia·Databricks, 2405.09673)
- Llama-2-7B로 코드·수학 CPT와 지시 튜닝을 비교했다.
- **CPT에서는 LoRA가 확실히 뒤처졌다.** 코드 CPT에서 rank 256 LoRA가 20B 토큰에서 낸 HumanEval 0.224는 전체 파인튜닝이 4B 토큰에서 낸 수준(0.218)이었다. 수학 CPT GSM8K는 LoRA 0.203 대 전체 0.293이었다.
- **지시 튜닝에서는 차이가 작았다.** 수학 IFT 0.634 대 0.642.
- LoRA는 덜 잊었다. 같은 학습량에서 드롭아웃이나 weight decay보다 망각이 적었다.
- 전체 파인튜닝 업데이트의 rank는 흔히 쓰는 LoRA rank의 10~100배였다.
- ChipNeMo도 LoRA DAPT가 전체 파라미터 DAPT보다 "상당한 정확도 격차"를 보였다고 보고했다.

**LoRA Without Regret** (Thinking Machines Lab 블로그, 2025-09)
- 학습할 정보량이 LoRA 용량보다 작으면 LoRA ≈ 전체 파인튜닝이다. 정보량이 많은 CPT에서 뒤처지는 이유가 이것이다.
- 실전 설정: **MLP 층에 적용**(attention만은 부족), 최적 학습률은 전체 파인튜닝의 **약 10배**, 큰 배치에 약하다. RL은 rank 1로도 충분했다.

**예외: 약한 이동.** 학습 마감 이후 웹 데이터로 이어 학습한 연구(2609.23916)에서는 rank 128 이상 LoRA가 전체 CPT와 맞먹었다. 새 데이터가 기존 분포와 많이 겹쳐서 넣을 정보량이 적기 때문으로 보인다.

**LoRA도 잊는다.** 망각은 학습 파라미터 수와 스텝 수의 거듭제곱 법칙을 따랐고 조기 종료로 피할 수 없었다(2401.05605). LoRA는 원래 없던 큰 특이벡터("intruder dimensions")를 만들고, 이것이 망각의 대부분을 설명했다(2410.21228).

**정리.** 지식을 넣는 CPT는 **전체 파라미터 학습**. 지시·형식·과제 튜닝, 그리고 보존이 가장 중요한 경우는 **LoRA**(모든 층, rank 128~256, α=2r, 학습률은 전체 파인튜닝의 약 10배).

### 3.4 토크나이저: 품질이 아니라 효율의 문제

- **ChipNeMo**: 칩 설계 용어 약 9K 토큰을 추가하고 새 임베딩을 기존 하위 토큰 임베딩의 평균으로 초기화했다. 토큰 효율은 1.6~3.3% 좋아졌고, **품질 차이는 무시할 만했다.**
- **Getting the most out of your tokenizer** (Meta, 2402.01035): 사전학습된 코드 LLM의 토크나이저를 통째로 바꾸면 5B 토큰 시점에는 성능이 떨어지고, 약 **50B 토큰**쯤 되어야 회복하거나 역전했다. 50B 토큰 이상 학습할 때만 바꾸라는 권고다.
- **AdaptiVocab** (2503.19693): 도메인 n-gram 토큰으로 일부 범용 토큰을 대체해 품질 손실 없이 토큰 수를 25% 넘게 줄였다.
- **한국어처럼 토큰화 효율이 낮은 언어**는 어휘 확장의 이득이 크다. EEVE-Korean(2402.14714)은 SOLAR-10.7B에 한국어 어휘를 추가하고 단계별로 파라미터를 고정해 약 2B 토큰만으로 좋은 한국어 성능을 냈다. 다만 리더보드 기반 근거다.

**정리.** 도메인 용어 추가로 **정확도가 오른다는 근거는 거의 없다.** 이득은 속도, 비용, 유효 컨텍스트 길이다. 토큰이 많이 쪼개지는 언어나 도메인에서 추론 비용이 중요할 때 고려한다.

---

## 4. 사후학습 레버

### 4.1 SFT는 지식이 아니라 사용법을 가르친다

**Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?** (Gekhman et al. — Technion·Google, 2405.05904)
- SFT 예제를 모델이 이미 아는 것(HighlyKnown, MaybeKnown, WeaklyKnown)과 모르는 것(Unknown)으로 나눴다.
- 모르는 예제는 **훨씬 늦게** 학습됐다. 그리고 결국 학습되고 나면 **이미 알던 사실에 대한 환각이 선형으로 늘었다.**
- 전체 데이터로 학습하면 조기 종료 시 43.0%, 수렴 시 38.8%였다. MaybeKnown만 학습하면 43.6%로 가장 좋았다.
- 모르는 예제의 정답을 "모르겠다"로 바꾸면 수렴 후에도 답한 질문의 정확도가 61.8%로 유지됐다.

**Why Fine-Tuning Encourages Hallucinations and How to Fix It** (Kaplan, Gekhman et al., 2604.15574)
- SFT 손실에 **원래 자기 모델 출력 분포로의 KL 항**을 더하자 기존 사실 손상이 약 15점에서 약 2점으로 줄었다. 새 사실 학습량은 비슷했다.
- attention 층만 학습하면 과제는 배우지만 새 사실은 배우지 못했다. FFN만 학습하면 사실은 배우지만 일반 SFT만큼 잊었다.
- 새 엔티티가 기존 엔티티와 표현이 많이 겹칠수록 많이 잊었다. 사람 이름 같은 가짜 엔티티는 크게 잊게 했고, UUID 같은 엔티티는 거의 그러지 않았다.

**실전 함의.**
- 새 사실은 **CPT나 mid-training 단계**에서 넣는다. SFT는 형식, 답변 방식, 지식 사용법에 쓴다.
- 형식과 스타일에는 1,000개 정도의 잘 만든 예제로 충분했다(LIMA, 2305.11206). 수학·코드처럼 기술이 무거운 영역은 데이터를 늘릴수록 계속 좋아졌다(2310.05492).
- SFT 데이터에 모델이 모르는 사실이 섞여 있다면 걸러내거나, "모르겠다"로 바꾸거나, 조기 종료하거나, 자기 분포로의 KL 항을 넣는다.

### 4.2 "문서를 본 나 자신"을 교사로 쓴다

문서 기반 지식 주입에서 가장 효율이 좋은 기본기는 **컨텍스트 증류**다. 문서를 프롬프트에 넣은 모델이 교사가 되고, 문서 없이 답하는 같은 모델이 학생이 된다.

**Efficient Knowledge Injection via Self-Distillation (Prompt Distillation)** (Kujanpää et al. — Aalto, 2412.14964)
- 합성 질문에 대해 교사의 토큰 분포 전체를 KL로 따라가게 한다(LoRA 학습).

| Llama-3-8B-Instruct, SQuADShifts NYT (닫힌 책) | 점수 |
|---|---|
| base | 38.2 |
| SFT | 87.5 |
| **Prompt Distillation** | **93.6** |
| RAG | 96.3 |

- SFT가 같은 성능을 내려면 **약 10배 많은 질문**이 필요했다. Tülu 3 데이터를 섞자 MMLU-Pro 하락이 0.5점 이내였다.

**SDFT의 지식 주입 실험** ([SDFT 리뷰](../sdft-self-distillation-review/), 2601.19897)
- 2025년 자연재해 위키 문서 9개(약 20만 토큰), 간접 질문 기준: CPT 7, QA-SFT 80, **SDFT 98**, RAG 100.

**Cartridges** (Stanford, 2506.06266)
- 코퍼스마다 작은 KV 캐시를 오프라인으로 학습한다. 합성 대화를 만들고 컨텍스트 증류로 학습한다. 문서 전체를 컨텍스트에 넣은 것과 같은 성능을 메모리 38.6배 절감으로 냈다.
- 같은 코퍼스에 질문이 반복되는 경우(코드베이스, 법률 문서)라면 **모델 가중치를 건드리지 않는** 실용적 대안이다.

**SEAL** (MIT, 2506.10943)
- 모델이 자기 학습 데이터("self-edit")를 만들고, 그 데이터로 업데이트한 뒤의 성능을 보상으로 RL을 돌린다. SQuAD 단일 문단에서 47.0으로 GPT-4.1 합성 데이터(46.3)와 비슷했다. 하지만 문단 2,067개 전체에서는 46.4로 GPT-4.1(49.2)에 뒤졌고, 보상 한 번에 30~45초가 들었다.
- 실무에서는 **강한 생성기로 다양하게 증강하는 쪽**이 비용 대비 낫다.

**주의: KL 방향.** grafting 논문(2610.05872)에서 온폴리시 자기증류 중 **forward KL만 새 사실을 제대로 주입했다.** reverse KL이나 순수 온폴리시 RL은 모델이 처음 보는 사실에 약하다. 처음 보는 사실에서는 롤아웃이 다 틀려 학습 신호가 없기 때문이다. 이를 보완하려고 정답을 섞어 넣는 mixed-policy RL(GRIN, 2608.25243)이 나왔다.

### 4.3 덜 잊는 비결은 "원래 분포에서 덜 멀어지는 것"

- **Retaining by Doing** (Princeton, 2510.18874): Llama-3.1-8B에서 IFEval을 학습하자 SFT는 목표 +25.2, 다른 과제 **−27.8**이었고, GRPO는 목표 +18.4, 다른 과제 **−3.4**였다. 원인은 KL 정규화가 아니라 **온폴리시 데이터 자체**였다. epoch마다 데이터를 다시 생성하는 "대략 온폴리시" SFT로도 망각이 거의 없었다.
- 그런데 2026년 10월 첫 주에 세 편이 반박했다([주간 논문 2026-10-05](../weekly-papers-2026-10-05/)).
  - 학습률을 공정하게 튜닝한 SFT와 온폴리시 증류의 차이는 작았고, 온폴리시 쪽 비용이 15~23배였다(Apple, 2610.04272).
  - 데이터를 MCMC로 "모델 자신의 말투"로 고쳐 쓰면 SFT도 OPSD만큼 덜 잊었다(Harvard, 2610.02140). 화학에서 기존 능력이 SFT 0.520, OPSD 0.568, 제안법 0.586이었다(base 0.597).
  - 업데이트를 줄여 병합하는 grafting이 OPSD를 이겼다(CMU, 2610.05872).

**실전 순서 제안 (싼 것부터).**
1. 학습률을 제대로 튜닝한 SFT + 일반 데이터 리플레이
2. 데이터를 대략 온폴리시로 재생성 (epoch마다 재생성, 모델 말투로 재작성)
3. 업데이트를 λ배로 줄여 병합 (WiSE-FT, grafting)
4. 그래도 부족하면 온폴리시 자기증류나 RL

### 4.4 도메인 RL: 지식을 더하나, 끌어내기만 하나

**논쟁의 지형.**
- **Does RL Really Incentivize Reasoning Capacity?** (칭화대, 2504.13837): RLVR은 pass@1을 올리지만, k가 커지면 base 모델이 따라잡고 앞섰다. RL이 내는 추론 경로는 이미 base 분포 안에 있었다. 즉 **끌어내기**다.
- **ProRL** (NVIDIA, 2505.24864): KL 제어와 참조 정책 리셋으로 RL을 오래 돌리면 base가 몇 번 시도해도 못 풀던 문제를 풀었다.
- **On the Interplay of Pre-Training, Mid-Training, and RL** (CMU, 2512.07783): 통제된 합성 과제에서 RL이 "진짜" 능력을 넓히는 조건은 두 가지였다. **사전학습이 여지를 남겼을 때**, 그리고 **RL 데이터가 역량의 경계에 있을 때**(어렵지만 아예 못 풀지는 않는 문제)다. 같은 연산량이면 mid-training을 넣는 쪽이 RL만 하는 쪽보다 나았다.

**의료에서는 지식이 병목이었다.**
- **m1** (UCSC, 2504.00869): 의료 추론 성능은 사고 토큰 약 4K에서 포화했고, 더 길게 생각하게 하면 오히려 나빠졌다. 병목은 추론 길이가 아니라 **의학 지식 부족**이었다.
- **Disentangling Reasoning and Knowledge in Medical LLMs** (Stanford 등, 2505.11462): 의료 QA 벤치 11개 문항 중 복잡한 추론이 필요한 것은 **32.8%뿐**이었다.
- **HuatuoGPT-o1** (CUHK-Shenzhen, 2412.18925): 복잡한 CoT로 SFT한 뒤 RL을 하면 RL 이득이 +3.6점이었지만, 단순 CoT 위에서는 +1.1점에 그쳤다. RL은 **추론형 SFT 위에서** 효과가 커진다.

**검증 가능한 도메인은 RL만으로도 강하다.** ether0(FutureHouse, 2506.17238)는 Mistral-Small-24B에 화학 CPT 없이 실험 데이터 기반 64만 문제로 GRPO를 해서 분자 설계에서 범용·화학 특화 모델과 인간 전문가를 넘었다.

**검증이 안 되는 도메인은 루브릭 보상.** Rubrics as Rewards(Scale AI, 2507.17746)는 참조 답을 바탕으로 문항마다 7~20개 채점 항목을 만들어 보상으로 썼다. Likert 방식 LLM 판정 대비 HealthBench에서 최대 31%(상대) 향상을 보고했다. **참조 답 없이 만든 루브릭은 크게 약했다.** 전문가 참조 답이 있으면 여러 LLM 판정자의 정답 판정이 높은 일치도를 보였다는 결과도 있다(2503.23829).

**정리.** 도메인 RL은 "이미 있는 지식을 정확하게 꺼내 쓰는" 도구다. 지식이 부족하다면 RL 전에 CPT나 mid-training으로 그 도메인을 **조금이라도** 보여줘야 한다. 순서는 (CPT/mid-training) → 추론형 SFT → RL이다.

---

## 5. 평가: 이득을 정직하게 재는 법

도메인 학습 논문의 이득 주장은 평가 방식에 크게 흔들린다.

- **base와 1:1로, 프롬프트는 모델별로 따로 튜닝하고, 통계 검정을 한다.** Jeong et al.(0장)에서 이 원칙만으로 의료 모델 승률이 70% 넘게에서 12.1%로 내려갔다.
- **오염을 확인한다.** "Reasoning or Memorization?"(2507.10532)은 Qwen2.5에서 "무작위 보상으로도 RL이 수학 점수를 올린다"던 결과가 벤치 오염 때문이었음을 보였다. 문제 앞부분만 줘도 모델이 나머지와 정답을 완성했다. **문제를 잘라 주고 완성하는지 보는 테스트**가 싸고 효과적이다.
- **디코딩 설정과 시드를 고정하고 분산을 보고한다.** 이것만 바꿔도 점수가 크게 흔들렸고, 표준화해 다시 재면 RL 이득이 보고치보다 작았다(2504.07086). AIME처럼 작은 벤치에서 특히 그랬다.
- **객관식만 쓰지 않는다.** HealthBench(의사 262명이 만든 루브릭 48,562개, 2505.08775)나 MedXpertQA(2501.18362)처럼 개방형이거나 유출 대책이 있는 벤치를 함께 쓴다.
- **일반 능력 유지를 함께 보고한다.** 지시 따르기(IFEval, MT-Bench)를 꼭 포함한다. FinDAP에서 CPT로 무너진 것이 바로 이것이었다. 겉으로 보이는 망각 중 상당 부분은 지식 손실이 아니라 형식 붕괴일 수 있다는 점도 구분해야 한다(Spurious Forgetting, [저망각 서베이](../low-forgetting-finetuning-survey-2026/) 참고).
- **RAG와 비교한다.** 파라미터 주입이 RAG를 이기는지, 둘을 합치면 더해지는지를 함께 보고한다.

---

## 6. 종합 레시피

위 근거를 실무 순서로 묶었다. 근거 강도는 **강**(여러 독립 그룹에서 재현), **중**(일관되지만 주로 8B 이하), **약**(단일 연구, 초기)으로 표시했다.

**0단계: 필요성 확인** — 강
- 가장 강한 범용 base + 튜닝한 프롬프트 + RAG를 먼저 재본다. 도메인 데이터가 웹에 흔하다면 이득이 작을 가능성이 크다.

**1단계: 데이터 준비**
- 작은 코퍼스는 **다양한 형태로 100~350배 이상 증강**한다. 프롬프트 풀을 다양하게, 엔티티 조합이나 질문 초점으로 다양성을 높인다. 생성기 크기보다 다양성이 중요하다. — 강
- **합성 문서와 합성 QA를 함께**(1:1 근처). QA를 학습할 때 원문을 컨텍스트에 붙이지 않는다. — 중
- 사실 단위로 커버리지를 맞춘다. 역방향 질문도 만든다. — 중
- 대규모 원문이 있다면 품질로 걸러 **작고 좋은 부분집합을 여러 epoch**. — 중

**2단계: CPT / mid-training**
- **base 체크포인트**에서 시작한다(가능하면 덜 감쇠된 것). — 중
- **전체 파라미터** 학습. — 강
- 일반 데이터 리플레이: 약한 이동 1~5%, 소규모 합성 CPT 10%, 강한 이동 25% 이상. 원래 사전학습과 비슷한 데이터일수록 좋다. — 강
- 학습률: 큰 예산·강한 이동이면 재워밍, 작은 예산이면 낮게(수e-6~수e-5), 큰 모델일수록 낮게. 짧은 실험으로 정한다. — 중
- 초반 지표 하락(stability gap)에 조기 종료하지 않는다. 마지막 10~20%는 최고 품질 데이터로 어닐링하고, 가능하면 어닐링 여러 개를 평균한다. — 중
- 토크나이저는 효율이 목적일 때, 50B 토큰 이상일 때만. — 중

**3단계: 대화 능력 복원과 SFT**
- CPT한 base에 instruct 차이 벡터를 더하거나, CPT와 지시 튜닝을 **함께** 학습하거나, 원래 instruct와 병합한다. — 중
- SFT는 형식과 사용법에. 모르는 사실이 섞인 예제는 거르거나 "모르겠다"로 바꾼다. — 강
- 새 사실을 SFT로 넣어야 한다면 자기 분포로의 KL 항이나 컨텍스트 증류를 쓴다. — 중
- SFT/IFT에는 LoRA(모든 층, rank 128~256, 학습률 약 10배)도 충분하다. — 강

**4단계: RL (선택)**
- 정답이 검증되는 도메인은 규칙 보상 RLVR, 아니면 참조 답 기반 루브릭 보상. — 중
- 역량 경계의 문제로, 추론형 SFT 다음에. 지식 부족은 RL로 메우지 않는다. — 중

**5단계: 평가**
- base와 1:1, 모델별 프롬프트, 통계 검정, 오염 점검, 일반 능력(지시 따르기 포함) 동시 보고, RAG와 비교. — 강

---

## 7. 의견이 갈리는 지점과 열린 문제

1. **파라미터 주입이 RAG를 이길 수 있나.** 2023년 Ovadia et al.은 "RAG가 압도한다"고 했다. 2025~2026년 Prompt Distillation, Active Reading, SDFT, Synthetic Mixed Training은 "잘 하면 RAG에 근접하거나 넘는다"고 한다. 하지만 후자는 대부분 문서 수천 개 이하, 8B 이하, 짧은 답 QA에서 나온 결과다. 수백만 문서나 자주 바뀌는 지식에서는 여전히 RAG가 기본값이다.
2. **모델이 짠 전략 대 고정 프롬프트.** Active Reading은 모델이 만든 전략이 낫다고 했고, SPA는 잘 다듬은 고정 프롬프트가 낫다고 했다. 서로 상대 방법을 재현한 수치가 엇갈린다.
3. **학습률은 높게 재워밍인가, 낮게 유지인가.** 분포 이동과 예산으로 대부분 설명되지만, 7B 이상에서 다운스트림까지 본 통합 연구는 없다.
4. **LoRA는 CPT에 쓸 수 있나.** 코드·수학·칩 설계에서는 안 됐고, 약한 이동(새 웹 데이터)에서는 됐다. 도메인 코퍼스의 "정보량"을 미리 재는 방법이 없다.
5. **리플레이는 망각 방지인가 학습 보조인가.** 목표 성능을 올린다는 보고(Kotha & Liang, Active Reading)와 거의 영향 없다는 보고(Apple)가 공존한다. 메커니즘은 미해결이다.
6. **혼합 법칙이 서로 맞지 않는다.** D-CPT의 최적 도메인 비율은 0.73~0.92, CMR은 0.30~0.48이다. 정의와 허용폭이 달라서다. 둘 다 손실 기준, 4B 이하다.
7. **도메인 CPT 자체가 필요한가.** ether0는 CPT 없이 RL만으로 최고 성능을 냈고, 의료 CPT 모델 대부분은 base를 못 이겼다. 반면 FinDAP과 Interplay 논문은 CPT/mid-training의 지식 공급이 필요하다고 본다. **"base가 이미 아는 도메인인가"**가 결정 변수로 보이지만, 이를 미리 재는 표준 방법이 없다.
8. **규모.** 대부분의 근거가 8B 이하다. 70B 이상에서도 순위가 유지되는지는 확인되지 않았다.

---

## 8. 관련 블로그 포스트

- [도메인 특화 RAG LLM 구축 가이드](../domain-optimized-llm-for-rag/) — 검색으로 붙이는 쪽의 전체 지도
- [RAFT 리뷰](../raft-review/) — RAG용 도메인 파인튜닝
- [RAG 학습용 합성 데이터 서베이](../rag-synthetic-data-survey/) — 문서만으로 학습 데이터 만들기
- [저망각 파인튜닝 서베이 (2026)](../low-forgetting-finetuning-survey-2026/) — 망각의 메커니즘과 네 가지 레버
- [SDFT 리뷰](../sdft-self-distillation-review/) — 4.2의 자기증류 지식 주입
- [주간 논문 2026-10-05](../weekly-papers-2026-10-05/) — 4.3의 2026년 10월 반박 논문들

---

## 9. 참고 자료

† 표시는 초록만 확인한 논문이다. 나머지는 본문 전체 또는 핵심 절을 읽었다.

### 왜 어려운가·필요성
- Physics of Language Models 3.1 — [2309.14316](https://arxiv.org/abs/2309.14316) · 3.3 † [2404.05405](https://arxiv.org/abs/2404.05405)
- Instruction-tuned LMs are Better Knowledge Learners — [2402.12847](https://arxiv.org/abs/2402.12847)
- The Reversal Curse † — [2309.12288](https://arxiv.org/abs/2309.12288)
- Efficient Continual Pre-training by Mitigating the Stability Gap — [2406.14833](https://arxiv.org/abs/2406.14833)
- Medical Adaptation of LLMs and VLMs: Are We Making Progress? — [2411.04118](https://arxiv.org/abs/2411.04118) · 확장판 [2411.08870](https://arxiv.org/abs/2411.08870)
- BloombergGPT † — [2303.17564](https://arxiv.org/abs/2303.17564) · GPT-4 금융 비교 † [2305.05862](https://arxiv.org/abs/2305.05862)
- Calderon et al. (recall이 병목) † — [2602.14080](https://arxiv.org/abs/2602.14080)

### 데이터 증강
- Synthetic Continued Pretraining (EntiGraph) — [2409.07431](https://arxiv.org/abs/2409.07431)
- Learning Facts at Scale with Active Reading — [2508.09494](https://arxiv.org/abs/2508.09494)
- SPA — [2603.22213](https://arxiv.org/abs/2603.22213)
- Synthetic Mixed Training — [2603.23562](https://arxiv.org/abs/2603.23562)
- Fine-Tuning or Retrieval? — [2312.05934](https://arxiv.org/abs/2312.05934)
- Injecting New Knowledge into LLMs via SFT † — [2404.00213](https://arxiv.org/abs/2404.00213)
- New News (Sys2-FT) † — [2505.01812](https://arxiv.org/abs/2505.01812)
- Demystifying Synthetic Data in LLM Pre-training † — [2510.01631](https://arxiv.org/abs/2510.01631)
- Rephrasing the Web (WRAP) † — [2401.16380](https://arxiv.org/abs/2401.16380)
- How new data permeates LLM knowledge — [2504.09522](https://arxiv.org/abs/2504.09522)
- AdaptLLM (Reading Comprehension) — [2309.09530](https://arxiv.org/abs/2309.09530)
- Instruction Pre-Training — [2406.14491](https://arxiv.org/abs/2406.14491)
- Synthesize-on-Graph † — [2505.00979](https://arxiv.org/abs/2505.00979)

### 리플레이·혼합·선별
- Simple and Scalable Strategies to Continually Pre-train LLMs — [2403.08763](https://arxiv.org/abs/2403.08763)
- Continual Pre-Training: How to (re)warm your model? — [2308.04014](https://arxiv.org/abs/2308.04014)
- Scaling Laws for Forgetting with Pretraining Data Injection — [2502.06042](https://arxiv.org/abs/2502.06042)
- Replaying pre-training data improves fine-tuning — [2603.04964](https://arxiv.org/abs/2603.04964)
- Self-generated replay † — [2605.26097](https://arxiv.org/abs/2605.26097)
- D-CPT Law — [2406.01375](https://arxiv.org/abs/2406.01375)
- CMR Scaling Law — [2407.17467](https://arxiv.org/abs/2407.17467)
- The Finetuner's Fallacy — [2603.16177](https://arxiv.org/abs/2603.16177)
- Knowledge Infusion Scaling Law — [2509.19371](https://arxiv.org/abs/2509.19371)
- Time-Incremental Continued Pretraining — [2609.23916](https://arxiv.org/abs/2609.23916)
- DSIR † — [2302.03169](https://arxiv.org/abs/2302.03169) · LESS — [2402.04333](https://arxiv.org/abs/2402.04333)
- Data Source Utility via Scaling Laws † — [2507.22250](https://arxiv.org/abs/2507.22250)
- French medical domain filters † — [2606.22079](https://arxiv.org/abs/2606.22079)

### 학습 레시피
- Reuse, Don't Retrain (NVIDIA) — [2407.07263](https://arxiv.org/abs/2407.07263)
- Learning Dynamics in Continual Pre-Training — [2505.07796](https://arxiv.org/abs/2505.07796)
- Does your data spark joy? † — [2406.03476](https://arxiv.org/abs/2406.03476)
- The Llama 3 Herd of Models — [2407.21783](https://arxiv.org/abs/2407.21783)
- 2 OLMo 2 Furious — [2501.00656](https://arxiv.org/abs/2501.00656)
- FinDAP — [2501.04961](https://arxiv.org/abs/2501.04961)
- Balancing CPT and Instruction Fine-Tuning — [2410.10739](https://arxiv.org/abs/2410.10739)
- Chat Vector † — [2310.04799](https://arxiv.org/abs/2310.04799)
- Shadow-FT † — [2505.12716](https://arxiv.org/abs/2505.12716)
- Arcee Llama3-70B CPT + Merging — [2406.14971](https://arxiv.org/abs/2406.14971)
- Off-Policy Merging Beats On-Policy Self-Distillation — [2610.05872](https://arxiv.org/abs/2610.05872)
- Overtrained Language Models Are Harder to Fine-Tune † — [2503.19206](https://arxiv.org/abs/2503.19206)
- LoRA Learns Less and Forgets Less — [2405.09673](https://arxiv.org/abs/2405.09673)
- LoRA vs Full Fine-tuning: An Illusion of Equivalence † — [2410.21228](https://arxiv.org/abs/2410.21228)
- Scaling Laws for Forgetting When Fine-Tuning † — [2401.05605](https://arxiv.org/abs/2401.05605)
- LoRA Without Regret (Thinking Machines Lab 블로그, 2025-09) — [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/)
- ChipNeMo — [2311.00176](https://arxiv.org/abs/2311.00176)
- Getting the most out of your tokenizer — [2402.01035](https://arxiv.org/abs/2402.01035)
- AdaptiVocab † — [2503.19693](https://arxiv.org/abs/2503.19693)
- EEVE-Korean † — [2402.14714](https://arxiv.org/abs/2402.14714)
- MEDITRON-70B — [2311.16079](https://arxiv.org/abs/2311.16079)
- SaulLM-7B — [2403.03883](https://arxiv.org/abs/2403.03883)

### 사후학습
- Does Fine-Tuning on New Knowledge Encourage Hallucinations? — [2405.05904](https://arxiv.org/abs/2405.05904)
- Why Fine-Tuning Encourages Hallucinations and How to Fix It — [2604.15574](https://arxiv.org/abs/2604.15574)
- LIMA † — [2305.11206](https://arxiv.org/abs/2305.11206) · SFT 데이터 구성 연구 † [2310.05492](https://arxiv.org/abs/2310.05492)
- Prompt Distillation — [2412.14964](https://arxiv.org/abs/2412.14964)
- SDFT — [2601.19897](https://arxiv.org/abs/2601.19897)
- Cartridges † — [2506.06266](https://arxiv.org/abs/2506.06266)
- SEAL — [2506.10943](https://arxiv.org/abs/2506.10943)
- GRIN † — [2608.25243](https://arxiv.org/abs/2608.25243)
- Retaining by Doing — [2510.18874](https://arxiv.org/abs/2510.18874)
- Finetuning with Sampling — [2610.02140](https://arxiv.org/abs/2610.02140) · Rethinking Self-Distillation for Multi-Teacher Merging — [2610.04272](https://arxiv.org/abs/2610.04272)
- Does RL Really Incentivize Reasoning Capacity? † — [2504.13837](https://arxiv.org/abs/2504.13837) · ProRL † [2505.24864](https://arxiv.org/abs/2505.24864)
- On the Interplay of Pre-Training, Mid-Training, and RL † — [2512.07783](https://arxiv.org/abs/2512.07783)
- m1 † — [2504.00869](https://arxiv.org/abs/2504.00869) · Disentangling Reasoning and Knowledge † [2505.11462](https://arxiv.org/abs/2505.11462)
- HuatuoGPT-o1 — [2412.18925](https://arxiv.org/abs/2412.18925)
- ether0 † — [2506.17238](https://arxiv.org/abs/2506.17238)
- Rubrics as Rewards — [2507.17746](https://arxiv.org/abs/2507.17746) · Crossing the Reward Bridge † [2503.23829](https://arxiv.org/abs/2503.23829)

### 평가
- Reasoning or Memorization? (RL 오염) † — [2507.10532](https://arxiv.org/abs/2507.10532)
- A Sober Look at Progress in LM Reasoning † — [2504.07086](https://arxiv.org/abs/2504.07086)
- HealthBench † — [2505.08775](https://arxiv.org/abs/2505.08775) · MedXpertQA † [2501.18362](https://arxiv.org/abs/2501.18362)
