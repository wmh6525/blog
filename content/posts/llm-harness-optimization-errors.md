---
title: "[리서치] LLM Harness 최적화와 오류 — 평가·에이전트 하네스 논문 30+ 종합"
date: 2026-06-26
tags: ["LLM", "Evaluation", "Harness", "Benchmark", "Agent", "리서치"]
categories: ["ML/AI"]
summary: "LLM harness는 겉으로 단순한 wrapper 같지만, prompt 한 글자에 51%→94% 점수 변동, few-shot 순서에 따른 극단 편차, contamination, judge model 편향 등 결정적 오류가 즐비하다. lm-eval-harness / HELM / Chatbot Arena 계열 평가 하네스와 LangChain/DSPy/Agent scaffold 실행 하네스의 오류 모드와 최적화 기법을 논문 근거로 정리."
math: true
toc: true
draft: false
---

## 들어가며 — Harness가 뭐고 왜 문제인가?

**Harness** = LLM을 "돌리기 위한 감싸는 코드". 두 종류:

- **A. 평가 하네스** (Evaluation harness): `lm-evaluation-harness` (EleutherAI), HELM (Stanford), OpenCompass, BIG-Bench, Chatbot Arena — 모델을 벤치마크에 붙여 점수 내는 코드
- **B. 실행 하네스** (Agent/tool-use harness): LangChain, LlamaIndex, DSPy, ReAct scaffold, MCP — 모델이 도구/사고를 사용하게 감싸는 코드

두 하네스 모두 **"모델보다 harness가 결과를 더 좌우한다"** — 최근 3년간 반복해서 증명됐다. 이 글은:

1. 어떤 오류가 존재하는가 (~20편)
2. 어떻게 최적화하는가 (~10편)
3. 실전 권고

를 논문 근거로 정리한다.

---

## Part A. 평가 하네스 오류 (Eval Harness Errors)

### A.1 Prompt Sensitivity — 한 글자에 점수가 요동친다

#### Sclar et al. 2023 — "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design" (FormatSpread)
- **arXiv 2310.11324, ICLR 2024**
- **핵심 발견**: 의미상 동일한 프롬프트 포맷만 바꿔서 **Llama-2-13B에서 최대 76 accuracy point 스윙**
- 좋은 포맷은 모델 간 상관 약함 → transfer 불가능
- 방법: **FormatSpread** — 컴퓨트 예산 내에서 의미 동등한 포맷 샘플링

#### Mizrahi et al. 2024 — "State of What Art? A Call for Multi-Prompt LLM Evaluation"
- **arXiv 2401.00595, TACL 2024**
- **핵심 발견**: **6.5M 평가 instance**, 20 LLM × 39 task × >175 검증된 paraphrase
- Template 회전 시 랭킹이 뒤집힘
- **권고**: 최소 5개 프롬프트 변형 평균 → 순위 안정화

#### Alzahrani et al. 2024 — "When Benchmarks are Targets"
- **arXiv 2402.01781, ACL 2024**
- **MMLU 미세 perturbation으로 리더보드 순위 ±8 위치 변동**
- Yi-6B: 3위 → 9위. **Yi-6B accuracy shift −42.78%** (choice shuffling)
- Llama-2-7B: position A일 때 +24.55% vs position D일 때 −18.44%
- Kendall τ vs 원본 순위 = 0.564 (약함)

#### Voronov et al. — "Mind Your Format"
- Few-shot 예제의 **구분자 문자** 하나 (`\n` vs `\n\n` vs `###`)가 점수에 큰 영향

**시사점**: 단일 프롬프트 벤치마크 결과는 **거의 무의미**. 최소 다중 프롬프트 평균 필요.

---

### A.2 Few-shot Ordering Effects

#### Lu et al. 2022 — "Fantastically Ordered Prompts and Where to Find Them"
- **ACL 2022, arXiv 2104.08786**
- **핵심 발견**: 순열에 따라 "near SOTA ~ random guess" 스윙 (abstract). Body table 기준 GPT2-XL SST-2에서 **88.7% → 51.6%** 폭
- GPT-3, GPT-2 모두 영향받음
- 좋은 순서 자동 발견: **entropy 기반 selection** → 11개 분류 태스크에서 **+13% relative** 개선

#### Zhao et al. 2021 — "Calibrate Before Use"
- **arXiv 2102.09690**
- **Recency bias** (마지막 예제 label로 편향), **Majority bias** (다수 label로 편향), **Common token bias**
- Contextual calibration으로 대응

**시사점**: Few-shot 순서는 **무작위화 + 평균** 하지 않으면 벤치마크가 chance보다도 신뢰성이 낮을 수 있음.

---

### A.3 Data Contamination — 벤치마크가 이미 학습 데이터에 있다

#### Sainz et al. 2023 — "NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for Each Benchmark"
- **arXiv 2310.18018**
- **핵심 발견**: 대부분 유명 벤치마크가 pretraining corpus에 존재
- 4대 오염 유형: guideline / raw text / annotated data / labels

#### Golchin & Surdeanu 2023 — "Time Travel in LLMs: Tracing Data Contamination in Large Language Models"
- **arXiv 2308.08493, ICLR 2024**
- Guided prompting으로 GPT-4의 벤치마크 오염 정량 측정
- **AG News, WNLI, XSum**에서 GPT-4 오염 감지

#### Deng et al. 2024 — "Investigating Data Contamination in Modern Benchmarks for LLMs"
- **arXiv 2311.09783**
- MMLU, HellaSwag, ARC 모두 GPT-4 학습 데이터에 존재 가능성

#### Zhou et al. 2023 — "Don't Make Your LLM an Evaluation Benchmark Cheater"
- Contamination이 성능을 인위적으로 부풀리는 정량 데이터

**시사점**: 벤치마크 점수 = "학습 데이터에 얼마나 있었나"를 재는 것일 수 있음. **rolling / private / synthetic 벤치마크로 이동 필요**.

---

### A.4 Log-Likelihood vs Generation — 같은 태스크 다른 점수

- `lm-eval-harness`에서 **loglikelihood-based** 채점 (`.loglikelihood`) vs **generative** 채점 (`.generate_until`)이 같은 태스크에 대해 다른 점수 산출
- **MMLU 대표 케이스**: log-likelihood 채점(각 선택지 확률 비교) vs 실제 생성 후 파싱은 상당한 gap
- Chain-of-thought을 요구하는 태스크에서 generation 채점이 필수

#### Wang et al. 2024 — "'My Answer is C': First-Token Probabilities Do Not Match Text Answers"
- **arXiv 2402.14499, COLM 2024**
- **OpinionQA (Llama-2-7B-Chat)에서 first-token vs 실제 텍스트 답변이 66.2% mismatch**
- MMLU **51.4% mismatch**
- 강제 text-consistent 채점 시 accuracy **41.0% → 34.9% 하락**

#### Zheng et al. 2024 — "LLMs Are Not Robust Multiple Choice Selectors"
- **arXiv 2309.03882, ICLR 2024**
- Token bias > position bias across **20 LLMs on 3 benchmarks**
- 심볼 "A"에 확률 mass가 체계적으로 몰림
- 방법: **PriDe** 디바이싱

#### Fourrier — Open LLM Leaderboard MMLU 참사 (HF blog, 2023.06)
- **LLaMA-65B MMLU 원 논문 (Hendrycks): 0.636**
- **HELM: 0.637**
- **Eleuther Harness (Jan 2023): 0.488**
- → **동일 데이터에서 14~15 abs 점수 gap / 30% 상대 gap**, 순위도 다름
- 원인: closed-set softmax vs first-token 생성 vs full-answer loglikelihood + length normalization
- 프리픽스·공백 처리 차이까지

---

### A.5 Reproducibility — 하네스 버전만 바꿔도 다른 점수

#### Biderman et al. 2024 — "Lessons from the Trenches on Reproducible Evaluation of Language Models"
- **arXiv 2405.14782**
- **핵심 발견**: 같은 모델, 같은 벤치마크가 **하네스 코드 버전 / 프롬프트 템플릿 / 답변 파싱**에 따라 다른 점수
- 유명 사례: `lm-eval-harness` v0.3 → v0.4 사이 MMLU 채점 로직 변경으로 논문 스코어 재현 불가
- **권고**: 하네스 커밋 해시 + 프롬프트 template 파일 명시

**HuggingFace Open LLM Leaderboard v1 → v2 (2024.06)**:
- MMLU, ARC, HellaSwag, TruthfulQA, Winogrande, GSM8K → **폐기**
- 대체: MMLU-Pro, GPQA, MuSR, MATH-Level5, IFEval, BBH
- 이유: **contamination + saturation**

---

### A.6 Answer Extraction — 정규식이 점수를 결정한다

- GSM8K, MATH 등에서 **최종 답을 정규식으로 추출**
- "The answer is 42" vs "So, 42." vs "$$42$$" — 파싱 실패 시 오답 처리
- 실제 모델은 맞았는데 harness가 파싱 실패로 오답 처리하는 사례 다수
- 대응: **structured output**, JSON mode, guided decoding

#### Fourrier — Math-Verify 참사 (HuggingFace blog, 2025.02) — 결정적 사례
- **정규식 추출기를 Math-Verify parser로 교체 = Open LLM Leaderboard 평균 +4.66점**
- 각 모델이 **~61개 문제 추가로 정답 처리**
- 일부 서브셋에서 **~90점 증가**
- **Qwen 점수 2배 이상, DeepSeek 거의 3배** — 이 모델들의 `\boxed{}` 출력이 이전 regex로 파싱 불가였음
- 근본 원인: **3줄짜리 수정**

#### `lm-eval-harness` GSM8K CoT의 정규식 함정
- 기본 regex: `The answer is (\-?[0-9\.\,]+).` — `24.0`, trailing period, `\boxed{}` 미매치
- `strict-match` vs `flexible-extract` variant로 다른 점수 생성

#### Zhang et al. 2024 — GSM1k
- **arXiv 2405.00332, NeurIPS 2024 D&B**
- 매칭된 fresh sample에서 선도 모델들이 GSM8k 대비 **최대 13 pp 하락** → **추출 오류 + contamination이 겹친 confound 확인**

---

### A.6.1 Batching / 비결정성 — 같은 seed에서도 답이 달라진다

#### Thinking Machines Lab — "Defeating Nondeterminism in LLM Inference" (2025.09.10)
- **Qwen3-235B temperature=0으로 1000번 completion → 80개 unique 출력** (최다 등장 78회)
- Token 103부터 divergence 시작
- **근본 원인은 "concurrency + FP associativity" myth가 아님** — 커널이 **batch-invariant하지 않은 것**이 원인
- 배치 크기에 따라 reduction 순서가 달라져 결과 변동
- 해결: batch-invariant 커널 (matmul fixed tile, FlashAttention fixed split, RMSNorm fixed order)
- 비용: ~1.6× 슬로우다운 (최적화 후)
- 라이브러리: `thinking-machines-lab/batch-invariant-ops` (vLLM 업스트림)

#### "Non-Determinism of 'Deterministic' LLM Settings" (arXiv 2408.04667)
- Greedy temp=0도 provider 간 재현 불가

#### "Numerical Sources of Nondeterminism" (arXiv 2506.09501)
- **FP32는 안정, BF16은 유의미한 변동** (BF16이 서빙 기본값인데도)

---

### A.7 Judge Model Bias (LLM-as-Judge)

#### Zheng et al. 2023 — "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
- **NeurIPS 2023, arXiv 2306.05685**
- **4가지 판단 편향**:
  1. **Position bias**: 첫 번째 응답 선호
  2. **Verbosity bias**: 긴 응답 선호
  3. **Self-enhancement bias**: 자기 자신 (같은 모델 계열) 선호
  4. **Limited reasoning**: 복잡한 수학/추론 판단 못함
- 대응: **order swapping**, reference-guided, few-shot judge

#### Wang et al. 2023 — "Large Language Models are not Fair Evaluators"
- **arXiv 2305.17926**
- Position bias가 극단적: 첫 번째 위치에 뭘 두느냐로 결과 뒤집힘
- 대응: **Multiple Evidence Calibration + Balanced Position Calibration**

#### Chen et al. 2024 — "Humans or LLMs as the Judge? A Study on Judgement Bias"
- LLM judge가 factual error에 관대함
- Human judge 대비 40%까지 다른 결론

**시사점**: LLM-as-judge는 편향을 명시적으로 제어하지 않으면 **거의 랜덤에 가까울 수 있음**.

---

### A.8 Lost in the Middle

#### Liu et al. 2023 — "Lost in the Middle: How Language Models Use Long Contexts"
- **arXiv 2307.03172, TACL 2024**
- **핵심 발견**: relevant 정보가 컨텍스트 **중간**에 있으면 성능 급락 (U-curve)
- 시작/끝은 잘 찾지만 중간 30-70%는 놓침
- GPT-3.5, Claude, MPT-30B 모두 확인
- **시사점**: RAG 순서, few-shot 배치가 결과 좌우

---

### A.9 Statistical Significance — 오차 막대가 없다

#### Miller 2024 — "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations"
- **arXiv 2411.00640**
- 대부분 벤치마크가 **신뢰구간 없이 순위** 발표
- 실제 계산해보면 1-2%p 차이는 유의하지 않은 경우 다수
- **권고**: Bootstrap CI, cluster-based CI 강제

#### Perlitz et al. 2024 — "Efficient Benchmarking (of Language Models)"
- **arXiv 2308.11696**
- HELM의 **>80% 벤치마크 예산**을 삭감해도 순위 유지 가능
- 벤치마크 crown 계산: 어떤 하위셋이 전체 순위를 대표하는가

---

### A.10 Chain-of-Thought 함정

- CoT가 항상 도움되지 않음 (일부 태스크에선 하락)
- **Faithful CoT**: CoT reasoning이 실제 결정 이유가 아닐 수 있음 (Turpin et al. 2023 "Language Models Don't Always Say What They Think")
- 특히 few-shot에 편향 힌트를 심으면 모델은 그 방향으로 결정하지만 CoT엔 노출 안 함

---

### A.11 Multi-turn / 장기 컨텍스트 평가

- **Needle-in-Haystack** 인기지만 한계: 단일 사실 검색 능력만 잼
- **RULER** (arXiv 2404.06654): needle에 대응 — multi-hop, aggregation
- **LongBench, LongBench v2, LOOGLE, InfiniteBench** — 장기 컨텍스트 다차원 평가

---

### A.12 벤치마크 포화 (Saturation)

- MMLU: 2020 → 2024 사이 baseline 25% → **>90% 도달** (GPT-4o, Claude 3.5)
- **MMLU-Pro** (Wang et al., arXiv 2406.01574, NeurIPS 2024)
  - 10 선택지로 확장 → **raw accuracy 16-33 pp 하락** (GPT-4o 88.7→72.6, Claude 3 Sonnet 81.5→55.1, Llama-3-70B 82.0→56.2)
  - **Prompt variance 절반 (~4-5% → ~2%)**
  - CoT gain이 양의 방향으로 뒤집힘 (+19.1 pp GPT-4o)
- **GPQA** (Rein et al., arXiv 2311.12022) — 448개 PhD-authored, PhD 전문가 65% (retro-오류 제거 후 74%), 웹 허용 non-expert 34%, GPT-4 39%
- **MMLU-Redux** (Gema et al., arXiv 2406.04127) — 5,700개 재-어노테이션. **전체 오류율 6.49%, Virology 서브셋 57%**
- **BIG-Bench Hard, MATH-Level 5, Humanity's Last Exam** (2025) — 어려운 방향으로

### A.13 Chain-of-Thought 함정 (더 깊이)

#### Sprague et al. 2025 — "To CoT or not to CoT?"
- **arXiv 2409.12183, ICLR 2025**
- 100+ CoT 논문 메타분석 + 14 모델 × 20 dataset
- **CoT gain은 "="가 들어간 태스크(수학·심볼)에 집중**
- Non-math MMLU에선 flat

#### Chen et al. 2025 — "Reasoning Models Don't Always Say What They Think" (Anthropic)
- **arXiv 2505.05410 (2025.05)**
- **CoT 충실도(faithfulness)**:
  - Claude 3.7 Sonnet **25%**
  - DeepSeek R1 **39%**
- 언어화 비율 <20% 다수
- Outcome-based RL 초기 +63% MMLU, +41% GPQA relative → 이후 평체
- **Reward hacking exploit 성공률 >99%, 그런데 CoT에서 언어화된 비율은 6개 환경 중 5개에서 <2%**

#### Stechly et al. 2024 — "Chain of Thoughtlessness"
- **arXiv 2405.04776, NeurIPS 2024**
- Blocksworld 문제 크기가 few-shot 예제 초과 시 CoT 붕괴
- Prompt-specific formulation만 도움

#### Liu et al. 2024 — "Mind Your Step"
- **arXiv 2410.21333**
- 6개 인지심리학 태스크 중 3개에서 CoT 성능 하락
- 일부: **o1-preview vs GPT-4o에서 −36.3 pp**

#### Lanham et al. 2023 (Anthropic) — CoT faithfulness는 스케일 클수록 **감소** (inverse scaling)
- **arXiv 2307.13702**

#### Turpin et al. 2023 — "Language Models Don't Always Say What They Think"
- **arXiv 2305.04388, NeurIPS 2023**
- Few-shot에 bias 심으면 정확도 **최대 −36%** (13 BBH task, GPT-3.5 / Claude 1.0)
- CoT는 bias를 절대 언어화 안 함

---

## Part B. Agent/실행 하네스 오류

### B.1 Reward Hacking / Spec Gaming

#### Sakana AI CUDA Engineer 스캔들 (2025.02)
- 자체 평가 harness가 **메모리 버그 이용해 correctness 우회**
- "10-100배 가속" → 실제 **3배 느림** (24시간 만에 폭로)
- 교훈: **eval harness = agent가 gaming할 대상**

#### MLR-Bench 발견 (NeurIPS 2025)
- **arXiv 2505.19955**
- Agent 실험 결과의 **~80%가 fabricated or invalidated**
- Agent가 "결과를 만들어냈다고 보고"하지만 실제 실행 안 함

---

### B.2 Tool Call 실패

- **Malformed args** — 스키마 어긋난 tool call
- **Wrong tool selection** — 잘못된 도구 선택
- **Silent failures** — 도구는 에러를 리턴했는데 agent가 성공으로 처리

#### Berkeley Function Calling Leaderboard v4 (2025.12)
- Composite: 40% Agentic + 30% Multi-Turn + 10% Live + 10% Non-Live + 10% Hallucination
- Top: **Claude Opus 4.5 (FC) 77.47%**, Claude Sonnet 4.5 (FC) 73.24%

#### Toolformer (Schick et al. 2023, arXiv 2302.04761)
- LM이 스스로 tool 호출 학습
- 하지만 tool 정의만 바뀌어도 fine-tune 무효화

#### Gorilla (Patil et al. 2023, arXiv 2305.15334, NeurIPS 2024)
- Fine-tuned LLaMA-7B가 **AST 정확도 +20.43 abs pp (vs GPT-4)** on APIBench
- Near-zero API 환각

#### ToolLLM/ToolBench (arXiv 2307.16789)
- ToolLLaMA: Pass 66.7% / Win 60.0% vs GPT-4 70.4/70.4 on **16,464 APIs**

#### τ-bench (arXiv 2406.12045) — 일관성 붕괴
- GPT-4o pass@1: retail 60.4% / airline 42.0%
- **pass^8 retail ~25%** — 8번 시도 모두 성공하는 비율 급락 → **일관성 문제**
- Claude 3.5 Sonnet: 69.2 / 46.0

#### ToolQA (arXiv 2306.13304)
- ChatGPT CoT ~2% hard / ~5.6% easy
- 최고 ReAct GPT-3.5 43.13% easy / 8.24% hard

---

### B.3 Context Length Degradation

- Lost in the Middle (위 A.8) 재확인

#### Levy et al. 2024 — "Same Task, More Tokens"
- **arXiv 2402.14848**
- 토큰 250 → 3000 사이에서 reasoning 정확도 **0.92 → 0.68 (~24 pp 하락)**
- GPT-4, GPT-3.5, Gemini, Mistral, Mixtral 모두
- **컨텍스트 최대치보다 훨씬 낮은 수준에서 이미 하락**

#### RULER (Hsieh et al. 2024)
- **arXiv 2404.06654, COLM 2024**
- **claimed vs effective context**:
  - GPT-4 claimed 128K → **effective ~64K**
  - Yi-34B claimed 200K → effective 32K
  - Mixtral 8x7B claimed 32K → effective 32K
- → **effective context ≈ claimed의 50-65%**

#### Kamradt "Needle in Haystack"
- Claude 2.1 초기 27% → **"Here is the most relevant sentence:" 힌트 추가 시 98%**로 급등

#### HELMET (arXiv 2410.02694)
- **NIAH는 실제 장기 컨텍스트 활용을 예측 못함**
- 7개 실제 응용 카테고리 간 pairwise 상관 낮음

---

### B.4 Prompt Injection (특히 Tool 출력 통한 Indirect)

리서치했던 **peer review 공격 5편** 외에도:
- **Greshake et al. 2023** — "Not what you've signed up for" (arXiv 2302.12173, AISec '23) — indirect prompt injection 최초 체계화, Bing Chat + GitHub Copilot PoC
- **Perez & Ribeiro 2022** — "Ignore Previous Prompt: Attack Techniques"
- **Yi et al. 2023** — Benchmarking prompt injection

#### AgentDojo (arXiv 2406.13352, NeurIPS 2024) — 정량 검증
- **97 tasks × 629 security cases**
- "Important message" 공격 하의 Targeted ASR:
  - **GPT-4o 47.7%**
  - Gemini 1.5 Pro 25.6%
  - Claude 3 Sonnet 26.7%
  - Llama 3 70B 20.0%
  - Claude 3 Opus 11.3%
- **Slack tool suite 시나리오에서 92% ASR**
- Tool filtering 방어 시 7.5%로 감소

#### InjecAgent (arXiv 2403.02691, ACL Findings 2024)
- Direct harm: GPT-4 **14.7% base / 33.3% enhanced**; Llama2-70B **91.9% / 98.3%**
- Data-stealing: GPT-4 **31.9% / 59.9%**
- ReAct GPT-4 overall **24% / 47%**

**대응**: input/output filter, spotlighting, ChatML role isolation, deliberative alignment

---

### B.5 멀티-에이전트 실패 — MAST Taxonomy가 정리

#### Cemri et al. 2025 — "Why Do Multi-Agent LLM Systems Fail?" (MAST) — **핵심 논문**
- **arXiv 2503.13657, UC Berkeley**
- **14가지 failure mode, 3 카테고리**
- **1,600+ trace annotated** (150개 taxonomy용), **7개 MAS 프레임워크**, 어노테이터 간 κ = 0.88

**Failure mode 빈도**:

| 카테고리 | Mode | 빈도 |
|---------|------|------|
| **FC1 시스템 설계** | Step repetition | **15.7%** |
| | Unaware of termination | **12.4%** |
| | Disobey task spec | 11.8% |
| | Loss of history | 2.8% |
| | Disobey role | 1.5% |
| **FC2 에이전트 간 어긋남** | Reasoning-action mismatch | **13.2%** |
| | Task derailment | 7.4% |
| | Fail-to-clarify | 6.8% |
| | Conversation reset | 2.2% |
| **FC3 태스크 검증** | Incorrect verification | 9.1% |
| | Incomplete verification | 8.2% |
| | Premature termination | 6.2% |

#### Wynn et al. 2025 — "Talk Isn't Always Cheap" — 토론이 오히려 정확도를 낮춘다
- **arXiv 2509.05396 (2025.09)**
- 강한 모델이 약한 모델보다 많아도 **debate가 정확도를 떨어뜨림**
- MMLU 1×LLaMA + 2×Mistral: **40.0 → 28.0 (−12.0 pp)**
- CSQA 2×LLaMA + 1×Mistral: 58.2 → 50.2
- GSM8K 2×LLaMA + 1×Mistral: 82.6 → 75.8
- 원인: **sycophantic conformity** (아첨/편승)

#### Du et al. — Multi-agent debate baseline (positive counterexample)
- arXiv 2305.14325, ICML 2024
- 3 agent × 2 round → MMLU 64→71, GSM 77→85
- (다양성 + 강한 base model이면 도움 — 조건부)

---

### B.6 메모리 손상 / Coherence Cliff

- 장기 horizon에서 컨텍스트 열화 (Scientist Agents 리서치에서 이미 다룸)
- **Semantic memory layer 제거 시 가장 큰 하락** (arXiv 2603.29194)
- **Context-Folding** (arXiv 2510.11967), **Acon** (arXiv 2510.00615)

#### MemGPT (arXiv 2310.08560)
- Deep-Memory-Retrieval에서 **GPT-4: 32.1 → 92.5% (+60.4 pp)**
- GPT-4-Turbo: 35.3 → 93.4

#### LoCoMo (arXiv 2402.17753, ACL 2024)
- 50 대화 × 평균 304.9 turn × 19.3 session × 9,209 token
- **인간 F1 87.9%** vs **GPT-4-Turbo 32.1%** (~55-70 pp gap)
- GPT-3.5-Turbo 22.4%, Llama-2-70B 17.9%
- GPT-4-Turbo temporal reasoning **10.4%**
- **GPT-3.5-Turbo-16K가 4K보다 event summarization에서 성능 낮음** (39.9 vs 45.9) — 컨텍스트 길이 ≠ 메모리 능력

---

### B.7 비용/지연 폭주

- Cost cap 없으면 무한 루프 가능
- Agent Lab GPT-4o $2.33 vs o1-preview $13.10 사례

#### AutoGPT 무한 루프 실증 (GitHub #1211, #1994, #2590, #2726, #3637, #3979 — 2023.03-05)
- **Vectara case study**: ~2시간, 300+ API 호출, 8+ 사이클, 결과 없음
- File-org run: 15+ 번 재정리 반복

#### HAL cost tier (GAIA)
- Generalist+o3 Medium: **$17.14/task** ($2,828 총, 165 문항)
- HF Open Deep Research + Claude Opus 4.1 High: **$8.93/task**
- Sonnet 4.5 최상위: **$1.08/task**

#### Efficient Agents (arXiv 2508.02694)
- GAIA L1→L3 비용 스케일: **Claude 3.7 +534%, o1 +646%**

#### FrugalGPT (arXiv 2305.05176)
- Cascade router로 **GPT-4 매칭하며 최대 98% 비용 절감**

#### AgentBench (arXiv 2308.03688, ICLR 2024)
- GPT-4 **4.01/10** vs 최고 오픈소스 Llama-2-13B-chat **1.01/10** (8개 환경) — 4배 gap

#### AgentBoard (arXiv 2401.13178, NeurIPS 2024 Oral)
- GPT-4 progress rate: easy 79.2% → hard 62.7%
- Success: 65.6 → 34.4 (**−31.2 pp**)
- Embodied AI 최악 (−33.2 pp)

---

### B.8 샌드박스 이슈

- Docker만으로 부족 → Firecracker microVM (E2B 표준)
- Python REPL 격리 실패 사례
- **Code exec + web browsing + file system** 조합에서 위험

#### CodeAct (arXiv 2402.01030, ICML 2024)
- Python action이 JSON action 대비 **+20 abs pp**, action 수 30% 감소
- GPT-4-1106: 74.4% (CodeAct) vs 53.7% (text)
- Claude-2: 54.9 vs 29.3

#### Prompt-to-SQL Injection (arXiv 2308.01990)
- Langchain LLM 앱들이 7개 SoTA LLM 전반에 "highly susceptible"

#### ChatGPT Code Interpreter 격리 결함 (Rehberger / Embrace The Red, 2024.02 disclosed → 2024.05 patched)
- 한 계정의 모든 GPTs가 `/mnt/data` 공유
- 공개 GPT가 private GPT 파일을 read/overwrite 가능

#### Agent Security Bench (arXiv 2410.02644, ICLR 2025)
- **평균 ASR 84.30%** (4 공격 단계 × 27 방법 × 13 LLM × 400+ 도구 × ~90k test case)
- 방어책들 "limited effectiveness"

### B.9 Reward Hacking / 조작 (더 깊이)

#### Denison et al. — "Sycophancy to Subterfuge" (Anthropic 2024)
- **arXiv 2406.10162**
- Specification-gaming 커리큘럼 후: **45/32,768 rollout (0.14%)이 reward tampering**
- **7/32,768 (0.021%)은 test까지 편집해 흔적 은폐**
- Baseline 모델은 100k 중 0

#### Sakana AI CUDA Engineer 심층 (arXiv 2509.14279 후속)
- 원 주장: 10-100× (최대 150×) → 실측 **~3× 슬로우다운**
- KernelBench "50-120× speedup"이 실제로는 `torch.cuda.Event` 타이밍 exploit, `torch.jit.fork` 백그라운드 스레드, state caching, correctness 체크 우회 메모리 exploit
- 수정 후 median **1.10× native / 1.19× compile**

#### RE-Bench (METR, arXiv 2411.15114)
- 2h budget → AI **4× 인간**
- 8h → 인간 우세
- **32h → 인간 2× AI** (long-horizon에서 인간 재역전)

---

## Part C. 최적화 기법

### C.1 프롬프트 자동 최적화

#### DSPy (Khattab et al. 2023-2024)
- **arXiv 2310.03714**
- **핵심 발상**: 프롬프트를 코드로 취급, few-shot을 자동 발견
- BootstrapFewShot, MIPRO 등 옵티마이저
- **결과**: HotPotQA에서 hand-tuned chain 대비 +30-40% 개선 다수

#### OPRO (Yang et al. 2023, arXiv 2309.03409)
- **"Optimization by PROmpting"**
- LLM 자체를 optimizer로 사용, meta-prompt로 다른 prompt 개선
- GSM8K에서 hand-crafted "Let's think step by step" 대비 +8%

#### EvoPrompt (Guo et al. 2023, arXiv 2309.08532)
- 진화 알고리즘으로 프롬프트 최적화
- **APE** (Zhou et al. 2022, arXiv 2211.01910) — Automatic Prompt Engineer

#### PromptBreeder (Fernando et al. 2023, arXiv 2309.16797)
- **self-improving** 프롬프트 진화

---

### C.2 Multi-Prompt Aggregation

- Mizrahi et al. 권고: **최소 5개 template 평균**
- **SEval** — bootstrap over prompts로 신뢰구간 산출
- HELM은 default로 다중 template 실행

---

### C.3 오염 탐지

#### Canary strings (BIG-Bench 초기부터)
- 벤치마크에 고유 문자열 심어놓기 → 모델이 그 문자열을 알면 오염
- 예: `BENCHMARK_DATA_SHOULD_NOT_BE_TRAINED_ON`

#### Membership Inference for Contamination
- Duan et al. 2024 — MIA로 오염 탐지
- **Min-K% probability**: 낮은 확률 토큰 비율로 오염 감지

#### Rolling / Live Benchmarks
- **LiveBench** (arXiv 2406.19314, Yann LeCun co-author) — 매월 갱신, 대회/논문/뉴스 최신
- **LiveCodeBench** — 코딩 문제 rolling

---

### C.4 Robust LLM-as-Judge (verified 수치 포함)

#### G-Eval (Liu et al. 2023, arXiv 2303.16634, EMNLP 2023)
- CoT + form-filling
- SummEval Spearman **0.514**

#### Prometheus (Kim et al. 2023, arXiv 2310.08491, ICLR 2024)
- Llama-2-13B judge, 45 rubric에서 인간과 Pearson **r = 0.897**
- (GPT-4 0.882, ChatGPT 0.392 능가)

#### Prometheus 2 (arXiv 2405.01535)
- 7B/8x7B, absolute + pairwise weight merging

#### Auto-J (arXiv 2310.05470, ICLR 2024)
- 13B judge, 58 실제 시나리오, pairwise + single + critique

#### Panickssery et al. 2024 — "LLM Evaluators Recognize and Favor Their Own Generations"
- **arXiv 2404.13076, NeurIPS 2024**
- GPT-4 self-recognition **73.5%**
- Self-preference GPT-4: **0.705 (XSUM) / 0.912 (CNN/DM)**
- Llama-2도 500-example fine-tune 후 self-recog >90%
- Fine-tuned GPT-3.5 Kendall τ = 0.74 (self-recog ↔ self-pref 연결)

#### Xu et al. 2024 — "Pride and Prejudice: Self-refine loop이 self-bias 증폭"
- **arXiv 2402.11436, ACL 2024**

#### PORTIA / Split & Merge (Li et al. arXiv 2310.01432, EMNLP 2024)
- Answer 정렬 + 세그먼트 → **judge consistency-rate 평균 +47.46%**

#### LLMBar (arXiv 2310.07641, ICLR 2024)
- 419 adversarial pair: verbose-wrong이 concise-correct 이김
- 대부분 judge 속음

#### Verba et al. 2024 — "Replacing Judges with Juries"
- Multi-judge ensemble

---

### C.5 Guided Decoding & Structured Output — **주의: 양날의 검**

- **Guidance** (Microsoft), **Outlines** (Willard & Louf, arXiv 2307.09702) — FSM 기반, near-zero token-time overhead
- **DOMINO** (Beurer-Kellner et al., arXiv 2403.06988, ICML 2024) — subword-aligned constraint, up to 2× speedup
- **XGrammar** (Dong et al. 2024, arXiv 2411.15100) — **prior grammar decoding 대비 최대 100× 속도**
- **LMQL**, **jsonformer**
- OpenAI JSON mode, structured outputs

**효과**: answer extraction 실패 근본 원인 제거

#### Tam et al. 2024 — "Let Me Speak Freely?" — **JSON이 reasoning을 심각하게 해친다**
- **arXiv 2408.02442**
- **GPT-3.5-Turbo GSM8K NL → JSON: 76.60 → 49.25 (−27.35 pp)**
- **Claude-3-Haiku GSM8K NL → JSON: 86.51 → 23.44 (−63.07 pp)** ⚠️
- **LLaMA-3-8B GSM8K NL → JSON: 74.73 → 48.90 (−25.83 pp)**
- Last Letter Concat도 유사 하락
- **분류 태스크에선 반대 — JSON이 도움**: DDXPlus GPT-3.5 +11.44 pp, Sports Understanding +12.84 pp
- **교훈**: reasoning-heavy에는 free-form, extraction/classification에는 structured

---

### C.6 Efficient Batched Evaluation

- **vLLM** (Kwon et al. 2023, arXiv 2309.06180) — PagedAttention, KV cache reuse
- lm-eval-harness의 vLLM 통합 → 10x+ 속도
- **SGLang** (Zheng et al. 2023, arXiv 2312.07104) — structured programming

---

### C.7 검증 가능한 보상 (RLVR) — verified 수치

#### DeepSeek-R1 (2025.01, arXiv 2501.12948, Nature 645:633–638)
- **AIME 2024 pass@1 79.8%**, MATH-500 97.3%
- Codeforces 2029 rating (96.3%ile)
- GPQA-Diamond 71.5%, MMLU 90.8%, LiveCodeBench 65.9%
- SWE-Bench Verified 49.2%
- **R1-Zero (pure RL, SFT 없음): AIME 71.0%** (초기 15.6에서)

#### Kimi K1.5 (Moonshot, arXiv 2501.12599)
- Long-CoT: AIME 77.5, MATH-500 96.2, MathVista 74.9
- Short-CoT: LiveCodeBench 47.3
- **NO MCTS, NO value function**

#### Tulu 3 — "RLVR" 명명 논문 (arXiv 2411.15124)
- DPO checkpoint 위에 RLVR: **MATH +1.7 pp, GSM8K +3.3 pp, IFEval +1.3 pp**

#### OpenAI o1 (2024.09 blog)
- AIME 2024 pass@1 74%, cons@64 83%, re-ranked@1000 **93%**
- (vs GPT-4o cons@64 13.4%)

**시사점**: LLM-as-judge의 대안으로 **verifiable reward**가 새 표준

---

### C.8 Chatbot Arena / LMSYS 방법론

- **Chatbot Arena** (Chiang et al. 2024, arXiv 2403.04132, ICML 2024)
- Pairwise human vote → Bradley-Terry Elo (sandwich robust SE)
- 크라우드-전문가 일치 **72.8-73.8%**, 전문가-전문가 79.4-89.8%
- Adaptive sampling으로 54% 샘플 절감 (0.2 precision)

#### Singh et al. 2025 — "The Leaderboard Illusion" — Arena에도 큰 편향
- **arXiv 2504.20879 (2025.04-05)**
- **Llama-4의 27개 비공개 variant가 사전 테스트됨**
- **Google 19.2%, OpenAI 20.4%** of 전체 배틀 vs 83개 open-weight 모델 합계 **29.7%**
- 추가 데이터로 "Arena 분포에서 최대 **112% 상대 성능 향상**" 가능

#### Length-Controlled AlpacaEval (Dubois et al., arXiv 2404.04475)
- Arena와 상관 **0.94 → 0.98** (length control 후), **비용 <$10, <3분**

#### Arena-Hard-Auto (Li et al., arXiv 2406.11939, NeurIPS 2024)
- 500 auto-curated prompt, Arena와 최고 상관·분리도

#### MixEval (Ni et al., arXiv 2406.06565, NeurIPS 2024)
- **Arena와 0.96 상관, MMLU의 6% 시간/비용**

#### WildBench (Lin et al., arXiv 2406.04770)
- 1,024 실제 사용자 태스크
- WB-Reward Pearson **0.98**, WB-Score **0.95** (vs Arena)
- Arena-Hard 0.91, AlpacaEval 2.0 0.89 능가

#### JudgeBench (Tan et al., arXiv 2410.12784, ICLR 2025)
- 판사를 판사 — 상위 judge들이 어려운 reasoning pair에서 near-chance

---

## Part D. 최근 트렌드 (2025-2026)

### D.1 HELM (Stanford CRFM) 최신
- HELM Instruct, HELM Safety, HELM Multimodal, HELM Legal, HELM Med
- **holistic** 평가 (accuracy만이 아니라 fairness/robustness/efficiency)

### D.2 컨테이너화된 벤치마크
- **NatureBench** (arXiv 2606.24530) — auto-containerization, web-search-disabled
- **ResearchGym / InnovatorBench**
- **PaperBench** (OpenAI) — rubric 8,316 항목

### D.3 Verifiable Rewards 확산
- DeepSeek-R1 이후 대부분 reasoning 모델이 RLVR 채택
- 수학, 코딩, 형식 검증 가능 도메인 우선

### D.4 Multi-Modal 하네스
- **MMMU**, **MMBench**, **MMMU-Pro** — 이미지+텍스트
- **VideoMME** — 비디오
- 오디오, 3D 등 확장 중

### D.5 안전성 하네스
- **HarmBench** (arXiv 2402.04249) — 400 harmful behaviors × 18 red-team × 33 targets
- **JailbreakBench** (arXiv 2404.01318, NeurIPS 2024) — 100 policy-aligned + jailbreak-artifacts repo
- **StrongREJECT** — jailbreak 평가 표준화

### D.6 통합 에이전트 벤치마크 스냅샷

| 벤치마크 | arXiv | Headline |
|---------|-------|---------|
| **GAIA** (Mialon et al.) | 2311.12983 | 인간 **92%** vs GPT-4+plugins **14.6%**, Level 3 **0.0%** |
| **WebArena** (Zhou et al.) | 2307.13854 | GPT-4 **14.41%** vs 인간 78.24% (812 task) |
| **OSWorld** (Xie et al.) | 2404.07972 | 인간 **72.36%** vs 최고 모델 **12.24%** (369 desktop) |
| **WebVoyager** (He et al.) | 2401.13919 | GPT-4V **59.1%** on 15 live sites |
| **SWE-Lancer** (OpenAI) | 2502.12115 | Claude 3.5 Sonnet: **$403,325** of $1M Upwork pool |
| **SWE-Bench Verified 2026** | | Claude Opus 4.5 **80.9%**, GPT-5.5 ~82.6% |
| **SWE-Bench Pro (contamination-controlled)** | | Claude Opus 4.5 **45.9%** (Verified 80.9%와 대비) → Verified 오염 시사 |

### D.7 HELM 최신 (2025.03)
- 5 시나리오 mean-score: **Gemini 2.0 Flash 0.679, Claude 3.7 Sonnet 0.674, DeepSeek-V3 0.665**
- 시나리오: MMLU-Pro, GPQA, IFEval, WildBench, Omni-MATH

### D.8 SWE-Bench Verified 위기 (2026)
- OpenAI가 Verified 리포팅 중단
- o3 fail 138건 audit: **59.4%가 material issue** (broken test / under-spec)
- **2024.06 이후 학습 모델은 GitHub 데이터로 오염** 가능성 flag

---

## Part E. 실전 권고 (요약)

### E.1 평가 하네스 사용 시

| # | 권고 |
|---|-----|
| 1 | **최소 5개 프롬프트 template 평균** — 단일 프롬프트는 무의미 |
| 2 | **Few-shot 순서 무작위화 + 평균** |
| 3 | **Bootstrap CI 병기** — 순위 뿐 아니라 신뢰구간도 |
| 4 | **하네스 커밋 해시 + 프롬프트 파일 명시** — 재현성 |
| 5 | **Contamination 체크** (LiveBench 등 rolling benchmark 병행) |
| 6 | **Structured output / guided decoding** — 파싱 실패 제거 |
| 7 | **LLM judge는 position swap + reference-guided + 다중 judge** |
| 8 | **Log-likelihood + generation 둘 다 측정** |
| 9 | **saturation 벤치마크(MMLU) 대신 MMLU-Pro/GPQA/MATH-L5** |
| 10 | Chatbot Arena 순위는 style bias 감안 |

### E.2 Agent 하네스 구축 시

| # | 권고 |
|---|-----|
| 1 | **External verifier 필수** — 자체 채점은 100% reward hacking |
| 2 | **Firecracker microVM sandbox** — Docker만으론 부족 |
| 3 | **Cost cap 강제** — token/$/wall-clock 각각 |
| 4 | **Tool output sanitize** — indirect prompt injection 방지 |
| 5 | **Structured tool call schema + validation** — malformed 회피 |
| 6 | **다중 backbone critic** — same-model self-preference 회피 |
| 7 | **Audit log mandatory** — Hidden Pitfalls 권고 |
| 8 | **Long-horizon은 memory tier 분리** — episodic/semantic/procedural |
| 9 | **Reward hacking을 사전에 gaming** — 자체 red-team |
| 10 | **AgentDojo / BFCL 등 표준 벤치마크로 검증** |

---

---

## Part F. 정정 사항 (배경 리서치 반영)

리서치 검증 과정에서 발견한 필자 초안 오류:

| 항목 | 정정 |
|------|------|
| Lu et al. "51.6%~93.4% SST-2" | 실제 abstract는 "near SOTA ~ random guess"; body 기준 GPT2-XL SST-2 **88.7% → 51.6%** (여전히 큰 스윙) |
| MMLU-Redux 오류율 | 초안 "9%" 아니라 **전체 6.49%, Virology 서브셋 57%** |
| Chain of Thoughtlessness 저자 | Liu et al. (2410.21333) 아니라 **Stechly et al. (2405.04776)**. 2410.21333은 "Mind Your Step"이 정답 |
| Sainz et al. venue | EMNLP 2023 Findings로 정정 |
| DeepSeek-R1 / Kimi K1.5 | 둘 다 2025.01.22 arXiv, DeepSeek-R1은 이후 Nature 645:633-638 게재 |

---

## 결론 — 두 문장 요약

> **"모델보다 harness가 결과를 좌우한다."** 프롬프트 한 글자, few-shot 순서, judge 편향, contamination — 이것들을 통제하지 않은 벤치마크 점수는 신뢰할 수 없다.

> **"Agent harness에서 자체 평가는 100% reward hacking이 된다."** External verifier / verifiable reward / audit log는 옵션이 아니라 architecture 1순위.

---

## 관련 블로그 포스트

- [Scientist Agents 차별화 설계 차원 20가지](scientist-agents-design-dimensions.md) — Agent harness 관점
- [Scientist Agents 1년 종합 보고서](scientist-agents-2025-2026-report.md) — MLR-Bench 등 fabrication 사례
- [Search-R1 리뷰](search-r1-review.md) — Verifiable reward 사례
- [강화학습 입문](reinforcement-learning-beginner-guide.md) — RLVR 이해 기반

---

## 참고 문헌 (핵심)

### 평가 하네스 오류
- [Sclar et al. "Quantifying Sensitivity" (arXiv:2310.11324)](https://arxiv.org/abs/2310.11324)
- [Lu et al. "Fantastically Ordered Prompts" (arXiv:2104.08786)](https://arxiv.org/abs/2104.08786)
- [Zhao et al. "Calibrate Before Use" (arXiv:2102.09690)](https://arxiv.org/abs/2102.09690)
- [Mizrahi et al. "State of What Art?" TACL 2024](https://aclanthology.org/2024.tacl-1.55/)
- [Sainz et al. "NLP Evaluation in Trouble" (arXiv:2310.18018)](https://arxiv.org/abs/2310.18018)
- [Golchin & Surdeanu "Time Travel" (arXiv:2308.08493)](https://arxiv.org/abs/2308.08493)
- [Biderman et al. "Lessons from the Trenches" (arXiv:2405.14782)](https://arxiv.org/abs/2405.14782)
- [Liu et al. "Lost in the Middle" (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172)
- [Miller "Adding Error Bars" (arXiv:2411.00640)](https://arxiv.org/abs/2411.00640)
- [Perlitz et al. "Efficient Benchmarking" (arXiv:2308.11696)](https://arxiv.org/abs/2308.11696)
- [Turpin et al. "LMs Don't Always Say What They Think"](https://arxiv.org/abs/2305.04388)
- [Zheng et al. "LLM-as-a-Judge / MT-Bench" (arXiv:2306.05685)](https://arxiv.org/abs/2306.05685)
- [Wang et al. "LLMs are not Fair Evaluators" (arXiv:2305.17926)](https://arxiv.org/abs/2305.17926)

### Agent 하네스 오류
- [Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761)
- [Gorilla (arXiv:2305.15334)](https://arxiv.org/abs/2305.15334)
- [AgentDojo (arXiv:2406.13352)](https://arxiv.org/abs/2406.13352)
- [MLR-Bench (arXiv:2505.19955)](https://arxiv.org/abs/2505.19955)
- [Greshake et al. "Not what you've signed up for"](https://arxiv.org/abs/2302.12173)

### 최적화 기법
- [DSPy (arXiv:2310.03714)](https://arxiv.org/abs/2310.03714)
- [OPRO (arXiv:2309.03409)](https://arxiv.org/abs/2309.03409)
- [APE (arXiv:2211.01910)](https://arxiv.org/abs/2211.01910)
- [PromptBreeder (arXiv:2309.16797)](https://arxiv.org/abs/2309.16797)
- [G-Eval (arXiv:2303.16634)](https://arxiv.org/abs/2303.16634)
- [Prometheus (arXiv:2310.08491)](https://arxiv.org/abs/2310.08491)
- [XGrammar (arXiv:2411.15100)](https://arxiv.org/abs/2411.15100)
- [vLLM (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180)
- [LiveBench (arXiv:2406.19314)](https://arxiv.org/abs/2406.19314)
- [Chatbot Arena (arXiv:2403.04132)](https://arxiv.org/abs/2403.04132)
- [DeepSeek-R1 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948)

### 벤치마크 (오염 대응)
- [MMLU-Pro (arXiv:2406.01574)](https://arxiv.org/abs/2406.01574)
- [GPQA (arXiv:2311.12022)](https://arxiv.org/abs/2311.12022)
- [BFCL (Berkeley Function Calling Leaderboard)](https://gorilla.cs.berkeley.edu/leaderboard.html)
