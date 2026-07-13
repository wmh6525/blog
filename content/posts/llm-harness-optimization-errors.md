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

#### Sclar et al. 2023 — "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design"
- **arXiv 2310.11324**
- **핵심 발견**: 프롬프트 표면 특징 (구분자, 대소문자, 공백)만 바꿔도 **최대 76% accuracy 스윙**
- 같은 태스크, 같은 모델, 같은 few-shot — **오직 formatting만 다름**
- Llama-2-7B에서 accuracy 분포가 0~80% 전체 범위 커버

#### Mizrahi et al. 2024 — "State of What Art? A Call for Multi-Prompt LLM Evaluation"
- **TACL 2024**
- **핵심 발견**: 단일 프롬프트로 순위 매기면 **39%의 페어에서 순위가 뒤집힘**
- **권고**: 최소 5개 프롬프트 변형 평균 → 순위 안정화

#### Voronov et al. — "Mind Your Format"
- Few-shot 예제의 **구분자 문자** 하나 (`\n` vs `\n\n` vs `###`)가 점수에 큰 영향

**시사점**: 단일 프롬프트 벤치마크 결과는 **거의 무의미**. 최소 다중 프롬프트 평균 필요.

---

### A.2 Few-shot Ordering Effects

#### Lu et al. 2022 — "Fantastically Ordered Prompts and Where to Find Them"
- **ACL 2022, arXiv 2104.08786**
- **핵심 발견**: 같은 few-shot 예제 4개, 순서만 바꿔서 24가지 순열 → **51.6% ~ 93.4% (SST-2)**
- GPT-3, GPT-2 모두 영향받음
- 좋은 순서 자동 발견: **entropy 기반 selection**

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
- 이후 **MMLU-Pro** (arXiv 2406.01574) — 다중 정답, 10 선택지, 오염 방지
- **GPQA** (arXiv 2311.12022) — Google-proof, PhD 수준
- **BIG-Bench Hard, MATH-Level 5, Humanity's Last Exam** (2025) — 어려운 방향으로

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
- **Berkeley Function Calling Leaderboard** (BFCL) — 도구 호출 표준 평가

#### Toolformer (Schick et al. 2023, arXiv 2302.04761)
- LM이 스스로 tool 호출 학습
- 하지만 tool 정의만 바뀌어도 fine-tune 무효화

#### Gorilla (Patil et al. 2023, arXiv 2305.15334)
- API 호출 특화, 환각 API 감소

---

### B.3 Context Length Degradation

- Lost in the Middle (위 A.8) 재확인
- **NIAH (Needle-in-a-Haystack) score**가 실제 활용도와 상관 낮음
- **Extended NIAH** — 다중 needle, 방해 요소

---

### B.4 Prompt Injection (특히 Tool 출력 통한 Indirect)

리서치했던 **peer review 공격 5편** 외에도:
- **Greshake et al. 2023** — "Not what you've signed up for" — indirect prompt injection 최초 체계화
- **Perez & Ribeiro 2022** — "Ignore Previous Prompt: Attack Techniques"
- **Yi et al. 2023** — Benchmarking prompt injection
- **AgentDojo** (arXiv 2406.13352) — realistic prompt injection benchmark

**대응**: input/output filter, spotlighting, ChatML role isolation, deliberative alignment

---

### B.5 멀티-에이전트 실패

- **Echo chamber / groupthink** — 여러 에이전트가 같은 backbone이면 diversity 없음
- **Coordination overhead** — communication cost가 quality 이익보다 큼
- **Multi-agent Debate ablation** (arXiv 2511.07784, 이전 리서치): 이익은 debate 구조가 아니라 **intrinsic reasoning + group diversity**에서 옴

---

### B.6 메모리 손상 / Coherence Cliff

- 장기 horizon에서 컨텍스트 열화 (Scientist Agents 리서치에서 이미 다룸)
- **Semantic memory layer 제거 시 가장 큰 하락** (arXiv 2603.29194)
- **Context-Folding** (arXiv 2510.11967), **Acon** (arXiv 2510.00615)

---

### B.7 비용/지연 폭주

- Cost cap 없으면 무한 루프 가능
- Agent Lab GPT-4o $2.33 vs o1-preview $13.10 사례
- **AgentBench** (arXiv 2308.03688), **AgentQuest** — 비용도 metric

---

### B.8 샌드박스 이슈

- Docker만으로 부족 → Firecracker microVM (E2B 표준)
- Python REPL 격리 실패 사례
- **Code exec + web browsing + file system** 조합에서 위험

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

### C.4 Robust LLM-as-Judge

#### G-Eval (Liu et al. 2023, arXiv 2303.16634)
- CoT + form-filling으로 judge 안정화
- SummEval에서 이전 지표 대비 human correlation ↑

#### Pandalm, JudgeLM (Zhu et al. 2023, arXiv 2310.17631)
- Fine-tuned judge 모델
- Open weights, reproducible

#### Prometheus (Kim et al. 2023, arXiv 2310.08491) / Prometheus 2 (2024)
- Fine-grained judge, rubric-based
- **오픈 소스 judge의 표준**

#### Calibration & de-biasing
- Position swap (Zheng)
- Reference-guided (MT-Bench)
- Multi-judge ensemble (Verga et al. 2024 "Replacing Judges with Juries")

---

### C.5 Guided Decoding & Structured Output

- **Guidance** (Microsoft), **Outlines**, **LMQL**, **jsonformer**
- **XGrammar** (Dong et al. 2024, arXiv 2411.15100) — grammar 강제 decoding, low overhead
- OpenAI JSON mode, structured outputs

**효과**: answer extraction 실패 근본 원인 제거

---

### C.6 Efficient Batched Evaluation

- **vLLM** (Kwon et al. 2023, arXiv 2309.06180) — PagedAttention, KV cache reuse
- lm-eval-harness의 vLLM 통합 → 10x+ 속도
- **SGLang** (Zheng et al. 2023, arXiv 2312.07104) — structured programming

---

### C.7 검증 가능한 보상 (RLVR)

#### DeepSeek-R1 (2025, arXiv 2501.12948)
- **RL with Verifiable Rewards** — 수학/코딩처럼 정답이 명확한 도메인
- Reward hacking 회피 (자체 채점 X)

#### Kimi K1.5 (Moonshot 2025, arXiv 2501.12599)
- 유사 RLVR 접근

**시사점**: LLM-as-judge의 대안으로 **verifiable reward**가 새 표준

---

### C.8 Chatbot Arena / LMSYS 방법론

- **Chatbot Arena** (Chiang et al. 2024, arXiv 2403.04132)
- Pairwise human vote → Bradley-Terry Elo
- **한계**: 
  - Prompt distribution 편향
  - Long-form 선호
  - Style-over-substance
- **Arena Hard** — 난이도 필터링

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
- **HarmBench** (arXiv 2402.04249), **AdvBench**, **JailbreakBench**
- **StrongREJECT** — jailbreak 평가 표준화

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
