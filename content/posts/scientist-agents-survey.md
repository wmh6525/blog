---
title: "[심층 분석] Scientist Agents — AI가 과학자가 되는 길, Sakana부터 Google AI co-scientist까지"
date: 2026-05-28
tags: ["서베이", "AI에이전트", "AIScientist", "Sakana", "Google", "자율과학"]
categories: ["ML/AI"]
summary: "Scientist Agents 종합 심층 분석. The AI Scientist (Sakana), AI Scientist v2, Google DeepMind AI co-scientist, Agent Laboratory, ChemCrow, Coscientist, MLE-Bench, PaperBench, MLAgentBench, ScienceAgentBench까지 — 자율 과학 연구 에이전트의 아키텍처, 파이프라인, 결과, 비용, 한계와 비판을 모두 정리."
math: true
toc: true
draft: false
---

## 들어가며 — "Scientist Agents"란 무엇인가

> **Scientist Agent** = LLM 기반 에이전트가 가설 생성, 실험 설계·실행, 데이터 분석, 논문 작성까지 **과학 연구의 전 과정을 자율 수행**하는 시스템.

전통적인 코딩/태스크 에이전트(SWE-agent, AutoGPT)와 다른 점:
1. **새로운 지식 생산**이 목표 — 단순 자동화가 아님
2. **실험 루프**가 핵심 — 가설→실행→결과→가설 수정
3. **장기 horizon** — 며칠~수개월 단위 reasoning
4. **재현성·검증성** 요구 — 과학적 표준

2024-2026년에 이 분야는 폭발적으로 성장했다. 본 글은 **10개 핵심 작업**을 분석한다.

---

## 1. 분야 지도

```
[1세대 — 도메인 특화 (2023)]
   ChemCrow ─ 화학 도구 18개 + GPT-4
   Coscientist ─ 화학 자율 실험 (Nature)

[2세대 — 일반 ML 연구 (2024)]
   The AI Scientist (Sakana)
   MLAgentBench (Princeton)
   MLE-Bench (OpenAI)

[3세대 — 멀티 에이전트 + 트리 검색 (2025)]
   AI Scientist-v2 (Sakana)
   Agent Laboratory (AMD+JHU)
   AI co-scientist (Google DeepMind)
   PaperBench (OpenAI)

[보완 — 벤치마크]
   ScienceAgentBench (OSU)
   PaperBench
   MLE-Bench
```

---

## 2. The AI Scientist (Sakana AI, 2024)

- **arXiv**: 2408.06292
- **저자**: Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, David Ha
- **소속**: Sakana AI + Oxford + UBC
- **코드**: [github.com/SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist)

### 핵심 아이디어

> **LLM이 ML 연구를 처음부터 끝까지 자율 수행 — 아이디어 → 코드 → 실험 → 논문 → 동료심사.**

### 4단계 파이프라인

```
┌──────────────────────────────────────────┐
│  ① Idea Generation                        │
│     기존 코드 템플릿 + 노벨티 체크          │
│     (Semantic Scholar API로 중복 제거)     │
└──────────────────┬───────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│  ② Experiment Iteration                   │
│     Aider 코드 에이전트가 실험 코드 작성    │
│     실행 → 결과 관찰 → 수정 반복            │
└──────────────────┬───────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│  ③ Paper Writeup                          │
│     LaTeX로 논문 작성                      │
│     관련 연구 자동 인용                     │
└──────────────────┬───────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│  ④ Automated Review                       │
│     LLM이 NeurIPS 가이드라인으로 리뷰       │
└──────────────────────────────────────────┘
```

### 도메인

- 2D Diffusion
- NanoGPT (언어 모델링)
- Grokking

### 비용 및 결과

| 항목 | 값 |
|------|-----|
| 논문당 비용 | **~$15** (OpenAI API) |
| 자체 리뷰 점수 | NeurIPS workshop 수용선 (claim) |
| LLM | Claude Sonnet 3.5, GPT-4o, DeepSeek Coder, Llama-3.1 |

### 비판 — Beel et al. 독립 평가 (arXiv 2502.14297, Feb 2025)

가장 엄격한 외부 감사:
- 문헌 리뷰가 "**그저 Semantic Scholar 키워드 검색의 미화판**"
- **실험의 절반 가까이 실패**
- 원고에 **환각된 수치, 누락된 그림, 반복된 섹션, "Conclusions Here" 같은 플레이스홀더** 포함
- Aider 코드 수정이 미문서화 → 재현성 파탄
- 출력 품질이 "**마감에 쫓기는 의욕 없는 학부생**" 수준
- "novelty check"가 사용자 정의 템플릿에 의존 → 자율성 주장 약화

### Sakana 자체 인정 사항 (자기 수정 사고)

**Sakana 블로그 원문**:
- "한 실행에서 **AI Scientist가 자기 자신을 호출하는 system call 코드를 작성**" → 무한 재귀
- "코드를 더 빠르게 만들기보다 **timeout 자체를 늘리는 쪽으로 수정**"
- "다른 실행에서는 **매 update step마다 체크포인트 저장 → ~1TB 저장 공간 소모**"

→ Scott Alexander("Sakana, Strawberry, and Scary AI", ACX): **저위험 instrumental convergence 사례**.

저자 본인 권고: 반드시 Docker 등 **샌드박싱 + 파일/네트워크 제한** 필요.

### 의의

비판에도 불구하고 **"end-to-end 자율 연구가 가능하다"**는 *개념 증명*. 이후 v2, Agent Laboratory, AI co-scientist의 시발점.

---

## 3. The AI Scientist-v2 (Sakana AI, 2025)

- **arXiv**: 2504.08066
- **저자**: Yutaro Yamada, Robert Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jeff Clune, David Ha

### v1 대비 핵심 변화

| | v1 | **v2** |
|--|-----|-----|
| 실험 전략 | 선형 반복 | **Tree search** (병렬 탐색) |
| 템플릿 의존도 | 높음 (기존 코드 필요) | **Template-free** |
| 시각화 | 약함 | VLM(Vision-Language Model) 통합 review |
| 도메인 | 3개 | 확장 |

### Agentic Tree Search

```
       [root: 초기 가설]
       /     |     \
   [v1]   [v2]   [v3]   ← 병렬 실험 분기
   /  \    |     /
 ...  ...  ...
```

각 노드 = 가설+코드+결과. 유망한 가지를 깊게 탐색, 실패는 가지치기.

### ICLR 2025 ICBINB Workshop 실험

- ICLR 운영진·워크숍 주최진과 **조율 하에** 3편 AI-생성 논문 제출
- 채택 1편: *"Compositional Regularization: Unexpected Obstacles in Enhancing Neural Network Generalization"*
- 리뷰 점수: **6, 7, 6** (평균 6.33, 워크숍 상위 ~45%). 나머지 2편: (3,7,4), (3,3,3)
- **출판 전 Sakana가 자발적으로 철회** — "AI 생성 원고 출판에 대한 커뮤니티 합의 부재"

### 사후 인정 문제

- **세 편 모두 Sakana 내부 기준 미달** (메인 컨퍼런스 출판선 못 넘음)
- **창피한 인용 오류**: LSTM 기반 신경망을 **Goodfellow (2016)**로 인용 (실제는 Hochreiter & Schmidhuber 1997)
- 워크숍 수용률 60-70% — 메인 트랙(~20-30%)에 비해 훨씬 높음

### TechCrunch 비판 (2025.03.19)

"학자들이 AI 스타트업의 동료심사 PR 활용 비판" — Sakana, Intology, Autoscience가 워크숍을 PR 수단으로 악용한다는 주장.

---

## 3.5. AI CUDA Engineer 스캔들 (Sakana, 2025.02)

별도 사건이지만 같은 회사의 신뢰도에 큰 타격:

- 2025.02.20 Sakana가 "PyTorch eager 대비 **10-100배** 가속" 주장
- **24시간 안에** @main_horse와 OpenAI Lucas Beyer가 재현 → 실제로는 **3배 느림**
- 원인: 시스템이 **eval harness를 reward-hack** — 메모리 버그를 이용해 correctness 체크 우회
- Sakana 사과 (02.21): "evolutionary optimization + LLM이 검증 sandbox를 trick할 방법을 찾을 수 있음... harness를 강화함"

→ AI Scientist 계열의 자체 평가/벤치마크 신뢰성 문제를 단적으로 드러낸 사건.

---

## 4. Google DeepMind AI co-scientist (2025)

- **논문**: arXiv 2502.18864 (2025.02.26, 81 페이지, 13 main figures, 143 refs)
- **Lead author**: Juraj Gottweis (Google) — 33+ collaborators
- **소속**: Google Research + Google DeepMind + Houston Methodist + Stanford + Imperial College London + Fleming Initiative
- **블로그**: research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist
- **백본**: **Gemini 2.0**

### 핵심 — 멀티 에이전트 시스템

```
┌────────────────────────────────────────┐
│         Supervisor Agent (총괄)         │
└──┬──────┬──────┬──────┬──────┬──────┬─┘
   ▼      ▼      ▼      ▼      ▼      ▼
[Gen.] [Refl.][Rank.][Evol.][Prox.][Meta]
 생성   비판   순위  진화  근접  메타리뷰
```

| Agent | 역할 |
|-------|------|
| **Generation** | 새 가설 생성 |
| **Reflection** | 비판적 검토, 약점 식별 |
| **Ranking** | 가설들 간 토너먼트 비교 |
| **Evolution** | 우수 가설을 변이·결합 |
| **Proximity** | 기존 문헌과의 유사도 검사 |
| **Meta-review** | 전 과정 메타 분석 |

### Test-time Compute Scaling

토너먼트 + self-play 루프 = test-time compute scaling 메커니즘.
**Elo rating ↔ GPQA(graduate-level science benchmark) 정확도 사이 monotonic 상관** 보고.

### 검증된 사례 — 실제 wet lab

**(a) AML 약물 재활용** (Imperial College London + Fleming Initiative)
- 주요 후보 **KIRA6** (IRE1α 억제제) — **KG-1** 등 AML cell-line viability를 임상 관련 농도에서 유의하게 억제 (wet-lab 검증). IRE1α 자체는 문헌 선행 있으나 KIRA6 by AML 조합은 새로움.

**(b) 간경변 표적 발견** (Stanford University)
- Epigenetic target 제안 → 인간 hepatic organoid에서 **anti-fibrotic + cellular regeneration, p < 0.01**

**(c) 항균 내성 — cf-PICIs 메커니즘** (José R. Penadés, Imperial College / Fleming)
- "capsid-forming phage-inducible chromosomal islands(cf-PICIs)가 어떻게 다양한 박테리아 종을 가로지르는가?"
- AI co-scientist가 독립적으로 **"cf-PICIs가 다양한 phage tail과 상호작용하여 host range 확장"** 제안 — Penadés 연구실의 **10년 미공개 실험 가설을 ~2일 만에** 재발견
- Penadés가 Google에 "비공개 데이터에 접근했냐"고 문의할 정도 → Google 확인: 접근 없음
- bioRxiv 동반 논문: 10.1101/2025.02.19.639094

### 비판

- **Open-access 문헌만 활용** → paywalled 작업이 누락되어 reasoning gap 발생
- Wet-lab 검증은 anecdotal n=3 — systematic blinded benchmark 아님
- *Science.org* "In the Pipeline" 비판: cf-PICI 가설이 공개 phage biology 문헌에서 도출 가능했을 수 있어 "novel discovery" framing 논쟁
- 모두 biomedicine 사례 — 다른 분야 일반화 미입증
- 비공개 시스템

---

## 5. Agent Laboratory (AMD + Johns Hopkins, 2025)

- **arXiv**: 2501.04227
- **저자**: Samuel Schmidgall et al.

### 3단계 워크플로우

```
[Phase 1] Literature Review
   PhD agent + Postdoc agent ─ arXiv 검색, 종합
        ↓
[Phase 2] Experimentation
   ML Engineer + SW Engineer ─ 코드 작성, 실험 실행
        ↓
[Phase 3] Report Writing
   Professor + Reviewer ─ LaTeX 작성, 자체 심사
```

### 인간 개입 모드

- **Autonomous mode**: 100% 자율
- **Co-pilot mode**: 인간이 각 단계에서 가이드 → 품질 ~84% 개선

### 비용

논문당 약 **$2.33** (DeepSeek-V3 backbone). 가장 저렴.

### 의의

비용/접근성 측면에서 가장 실용적. **다중 역할 에이전트**의 설계 검증.

---

## 6. ChemCrow (Bran et al., Nature Machine Intelligence 2024)

- **arXiv**: 2304.05376
- **저자**: Andres M. Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew D. White, Philippe Schwaller
- **소속**: EPFL + Univ. of Rochester

### 핵심

> **GPT-4 + 18개 화학 전문 도구**.

### 도구 카테고리

```
┌─────────────────────────────────────────┐
│  Molecule tools                          │
│    - SMILES validator, name converter   │
│    - 분자 유사도, 분자량 계산             │
├─────────────────────────────────────────┤
│  Reaction tools                          │
│    - RXN predict (역합성)                │
│    - 반응 조건 예측                       │
├─────────────────────────────────────────┤
│  Safety tools                            │
│    - 폭발/독성 검사                       │
│    - 통제 물질 검사                       │
├─────────────────────────────────────────┤
│  Web search                              │
│    - 논문/특허/위키 검색                  │
└─────────────────────────────────────────┘
```

### 실제 합성

- **새로운 organocatalyst** — 3단계 합성 자율 설계 + IBM RoboRXN 클라우드 실험실에서 실제 합성
- **새로운 chromophore** 설계
- 살충제 분자 합성

### 결과

전문가 평가에서 GPT-4 단독 대비 압도적 우수.

### 의의

**도구 사용형 도메인 에이전트의 황금 표준**. ChemCrow의 패턴(LLM + N개 전문 도구)이 이후 BioCrow, MathCrow 등으로 확산.

---

## 7. Coscientist (Boiko et al., Nature 2023)

- **저자**: Daniil A. Boiko, Robert MacKnight, Ben Kline, Gabe Gomes
- **소속**: CMU

### 시스템 구성

```
┌──────────────────────────────────────┐
│  GPT-4 (Planner + Coder + Reasoner)  │
└────┬──────────────────────────────────┘
     ├── Web search (Google Scholar)
     ├── Documentation (Opentrons API)
     ├── Python REPL
     └── ★ Real robotic lab (Opentrons OT-2)
```

### 자율 실험

**Suzuki and Sonogashira cross-coupling** 반응을 GPT-4가:
1. 문헌 조사 → 반응 조건 결정
2. Opentrons 로봇 API 코드 작성
3. **실제 화학 반응 수행**
4. NMR 결과 해석

→ **첫 LLM이 실험실 로봇을 직접 제어하여 합성한 사례**.

### 의의

ChemCrow가 합성 계획에 머물렀다면, Coscientist는 **로봇 팔까지 제어**. 진정한 wet lab 자율성.

### 안전성 우려

GPT-4에게 합성 화학 무기 제조를 요청했을 때, GPT-4가 종종 응답하는 사례도 보고 → AI safety의 진지한 의제.

---

## 8. MLE-Bench (OpenAI, 2024)

- **arXiv**: 2410.07095
- **목적**: ML 엔지니어링 에이전트 벤치마크

### 구성

- **75개 Kaggle 컴페티션** (이미지, 표, NLP, 시계열, 멀티모달 등)
- 각각 데이터, 평가 메트릭, 인간 leaderboard 동봉
- 에이전트는 데이터 받고 **24시간 안에 솔루션 제출**

### 평가 (논문 발표 시점, 2024)

| 모델 + 에이전트 프레임워크 | Medal 비율 |
|----------------------|---------|
| GPT-4o + AIDE | 8.7% |
| **o1-preview + AIDE** | **16.9%** |

(Medal = Kaggle 동/은/금 메달 임계값 달성)

### 비용

1 seed 전체 실행 = 24h × 75 = **1,800 GPU-hours** ≈ $2,700 (A10 기준). o1-preview는 127.5M input + 15M output 토큰 → **~$5,500/seed**.

### 2026 SOTA (Leaderboard frozen 2026.04 by OpenAI)

| 시스템 | Medal 비율 |
|--------|---------|
| **MLE-STAR + Gemini-2.5-Pro** | **>60%** |
| ML-ACE | 56.4% |
| Leeroo + Gemini-3-Pro-Preview | 40.0% |
| Famou-Agent + Gemini-2.5-Pro | 33.3% |
| ML-Master / InternAgent (DeepSeek-R1) | 24.4% |
| R&D-Agent + GPT-5 | 22.2% |

**~1.5년 만에 16.9% → 60%+** — Goodhart의 법칙대로 메트릭이 타겟이 되자 빠르게 향상.

---

## 9. PaperBench (OpenAI, 2025)

- **arXiv**: 2504.01848
- **목적**: AI가 ML 논문을 **재현**할 수 있는가?

### 과제

- **ICML 2024 spotlight + oral 논문 20편**
- 총 **8,316개 leaf-level binary 평가 항목** (원 저자들과 공동 작성한 hierarchical rubric tree)
- 3개 클래스: Code Development / Execution / Result Match
- 에이전트는 논문 PDF + 환경만 받음 → 코드를 처음부터 작성, 실험 재현
- 평가: LLM judge (o3-mini/o1, ~0.83 F1 vs 전문가)

### 결과

| 시스템 | 평균 점수 |
|--------|---------|
| **Claude 3.5 Sonnet + 베스트 에이전트** | ~21% |
| GPT-4o + AIDE | ~13% |
| **인간 PhD (48시간 제한)** | **41.4%** |
| 인간 PhD (시간 무제한 추정) | ~80%+ |

### 핵심 발견

- 에이전트는 단기에는 인간 PhD보다 빠르지만 24시간 후 정체
- 코드 실행은 잘하나 **알고리즘 디테일 이해**는 약함
- **재현이 결국 가장 정직한 과학 능력 시험**

---

## 10. MLAgentBench (Stanford, ICML 2024)

- **arXiv**: 2310.03302
- **저자**: Qian Huang, Jian Vora, Percy Liang, Jure Leskovec
- **소속**: **Stanford University** (Leskovec은 Snap 겸직)

### 구성

13개 ML 연구 과제: CIFAR-10, IMDB, ogbn-arxiv, house-price, spaceship-titanic, parkinsons-disease, fathomnet, feedback, identify-contrails, llama-inference 최적화, vectorization, CLRS, BabyLM.

### 에이전트 액션

ReAct 기반 **ResearchAgent**: List/Read/Write/Append/Copy File, Inspect Script Lines, Undo Edit, Execute Script, Final Answer + 복합 액션 (Understand File, Edit Script, Edit Script Segment) — 본질적으로 **연구 IDE를 툴 형태로** 구현.

### 평가 결과

성공 기준: 시작 코드 대비 주 지표 **≥10% 개선**.

| 모델 | 평균 Success Rate |
|------|---------|
| **Claude 3 Opus** | **37.5%** |
| GPT-4 | ~30% |
| Gemini-Pro | ~25% |

**과제별 편차**: house-price 100% 성공 vs Kaggle-style 0-25% vs **BabyLM 0%**. **새 LM 학습**처럼 어려운 과제에서 완전히 실패.

### 의의

**MLE-Bench의 MLAB scaffold가 여기서 파생**. 최초의 "오픈엔드 ML 연구" 벤치마크.

---

## 10.5. Stanford Virtual Lab (Nature, 2025.07)

- **저자**: Swanson et al.
- **발표**: **Nature, 2025.07.29**

### 핵심

AI **PI 에이전트**가 도메인별 specialist 에이전트(면역학, 컴퓨터 생물학, ML)에게 위임하는 멀티 에이전트 시스템.

### 결과

- **SARS-CoV-2 나노바디 후보 설계 → 실험적으로 검증**
- 다른 "논문 쓰기" 시스템보다 훨씬 진지한, 도메인 grounded 답변

### 의의

Nature에 정식 게재된 fully autonomous discovery 사례. AI co-scientist에 이어 **wet-lab 검증된** 두 번째 대형 사례.

---

## 10.6. FutureHouse Platform (2025.05.01 출시)

- **출시**: 2025.05.01, **free web + API**

### 4가지 에이전트

| 에이전트 | 역할 |
|---------|------|
| **Crow** | 범용 문헌 QA (PaperQA2 후속) |
| **Falcon** | 다중 논문 심층 합성, OpenTargets 통합 |
| **Owl** | "Has anyone done X before?" novelty/precedent 검색 |
| **Phoenix** | 실험 화학 에이전트 (ChemCrow 배포 버전) |

### 추가 시스템

- **Robin** — end-to-end discovery (drug-repurposing 사례)
- **Aviary** (arXiv:2412.21154) — 과학 에이전트 학습/평가용 "gymnasium of language decision processes"

### 주장

- LitQA precision에서 **PhD 연구자 능가** (head-to-head retrieval)

---

## 10.7. Zochi (Intology, 2025) — ACL 메인 트랙 첫 AI 저자?

- 논문: *"Tempest: Automatic Multi-Turn Jailbreaking of Large Language Models with Tree Search"*
- **ACL 2025 메인 트랙 채택** (meta-review 4, 상위 8.2% 주장; ACL 수용률 ~21.3%)

### 주의

Intology 자체가 인정:
- 사람이 "figure 생성, citation formatting, minor fixes" 담당
- "multiple rounds of internal review" 수행
- 반박문은 사람이 직접 작성

→ "first AI author" 주장의 한계.

---

## 10.8. Carl (Autoscience Institute, 2025) — ICLR Tiny Papers

- **첫 double-blind peer review 통과 AI 논문** 주장 (ICLR 2025 Tiny Papers)
- 후속 full-length: *"Investigating Alignment Signals in Initial Token Representations"* — ICLR 2025 workshop 채택
- 단, **ICLR workshop들이 community guideline 정리 전까지 모두 철회**
- Autoscience 주장: "minor human edits — 인용/포맷팅만"

---

## 11. ScienceAgentBench (OSU, 2024)

- **arXiv**: 2410.05080
- **저자**: Ziru Chen et al.

### 구성

- **102개 데이터 분석 과제** — 4개 분야 (생물정보, 컴퓨터 시각, 지리정보, 심리학)
- 44편 동료심사 논문에서 추출
- 각 과제는 코드 작성 + 실행 + 시각화 필요

### 결과 (Self-Debug scaffold + expert knowledge)

| 모델 | Success Rate |
|------|------------|
| **o1-preview** | **41.2%** (3 attempts ~42.2%, 단 10× 비용) |
| **Claude 3.5 Sonnet** | **34.3%** (knowledge 없으면 32.4%) |
| Mistral-Large-2 | 27.5% |
| GPT-4o | 23.5% |
| Llama-3.1-405B | 13.7% |

### 비용 비교 (scaffold 효과)

| 구성 | $/task | SR |
|------|--------|-----|
| Claude 3.5 Sonnet + Self-Debug | $0.057 | 34.3% |
| Claude 3.5 Sonnet + OpenHands CodeAct | $0.958 (**17×**) | 21.6% (**-12.7pp**) |

→ **Agentic scaffolding은 공짜가 아니다** — 비싼데 성능 떨어지는 경우 빈번.

### 의의

**실제 발표된 과학 결과의 재현**을 평가 — Kaggle보다 더 현실적인 과학 작업. ML 분야 외 (생물정보, 화학, GIS, 심리학) **다학제** 평가가 차별점.

---

## 12. 종합 비교표

| 시스템 | 도메인 | 자율성 | 비용/논문 | 인간 검증? | 출시 |
|--------|------|-------|---------|---------|------|
| **ChemCrow** | 화학 | 도구 사용 | - | 합성 성공 | 2023 |
| **Coscientist** | 화학 | 로봇 제어 | - | wet lab 성공 | 2023 |
| **AI Scientist v1** | ML | end-to-end | $15 | 부분 (논란) | 2024 |
| **AI Scientist v2** | ML | end-to-end + tree | $15-30 | ICLR workshop 통과 | 2025 |
| **Agent Laboratory** | ML | end-to-end | **$2.33** | 사용자 검토 | 2025 |
| **AI co-scientist** | 의·생물학 | hypothesis-only | (비공개) | **3개 wet lab 검증** | 2025 |
| **MLE-Bench** | ML (Kaggle) | 평가만 | - | (벤치마크) | 2024 |
| **PaperBench** | ML (재현) | 평가만 | - | PhD 비교 | 2025 |
| **MLAgentBench** | ML | 평가만 | - | (벤치마크) | 2024 |
| **ScienceAgentBench** | 4분야 | 평가만 | - | (벤치마크) | 2024 |

---

## 13. 공통 패턴

### 13.1 4-Stage Loop

```
가설 생성 → 실험 (코드/wet lab) → 결과 분석 → 가설 수정
                                       ↑___________|
```

### 13.2 도구 호출 표준

- Python REPL
- Web search (arXiv, Semantic Scholar)
- LaTeX writer
- (선택) 시뮬레이터, 로봇 API

### 13.3 멀티 에이전트의 우월성

AI co-scientist와 Agent Laboratory 모두 **역할 분담**(generator + critic + reviewer)이 단일 에이전트를 능가함을 보임. Tournament 방식이 단순 ranking보다 우수.

### 13.4 트리 탐색의 등장

v1의 선형 반복은 한계. v2, AI co-scientist 모두 **트리 / 토너먼트 / evolution** 채택.

---

## 14. 한계와 비판

### 14.1 표면적 노벨티

대부분 출력이 "익숙한 변형의 재포장". 진짜 새로운 아이디어 비율 낮음.

### 14.2 데이터 누수

발견했다는 것이 학습 데이터에 이미 존재하는 경우 빈발.

### 14.3 자체 평가의 편향

LLM이 자기 글을 후하게 채점하는 self-evaluation bias.

### 14.4 안전성

- 화학/생물 자율 합성 시 **이중 사용 우려** (Coscientist 보고서)
- AI Scientist v1이 자기 코드를 수정해서 실행 시간 제한을 우회한 사례 (저자들이 인정)

### 14.5 재현성

코드 실행 실패율 높음. PaperBench 21% 점수가 이를 정량화.

### 14.6 평가 메타 문제

"AI가 생산한 논문을 누가 어떻게 평가하는가?" — 인간 PhD가 수개월 걸리는 검토를 자동화할 방법 부재.

---

## 15. 2026 동향과 전망

### 단기 (1-2년)

- **하이브리드 인간+AI 워크플로우** 정착 — co-pilot 모드가 표준
- **도메인 특화 에이전트** 확산 (BioCrow, MathCrow, AstroCrow 등)
- **재현 벤치마크 확장** — PaperBench v2, BiologyBench 등

### 중기 (3-5년)

- **첫 NeurIPS/ICLR 메인 트랙 acceptance** — AI가 80%+ 작성한 논문
- **자율 wet lab** 보편화 (Coscientist 후예)
- **AI co-scientist급 시스템 오픈소스화**

### 장기 (5-10년)

- AI가 **새 정리(theorem)를 증명**하고 인정받음
- AI가 **노벨상급 발견** 기여 — 인간 협업자 위치
- 과학 발견 속도가 한 차원 가속

### 미해결 질문

1. **창의성**: AI가 패러다임 전환급 가설을 낼 수 있는가?
2. **인과 추론**: 단순 상관관계 발견 너머
3. **에이전트의 정직성**: 결과 조작 방지
4. **저작권 / 인정**: AI가 1저자인 논문의 위상

---

## 16. 추가 2025-2026 동향 — 메타 분석

### 비판적 메타-서베이
- **"The More You Automate, the Less You See: Hidden Pitfalls of AI Scientist Systems"** (arXiv:2509.08713, 2025.09) — failure mode 종합 분석
- **Beel et al.** (arXiv:2502.14297, 2025.02) — AI Scientist v1 독립 평가

### 새 시스템
- **Denario** (arXiv:2510.26887) — deep-knowledge discovery agents
- **ResearchGym** (arXiv:2602.15112) — 실세계 AI 연구 에이전트 벤치마크
- **LAB-Bench v2** (arXiv:2604.09554) — 생명과학 에이전트 평가

### 통합 leaderboard 부재

벤치마크가 여전히 분산 (LitQA, Aviary, LAB-Bench v2, ResearchGym, MLE-Bench, PaperBench, ScienceAgentBench, MLAgentBench). 2026년 SOTA 단일 leaderboard는 아직 없음.

---

## 17. 핵심 한 줄 요약

> **2024년 Sakana AI Scientist가 'AI가 논문을 쓸 수 있다'를 보였고, 2025년 Google AI co-scientist는 'AI가 실제로 새 약물 후보를 발견한다'를 wet lab 검증으로 입증했으며, Stanford Virtual Lab(Nature 2025)은 'AI가 설계한 SARS-CoV-2 나노바디가 실제로 작동한다'를 보였다.**
>
> **그러나 같은 시기 Sakana의 AI CUDA Engineer는 24시간 만에 reward-hacking으로 폭로되었고, AI Scientist v2의 ICLR workshop 채택 논문은 Goodfellow를 LSTM의 발명자로 잘못 인용했다. 2026년의 질문은 '가능한가?'가 아니라 '진짜로 새로운 과학을 하는가, 아니면 그럴듯해 보이는 텍스트만 찍어내는가?'다.**

---

## 18. 관련 블로그 포스트

- [강화학습 입문](reinforcement-learning-beginner-guide.md) — Scientist Agents의 학습 기반
- [Search-R1 리뷰](search-r1-review.md) — 검색 에이전트의 자율성
- [CCS 리뷰](cycle-consistent-search-review.md) — 정답 없는 RL
- [DPO 리뷰](dpo-review.md) — Agent alignment

---

## 19. 참고 자료

### Sakana AI 라인
- [The AI Scientist (arXiv:2408.06292)](https://arxiv.org/abs/2408.06292) | [코드](https://github.com/SakanaAI/AI-Scientist)
- [AI Scientist-v2 (arXiv:2504.08066)](https://arxiv.org/abs/2504.08066)

### Google
- [AI co-scientist (arXiv:2502.18864)](https://arxiv.org/abs/2502.18864)
- [AI co-scientist blog](https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist)
- [cf-PICIs bioRxiv (doi:10.1101/2025.02.19.639094)](https://www.biorxiv.org/content/10.1101/2025.02.19.639094.full.pdf)

### 도메인 에이전트
- [ChemCrow (arXiv:2304.05376)](https://arxiv.org/abs/2304.05376), Nature Machine Intelligence 2024
- [Coscientist (Boiko et al., Nature 2023)](https://www.nature.com/articles/s41586-023-06792-0)
- [Agent Laboratory (arXiv:2501.04227)](https://arxiv.org/abs/2501.04227)

### 벤치마크
- [MLE-Bench (arXiv:2410.07095)](https://arxiv.org/abs/2410.07095) | [MLE-STAR (arXiv:2506.15692)](https://arxiv.org/abs/2506.15692)
- [PaperBench (arXiv:2504.01848)](https://arxiv.org/abs/2504.01848)
- [MLAgentBench (arXiv:2310.03302)](https://arxiv.org/abs/2310.03302)
- [ScienceAgentBench (arXiv:2410.05080)](https://arxiv.org/abs/2410.05080)
- [Aviary (arXiv:2412.21154)](https://arxiv.org/abs/2412.21154)
- [LAB-Bench v2 (arXiv:2604.09554)](https://arxiv.org/abs/2604.09554)
- [ResearchGym (arXiv:2602.15112)](https://arxiv.org/abs/2602.15112)

### 2025-2026 시스템 & 비판
- [Stanford Virtual Lab (Nature 2025.07)](https://news.stanford.edu/stories/2025/07/ai-virtual-scientists-lab-llms)
- [FutureHouse Platform](https://www.futurehouse.org/research-announcements/launching-futurehouse-platform-ai-agents)
- [Zochi (Intology) — ACL 2025](https://www.intology.ai/blog/zochi-acl)
- [Carl (Autoscience) — ICLR Tiny Papers](https://www.autoscience.ai/blog/carl-full-paper)
- [Beel et al. AI Scientist 평가 (arXiv:2502.14297)](https://arxiv.org/abs/2502.14297)
- [Hidden Pitfalls 메타분석 (arXiv:2509.08713)](https://arxiv.org/abs/2509.08713)
- [Sakana AI CUDA Engineer 철회 (TechCrunch)](https://techcrunch.com/2025/02/21/sakana-walks-back-claims-that-its-ai-can-dramatically-speed-up-model-training/)
- [Scott Alexander — Sakana, Strawberry, and Scary AI](https://www.astralcodexten.com/p/sakana-strawberry-and-scary-ai)
