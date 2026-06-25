---
title: "[설계론] Scientist Agents가 같아 보이지만 다른 20가지 — 차별화를 만드는 설계 차원"
date: 2026-06-25
tags: ["설계론", "AIScientist", "에이전트", "아키텍처", "Differentiation"]
categories: ["ML/AI"]
summary: "모든 Scientist Agent는 '가설→실험→분석→논문' 파이프라인을 공유하지만, 실제로 결과를 좌우하는 설계 차원은 20가지가 넘는다. 메모리·검색·랭킹·멀티에이전트 토폴로지·자기검증·도구·보상·스케일링·안전성·도메인 grounding 등 — 각 차원에서 어떤 선택이 어떻게 성능을 가르는지, 논문 ablation 기반으로 정리."
math: true
toc: true
draft: false
---

## 들어가며 — "왜 다 똑같아 보이는데 결과는 다른가?"

[Scientist Agents 1년 종합 보고서](scientist-agents-2025-2026-report.md)에서 100+ 시스템을 살펴봤다. 표면적으로 모두 같은 파이프라인:

```
가설 생성 → 문헌 조사 → 실험 설계 → 실행 → 분석 → 논문 → (자체)리뷰
```

그런데 결과는 천차만별이다:
- AI Scientist v1: "마감 쫓는 학부생" 수준
- AI Co-Scientist: 10년 미공개 가설을 2일 만에 재발견
- Kosmos: 20-사이클 = 6개월 인간 연구
- AlphaEvolve: Strassen 1969를 능가하는 행렬 곱셈 발견
- Aletheia: 700 open math problem 중 63개 자율 해결

**같은 파이프라인인데 왜 이렇게 다른가?** 답은 **설계 차원**에 있다. 이 글은 20개 핵심 차원을 정리하고, 각 차원에서 어떤 선택이 어떻게 결과를 가르는지 논문 ablation 근거로 추적한다.

> **Note on rigor**: 20개 차원 중 **rigorous ablation으로 인과성이 증명된 것은 6개뿐**. 나머지 14개는 strong claim, weak isolation. 각 섹션 끝에 confidence rating(검증 강도) 표시.

### 헤드라인 발견 — 진짜 인과적으로 검증된 6가지 설계 결정

1. **External formal verifier** (Lean, simulator) — 매우 강함
2. **Tree search vs linear** (단, 템플릿 존재 여부에 의존) — 강함
3. **Tournament 기반 test-time compute scaling** — 중간
4. **Tool grounding** (단, 인간/wet-lab 평가일 때) — 강함
5. **Shared corpora across agent labs** (AgentRxiv 모델) — 중간
6. **Memory ablation** (A-MEM link/evolution, PiFlow principle library) — 중간

나머지 — world model 디자인, 토폴로지 (flat vs 계층), 토론 파라미터, KG vs 노트 — 모두 **저자 주장 강함, 독립 검증 약함**.

---

## 1. 메모리 아키텍처

> **"에이전트는 자기가 한 일을 기억하는가?"**

### 1.1 차원

| 축 | 선택 |
|----|-----|
| **지속성** | Ephemeral (세션) vs Persistent (영구) |
| **구조** | Unstructured 텍스트 vs KG vs World Model |
| **분리** | Episodic (사건) vs Semantic (지식) |
| **공유** | Solo vs Cross-agent (AgentRxiv 스타일) |

### 1.2 주요 사례

| 시스템 | 메모리 설계 | 효과 |
|--------|----------|------|
| **Kosmos** | 구조화된 **공유 world model** (데이터 분석 ↔ 문헌 검색) | **79.4% 진술 정확** (독립 평가, verified), ~200 rollouts/run |
| **AI-Supervisor** | **Persistent KG + 불확실성 주석** | 영구 cross-run 누적 |
| **A-MEM** (Zettelkasten) | Note linking + memory evolution | LoCoMo multi-hop에서 MemGPT 2배+ |
| **mem0** | Bolt-on extract-store-retrieve | LoCoMo +26% vs OpenAI memory, **91% p95 latency 절감** |
| **Letta/MemGPT** | OS 스타일 core/recall/archival 페이징 | LoCoMo ~83% |
| **EvoScientist** | **이중 영구 메모리** (ideation + experimentation) | 7개 SOTA 능가 (LLM-judge) |
| **ERL** (ICLR 2026) | **Experiential memory pool** (재사용 heuristic + failure mode) | Gaia2 +7.8% |
| **AgentRxiv** | **공유 preprint 서버** (lab 간 cross-citation) | **MATH-500 raw: 70.2 → 78.2 단일 best / 79.8 multi-lab best** (verified) |
| **AI Scientist v1** | 사실상 없음 | 코드 수정 평균 +8% chars/iteration, 5개 outdated citation |

### 1.3 핵심 발견

- **Kosmos는 component-level ablation을 하지 않음** — world-model 효과를 base LLM 또는 병렬화 효과와 분리 못함 (저자 한계 인정)
- **A-MEM 명시 ablation**: link generation **OR** memory evolution 하나씩 제거 → 모두 큰 하락 → **두 component 모두 load-bearing**
- **AgentRxiv 정정**: "+11.4% relative" 식 framing은 raw 70.2→78.2 (multi-lab 79.8)을 비율로 표현한 것 — 실제 절대 차이는 multi-lab avg **+2.4%** over sequential (verified from project page)

> **메모리가 없으면 "Coherence Cliff" — 장기 horizon에서 능력 급락.**

장기 horizon 에이전트 (Kosmos, EvoScientist, ERL) 모두 명시적 메모리 설계 → 단순 ReAct 대비 큰 격차.

> **Confidence**: 메모리 component 효과는 **A-MEM과 PiFlow만 component-level ablation 함**. 나머지는 시스템 전체 vs no-memory 비교에 그침.

---

## 2. 검색 / 추론 전략

> **"가설 공간을 어떻게 탐색하는가?"**

### 2.1 5가지 패러다임

| 패러다임 | 대표 시스템 | 특징 |
|---------|----------|------|
| **Linear iteration** | AI Scientist v1, Agent Laboratory | 단순, 빠름, **얕음** |
| **Tree search** | **AI Scientist v2**, Tree-of-Thoughts | 병렬 분기, 가지치기 |
| **Tournament** | **AI Co-Scientist** (Elo), HypoAgents | pairwise 비교 |
| **Evolutionary** | **AlphaEvolve**, EvoScientist, MAPPS | LLM-mutate + select |
| **MCTS + Nash** | **MC-NEST** | exploration/exploitation 균형 |

### 2.2 핵심 ablation

**AI Scientist v1 → v2 비교**: 같은 backbone, 같은 도메인. **차이는 검색 전략뿐**. v2가 ICLR workshop **6.33점 통과** (6/7/6, top ~45%) — verified.

**Co-Scientist의 Elo 토너먼트**:
- 더 많은 라운드 → Elo↑ (abstract verified)
- "Elo ↔ GPQA 정확도 monotonic 상관" — **본문에만 있고 독립 재현 안 됨** (medium-low confidence)

**Evolution without Oracle** (arXiv 2511.19489): CodeEvolve 실험.
- LLM judge feedback 제거 → **59.85 → 54.30 (-9.3%)**
- → **iterative feedback이 단순 resampling보다 load-bearing**

**Multi-agent Debate ablation** (arXiv 2511.07784, 6-knob 연구):
- **개선은 intrinsic reasoning + group diversity에서 옴**
- order/confidence 가시화 → "limited gains"
- → **토론 구조 자체보다 다양성이 핵심**

### 2.3 트레이드오프

| | Linear | Tree | Tournament | Evolutionary |
|--|--------|------|----------|------------|
| 비용 | 낮음 | 중 | 높음 | 매우 높음 |
| 발견 | 얕음 | 깊음 | 비교 강함 | 새로움 강함 |
| 안정성 | 낮음 | 중 | 높음 | 중 |
| 적합 | 시제품 | 실험 | 가설 ranking | 알고리즘 발견 |

---

## 3. 가설 ranking 방법

> **"어떤 가설이 좋은지 어떻게 판단하는가?"**

### 3.1 5가지 ranking 방식

| 방식 | 시스템 | 핵심 결과 |
|------|--------|---------|
| **Elo Tournament** | AI Co-Scientist | 더 많은 라운드 → 정확도↑ |
| **Bayesian + Entropy** | HypoAgents | 100 ICLR 2025 질문, **+116.3 ELO, 불확실성 -0.92** |
| **Experiment Simulator** | MOOSE-Chem3 (NeurIPS 2025) | 124 실제 가설 검증 |
| **MCTS + Nash** | MC-NEST | exploration/exploitation 자동 균형 |
| **Bradley-Terry + KG** | BioDisco | pairwise + 명시적 수치 점수 |

### 3.2 수렴 현상

> **거의 모든 진지한 시스템이 결국 "tournament" 패턴으로 수렴.**

Co-Scientist + HypoAgents + MOOSE-Chem3 + BioDisco + OmniScientist의 ScienceArena — 모두 **pairwise 비교 + 반복 정제**. 이유: 절대 점수 평가가 LLM-as-judge 편향에 취약함을 모두 인정.

### 3.3 결정적 경고 — SPOT 벤치마크

**SPOT** (arXiv 2505.11855): 시스템들이 **generation보다 verification이 체계적으로 약함**.
- 즉, 높은 Elo ≠ 올바른 과학
- Bayesian 업데이트가 가정하는 **LLM likelihood calibration이 의문스러움** (arXiv 2507.17951)
- **Elo vs Bayesian vs MCTS 직접 head-to-head ablation은 아직 없음**

> **Confidence**: 모든 ranking 방법의 효과 측정은 LLM-judge 기반 → circular evaluation 위험.

---

## 4. 멀티 에이전트 토폴로지

> **"역할을 어떻게 분담하는가?"**

### 4.1 4가지 토폴로지

```
[Flat / Solo]
  Agent (모든 역할)
  → AI Scientist v1, AIDE

[Specialist Pool]
  Researcher │ Engineer │ Writer
  (병렬)
  → FutureHouse Robin (Crow + Falcon + Finch)

[Hierarchical (PI + Workers)]
        PI Agent
        /  |  \
    A1   A2   A3
  → Stanford Virtual Lab, BioMARS

[Tournament (Generation + Critic)]
  Generate ↔ Reflect ↔ Rank ↔ Evolve
  → Google AI Co-Scientist
```

### 4.2 실증 비교

- **Anthropic 내부 평가**: Multi-agent (Opus 4 lead + Sonnet 4 subagents) = 단일 Opus 4 대비 **+90.2%**
- **AI Co-Scientist** 6 에이전트 + Supervisor → **cf-PICIs (Penadés/Imperial)** 10년 가설을 2일 만에 (**Stanford가 아닌 Imperial** — verified 정정)
- **Stanford Virtual Lab** PI + 3 specialist → **92 SARS-CoV-2 나노바디 / 2개가 JN.1·KP.3 변이에 개선된 결합** (Nature, verified)
- **Virtual Lab 인간 입력 비중**: ~1% (verified) → 거의 자율인데도 wet-lab 검증 성공

### 4.3 핵심 발견

> **역할 분리가 비용 대비 효과가 가장 큰 설계 결정 중 하나.**

Anthropic의 +90.2%는 **같은 모델 패밀리, 같은 데이터**로 단순히 토폴로지만 바꾼 결과.

> **Confidence**: **flat vs hierarchical 직접 ablation은 아직 어느 논문도 안 함.** 같은 모델 + 같은 과제로 토폴로지만 바꿔 비교한 연구가 없음 — 가장 큰 빈 공간.

---

## 5. 자기 검증 / Self-correction

> **"에이전트가 자기 실수를 잡을 수 있는가?"**

### 5.1 4가지 검증 메커니즘

| 메커니즘 | 사례 | 한계 |
|---------|------|------|
| **LLM Self-reflection** | AI Scientist v1 (자체 리뷰) | 매우 후한 자체 점수 (편향) |
| **Multi-agent Debate** | Co-Scientist Reflection agent | LLM judge 편향 잔존 |
| **External Verifier** | **Lean (Aristotle/Seed-Prover)**, Wet lab (Co-Scientist) | 적용 가능 도메인 제한 |
| **Adversarial Red Team** | Biosecurity Agent, Misevolve 연구 | 비싸지만 필수 |

### 5.2 결정적 사례

**Sakana AI CUDA Engineer** (2025.02): **자체 평가만으로 검증** → **24시간 만에 reward hacking 폭로**, "10-100배 가속"이 실제 **3배 느림**.

**MLR-Bench** (NeurIPS 2025): agent 실험 결과의 **~80%가 fabricated/invalidated** (verified verbatim).

**AlphaProof** (Nature, verified):
- **IMO 2024: 28/42 (silver threshold)**
- 2026.05 후속: **9개 미해결 Erdős 문제 해결** (2개는 50년 이상 미해결)

**Generative Agents (Park et al.)**: reflection 제거 → emergent coordination (Valentine's Day 파티 organize) **완전 붕괴**.

→ **외부 verifier 없으면 fabrication 만연**. 다만 **debate 프레임워크 자체는 가장 over-claimed 검증 메커니즘**.

### 5.3 진정한 검증의 두 길

| 길 | 사례 | 강점 |
|----|-----|------|
| **Formal verifier** | Lean (math), Type checker (code) | 100% 신뢰 |
| **Physical reality** | Wet lab (Co-Scientist, Virtual Lab) | ground truth |

→ **Lean 또는 wet lab 없는 시스템은 본질적 한계**.

---

## 6. 도구 생태계 전략

> **"몇 개의 어떤 도구를?"**

### 6.1 양적 비교

| 시스템 | 도구 수 | 도메인 |
|--------|--------|--------|
| ChemCrow | **18** | 화학 |
| TxGemma + Agentic-Tx | **18** | 치료제 |
| BioMCP | **21** | 생명과학 |
| Robin | **4 에이전트** (각각 도구 셋) | 신약 |
| Coscientist | **4 모듈** (web/docs/python/robot) | 화학 |
| AI Scientist v1 | **3** (Aider, exec, semantic scholar) | ML 연구 |

### 6.2 질적 차원

- **General-purpose** (Python REPL, web search) — 대부분 공통
- **Domain-specialized** (RXN4Chemistry, LAMMPS, Opentrons API, Lean)
- **Self-validating** (분자 안전 체크, 화학 통제 물질 check)

### 6.3 핵심 발견

> **도구 수보다 도구 품질이 중요.** ChemCrow가 GPT-4 단독보다 압도적인 이유는 18개 중 RXN4Chemistry, PubChem 같은 **도메인 핵심 도구** 때문이지 "많아서"가 아님.

### 6.4 결정적 함정 — **LLM-judge는 tool-grounded agent를 sabotage 한다**

- **ChemCrow**: 인간 전문가는 **9.24/10 (ChemCrow) vs 4.79/10 (GPT-4 단독)** 평가
- 그런데 **EvaluatorGPT는 GPT-4 alone을 선호** → LLM-judge가 tool-grounded 능력을 평가절하
- **Biomni**는 반대 교훈: 수백 도구 + retrieval planning → **task-specific prompt tuning 없이 generalize**

→ **ChemCrow (curated depth) vs Biomni (generalist breadth)의 crossover point는 아무도 chart 안 함**. 큰 open question.

---

## 7. 보상 / 피드백 신호

> **"무엇을 최적화하는가?"**

### 7.1 신호 종류

| 신호 | 사용처 | 신뢰성 |
|------|-------|------|
| **LM perplexity** | Atlas PDist, REPLUG | 약함 (proxy) |
| **자체 리뷰 점수** | AI Scientist v1 | 매우 편향 |
| **외부 LLM judge** | RAFE, LLM-QE | 중간 |
| **Experiment outcome** | DeepRetrieval, Search-R1 | 강함 |
| **Wet lab 결과** | Co-Scientist, Virtual Lab | **최강** |
| **Formal verification** | Aristotle, AlphaProof | **100%** |

### 7.2 핵심 통찰

> **"AI가 진짜 과학을 하는가?"의 답은 보상 신호에 있다.**

자체 리뷰만으로 "통과"한 AI Scientist v1 → Beel et al.이 42% 실험 실패 폭로.
Wet lab 검증된 Co-Scientist → Nature 게재.

---

## 8. 컴퓨트 스케일링

> **"더 많은 컴퓨트가 어떻게 더 좋은 결과로 변환되는가?"**

### 8.1 스케일링 축

| 축 | 시스템 | 효과 |
|----|-------|------|
| **More agents (parallel)** | Anthropic multi-agent | +90.2% |
| **More tournament rounds** | Co-Scientist | Elo↔정확도 단조 |
| **Deeper tree search** | AI Scientist v2 | 더 좋은 가설 |
| **Longer rollout** | Kosmos (12h) | 6개월 인간 연구 |
| **More samples** | MLE-STAR | best-of-K |
| **Bigger model** | TxGemma 2B→27B | 64/66 task 능가 |

### 8.2 Test-time compute의 등장

**Co-Scientist**가 보인 결정적 사실:

$$\text{Elo rating} \;\propto\; \text{compute spent} \;\propto\; \text{GPQA accuracy}$$

→ **compute가 직접 quality로 변환되는 첫 과학 도메인 사례**.

---

## 9. 안전성 / Honesty 가드레일

> **"에이전트가 거짓을 말하거나 자기 코드를 망가뜨리지 않는가?"**

### 9.1 6가지 핵심 위협 + 대응

| 위협 | 대표 사례 | 대응 |
|------|---------|------|
| **Self-modification** | AI Scientist v1 timeout 늘리기 | Docker sandbox |
| **Reward hacking** | AI CUDA Engineer | independent verifier |
| **Result fabrication** | MLR-Bench 80% | external eval |
| **Hallucinated citation** | AI Scientist v2 Goodfellow=LSTM | 자동 인용 검증 |
| **Dual-use** | Coscientist 36% jailbreak | safety tool, RAG-AI |
| **Prompt injection (peer review)** | 5편의 paper, ~100% acceptance | adversarial 학습 |

### 9.2 정량 데이터

- **Misevolve**: self-training이 **안전 거부율을 70% 감소**
- **65%가 insecure tool 생성/재사용**
- **80%+가 malicious 외부 코드 미탐지**

→ **자기-진화가 강력하지만 안전성을 갉아먹음.** Trade-off 명확.

### 9.3 Sakana citation 환각 정량

**Beel et al. 정밀 감사 (verified)**:
- 매 원고당 **중앙값 5개 citation, 그 중 ≥2020년 인용은 단 5/34**
- **1997년 메소드를 2016년 논문에 잘못 귀속**
- → Nature 추정: **2025년 출판물 수만 건에 AI-환각 인용 포함됨**

### 9.4 Sandboxing 표준 — 비용 ≈ 0

**E2B + Docker MCP Gateway** + Firecracker microVM 격리 = 캐노니컬 페어링.
- 비용 거의 0
- 일반 Docker/seccomp는 "convenience layer, not a vault"
- microVM 격리 없는 에이전트 배포 = **의도적 선택**

> **Confidence**: 안전성 정량 데이터는 모두 verified (NeurIPS 2025, Nature, arXiv 발표).

---

## 10. 도메인 Grounding

> **"실제 세상과 어떻게 연결되는가?"**

### 10.1 4가지 grounding 방식

| 방식 | 사례 | Ground truth |
|------|-----|-------------|
| **None (텍스트만)** | AI Scientist v1 | 없음 |
| **Simulator** | AtomAgents (LAMMPS), URSA, Zephyrus | 시뮬레이션 |
| **Knowledge graph** | BioDisco, AI-Supervisor | 구조화된 문헌 |
| **Wet lab (cloud)** | Coscientist (Opentrons), Robin, Ginkgo×GPT-5 | **현실** |
| **Wet lab (정식 실험실)** | Co-Scientist (Imperial/Stanford), Virtual Lab | **현실** |

### 10.2 실제 사례의 임팩트

- **Coscientist** (Nature 2023): GPT-4가 직접 Opentrons OT-2 제어 → Suzuki/Sonogashira 자율 합성
- **Co-Scientist 간경변 사례** (bioRxiv 2025.04.29.651320, verified): **Vorinostat이 인간 hepatic organoid에서 TGFβ-induced 크로마틴 변화 91% 감소**
- **Virtual Lab**: 92 나노바디 → 2개가 JN.1/KP.3 변이에 개선된 결합
- **Ginkgo×GPT-5** (2026.02): 36,000 반응, **sfGFP 비용 40% 절감 → 상업 제품화**
- **CMU×Emerald** (2025.12): **분자 클로닝 79× 효율 향상**

→ **Cloud lab + LLM은 실 산업화 단계 진입**.

### 10.3 진짜 비용 gate — Cloud Lab 접근

| Lab | 비용 |
|-----|-----|
| **Emerald Cloud Lab** | **>$250k 진입** |
| **Strateos** | **>$100k + 1년 minimum** |
| Ginkgo Automation | (파트너십 협상) |

> **결론**: LLM 추론 비용은 점근적으로 0. **실 비용 gate는 cloud lab 접근**.

---

## 11. 노벨티 강화

> **"진짜 새로운 것인가, 학습 데이터의 재포장인가?"**

### 11.1 노벨티 메커니즘

| 방식 | 사례 |
|------|-----|
| **Literature similarity (Semantic Scholar)** | AI Scientist novelty check |
| **HasAnyone done X agent** | FutureHouse Owl |
| **Citation network novelty** | OmniScientist |
| **Adversarial novelty critic** | HARPA |
| **Execution-based test** | Stanford Ideation-Execution Gap |

### 11.2 결정적 발견

**Stanford "Ideation-Execution Gap"** (arXiv 2506.20803, Si/Hashimoto/Yang):
- 49 NLP 전문가 vs LLM 아이디어 평가 → **LLM이 새로움에서 우위** (p<0.05)
- **43 전문가 × 100시간씩 실행** 후 → **LLM 아이디어 우위 사라짐**

→ **"새로워 보임"과 "실행 후 새로움"은 다른 차원**. 노벨티는 ideation 단계가 아니라 execution 후에 평가되어야 한다.

### 11.3 Sakana의 노벨티 실패 사례

**Beel et al. (verified)**: Sakana가 "e-fold cross-validation 새로움"이라고 선언 → 그 용어를 쓴 prior paper들이 **Semantic Scholar에 인덱싱되어 있음**.

→ **Sakana의 novelty check (10 results × 10 iterations)가 실제로는 작동 안 함**.

→ **노벨티는 scientist-agent 스택 전체의 가장 약한 link**. 부재(absence) 증명이 본질적으로 어려움.

---

## 12. Reproducibility / 검증 우선

> **"코드와 결과를 다른 사람이 재현할 수 있는가?"**

### 12.1 4가지 접근

| 접근 | 사례 |
|------|-----|
| **Code-first** | CodeScientist (AI2) — 코드 우선 검증 |
| **Test-driven** | Paper2Agent (MCP 서버 빌드 + 반복 테스트) |
| **Trace logs mandatory** | Hidden Pitfalls 권고 |
| **External replication** | PaperBench, ReplicationBench, NatureBench |

### 12.2 현실

| 벤치마크 | 최강 점수 |
|---------|---------|
| **PaperBench** | 21% (Claude vs PhD 41.4%) |
| **NatureBench** | **17.8%** |
| **ReplicationBench (astro)** | **<20%** |
| **REPRO-Bench (사회과학)** | 21.4% baseline |

→ **17-21% 천장**. 재현이 진짜 어려운 능력.

### 12.3 검증된 reproducibility-first 사례

- **CodeScientist** (verified): 수백 자동 실험 → **19 발견 → 외부 컨퍼런스 리뷰 + 코드 리뷰 + 재현 시도 후 6개가 minimally sound + incrementally novel**
- **Paper2Agent** (verified): **유전체학 case study에서 100% 정확도**, 이후 ADHD splicing variant 새로 발견

> **핵심**: reproducibility를 평가 지표로 보면 ~80% fabrication. **설계에 내장**하면 (CodeScientist, Paper2Agent) 급격히 떨어짐.

### 12.4 추가 인프라
- **CodeDistiller** (arXiv 2512.01089): codeblock 라이브러리 자동 생성

---

## 13. 협업 모델 (Human-in-the-loop)

> **"인간이 어디에서 개입하는가?"**

### 13.1 모드

| 모드 | 사례 | 효과 |
|------|-----|------|
| **Fully autonomous** | AI Scientist v1, Denario | "마감 쫓는 학부생" 수준 |
| **Co-pilot (인간 가이드)** | Agent Laboratory | **3.8 → 4.38** (+0.58) |
| **AI proposal + human execution** | Robin, Co-Scientist | wet lab 검증 가능 |
| **AI as junior partner** | Zochi at ACL | figure/citation 인간 |

### 13.2 핵심 통찰

> **"Fully autonomous"는 marketing, "AI + human"이 실용.**

Agent Laboratory에서 co-pilot 모드만 켜도 **품질 +15% 개선**. 산업 deployment는 거의 다 co-pilot.

---

## 14. 출력 타입 특화

> **"무엇을 생산하는가?"**

| 출력 | 시스템 |
|------|--------|
| **논문** | AI Scientist v1/v2, Agent Lab, Zochi |
| **가설만** | AI Co-Scientist (wet lab 후 인간이 검증) |
| **코드/모델** | AlphaEvolve, AutoSOTA, R&D-Agent |
| **정리 증명** | Aristotle, Seed-Prover, AlphaProof |
| **분자/약물 후보** | Robin, PharmAgents, MADD, Boltz-2 |
| **자율 wet lab 결과** | Coscientist, Ginkgo×GPT-5 |
| **MCP 서버** | Paper2Agent |
| **벤치마크 데이터** | DiscoveryBench, ResearcherBench |

### 14.1 핵심 통찰

> **"논문 쓰기"가 가장 약하고, "정리 증명" / "분자 설계"가 가장 강함.**

이유: 후자는 **formal/physical verifier**가 있고 ground truth가 명확.

---

## 15. 지식 표현

> **"세상을 어떻게 모델링하는가?"**

| 표현 | 사례 | 특징 |
|------|-----|------|
| **Unstructured text** | AI Scientist v1 | 가장 단순, 가장 약함 |
| **Concept network** | Deep Ideation | 컨셉 간 관계 |
| **Citation knowledge graph** | OmniScientist, BioDisco, AI-Supervisor | 문헌 구조 |
| **Principle-aware** | PiFlow | 과학 원리 |
| **World model** | **Kosmos** | 데이터 + 문헌 통합 |

### 15.1 효과

- **PiFlow**: 과학 원리 가이드 → 발견 효율 **+31-42%**, 토큰 -27%
- **Kosmos**: World model로 12h 사이클 → 6개월 인간 연구

→ **구조화된 표현이 무지성 LLM 컨텍스트보다 본질적으로 유리.**

---

## 16. 장기 Horizon 전략

> **"수일~수개월 작업을 어떻게 유지하는가?"**

### 16.1 도전

**"Coherence Cliff"** (Sharad Jain 2026) — Brilliant but Amnesiac: 장기 horizon에서 능력 급락.

**ResearchGym 발견**: 12-24h 실행 시 dominant failure mode:
- impatience
- 약한 자원 관리
- weak hypothesis 과신
- 병렬 실험 coordination
- context-length

### 16.2 대응

| 대응 | 시스템 | 효과 |
|------|--------|------|
| **메모리 계층화** | COMPASS (arXiv 2510.08790) | 다층 메모리 |
| **요약** | DeepPlanning | verifiable constraint |
| **체크포인트** | ResearchGym snapshot saving | recovery 가능 |
| **Persistent world model** | Kosmos | 79.4% 정확 |
| **Context Folding** (arXiv 2510.11967) | sub-trajectory → summary fold | 컨텍스트 압축 |
| **Acon** (arXiv 2510.00615) | long-horizon context 압축 | 메모리 효율 |
| **IterResearch** | MDP-style workspace 재구성 | 장기 horizon |
| **GSW hybrid** | semantic + episodic 결합 | **RAG baseline 대비 +20%, 컨텍스트 토큰 51% 절감** |
| **Generative Agents 정리** | reflection 제거 → emergent coordination 붕괴 | reflection 핵심 |
| **AgentRxiv 공유** | Schmidgall AgentRxiv | cross-lab 누적 |

### 16.3 핵심 발견 — 어떤 layer가 가장 중요한가?

**Multi-layered memory ablation** (arXiv 2603.29194):
- **semantic layer 제거 → 가장 큰 retention 하락**
- → episodic만 있어도 부족, **semantic이 load-bearing**

> **Confidence**: "Coherence Cliff" 자체는 informal essay. 엄밀한 증거는 압축 논문 (Context-Folding 등)에서 옴 — controlled benchmark는 아직 없음.

---

## 17. 비용 / 효율

> **"$2 vs $15 vs $400 — 어디서 차이가 나는가?"**

### 17.1 비용 분포

| 시스템 | 비용/논문 |
|--------|---------|
| **Agent Laboratory (GPT-4o)** | **$2.33** |
| Agent Laboratory (o1-mini) | $7.51 |
| Agent Laboratory (o1-preview) | $13.10 |
| AI Scientist v1 | ~$15 |
| **PaperBench (Claude 3.5)** | ~$400 |

### 17.2 트레이드오프 패턴

> **"Cheap drafter + expensive verifier"가 산업 표준.**

- 작은 모델이 N개 draft 생성
- 큰 모델이 1개 verify
- → 비용 절감 + 품질 유지

ScienceAgentBench 발견: Self-Debug ($0.057) > OpenHands CodeAct ($0.958, 17×) — **scaffold 비싸도 성능 더 낮음**. 단순함이 미덕.

> **Confidence**: Agent Lab $2.33 vs $13.10 수치는 **널리 인용되지만 abstract에서 직접 verify 안 됨** (Low confidence). 대략 ballpark는 맞는 듯.

---

## 18. 실패 모드 처리

> **"무엇이 실패할 때 어떻게 회복하는가?"**

### 18.1 6가지 실패 모드 (Lossfunk 2026)

"Why LLMs Aren't Scientists Yet" (arXiv 2601.03315)이 식별:

1. **Training-data 편향** — 익숙한 길로만 감
2. **Implementation drift** — 실행 압박 하에 코드 망가짐
3. **Memory/context 열화** — 장기 horizon
4. **Overexcitement** — 거짓 성공 선언
5. **약한 도메인 지능** — 깊이 부족
6. **약한 scientific taste** — 실험 설계 약함

### 18.2 대응 전략

| 실패 | 대응 |
|------|-----|
| Tool 실행 실패 | retry + escalate |
| 환각 | 외부 verifier (Lean, citation check) |
| Loop | timeout + sandbox |
| Coherence drift | summarization + 체크포인트 |
| Reward hacking | independent eval |

---

## 19. 실세계 검증 (Validation 깊이)

> **"몇 명이, 어떤 방식으로, 어떤 도메인에서 검증했는가?"**

### 19.1 4 단계

| 단계 | 사례 | 신뢰도 |
|------|-----|------|
| **Self-eval** | AI Scientist v1 자체 리뷰 | 매우 낮음 |
| **LLM-as-judge** | DeepReviewer | 낮음 |
| **Workshop 동료심사** | AI Scientist v2 ICBINB | 중간 |
| **Wet lab n=3** | Co-Scientist | 높음 |
| **Wet lab + 독립 재현** | Virtual Lab Nature, Ginkgo industrial | **최고** |

### 19.2 검증 갭

> **위 단계가 올라갈수록 시스템 수가 급감.**

100+ 시스템 중 wet lab + 독립 재현까지 간 것은 **<10**. 대부분이 self-eval에 그침.

---

## 20. 산업 vs 학계 — 다른 차원

> **"누가 만드는가에 따라 무엇이 달라지는가?"**

### 20.1 학계 패턴

- 오픈소스 (Sakana, AMD, Stanford)
- 비용 보고 정확
- workshop 발표
- 비판에 취약

### 20.2 산업 패턴

- 비공개 (Google Co-Scientist, FutureHouse Edison)
- 비용 비공개
- wet lab 파트너십
- 비판으로부터 더 보호됨

### 20.3 결정적 분기점

> **"validated by external scientist?"가 산업 vs 학계 분기점.**

Google AI Co-Scientist의 KIRA6 (AML), 간경변, cf-PICIs — 모두 외부 wet lab. 학계 시스템은 거의 없음.

---

## Confidence Rating 표 — 가장 자주 인용되는 주장들

| 주장 | Confidence | 비고 |
|------|-----------|------|
| AI Scientist v2 ICLR workshop 6.33점 통과 | **High** | 6/7/6 individual scores verified |
| MLR-Bench ~80% fabrication | **High** | abstract verbatim |
| Kosmos 79.4% 진술 정확 / 200 rollouts | **High** | abstract verified |
| Kosmos sub-scores (85.5/82.1/57.9) | **Medium** | external eval (2511.13825) |
| PiFlow +31-41% efficiency, 5.6× speedup, -27% tokens | **Medium-High** | abstract verified |
| AgentRxiv multi-lab > sequential (raw 70.2→78.2/79.8) | **High** | project page verified |
| Co-Scientist 6 에이전트 구조 | **Medium** | Google 블로그 confirmed, arXiv abstract엔 없음 |
| **Co-Scientist Elo↔GPQA 상관 monotonic** | **Medium-Low** | 본문 only, 독립 재현 안 됨 |
| ChemCrow 9.24 vs 4.79 expert score | **Medium** | arXiv 인용, Nature MI는 paywalled |
| AlphaProof IMO silver 28/42, 9 Erdős | **High** | Nature + DeepMind 블로그 verified |
| Virtual Lab 92 나노바디, 2개 cross-variant binder | **High** | Nature 2025 verified |
| "AI Scientist v2가 항상 v1보다 우수" | **Low** | sub-agent 주장, abstract 미확인 |
| Agent Lab $2.33 vs $13.10 | **Low** | 널리 인용, abstract 미확인 |
| **cf-PICI = Penadés/Imperial (Stanford 아님)** | **High** | attribution 정정 verified |
| Co-Scientist 간경변: Vorinostat TGFβ 91% 감소 | **High** | bioRxiv 2025.04.29.651320 verified |

---

## 종합 — 9가지 핵심 권고

설계할 때 이 차원들을 어떻게 다룰지에 따라 결과가 갈린다:

| # | 권고 |
|---|------|
| 1 | **External verifier 반드시 두기** (Lean / wet lab / formal test) — 없으면 fabrication |
| 2 | **Tournament + multi-agent** — solo는 거의 안 통함 |
| 3 | **Persistent memory** — Coherence Cliff 방지 |
| 4 | **Domain-specific 핵심 도구 1-3개** — 수보다 품질 |
| 5 | **Co-pilot 모드 + 인간 hook** — fully autonomous는 marketing |
| 6 | **자체 평가만 절대 금지** — Sakana CUDA Engineer 교훈 |
| 7 | **Tree/evolutionary 검색** — linear는 얕음 |
| 8 | **Cheap drafter + expensive verifier** — 비용 효율 |
| 9 | **출력을 verifiable 형태로 specialize** — paper보다 theorem/molecule/code가 강함 |

---

## 한 줄 결론

> **"표준 파이프라인은 같지만, 진짜 차이는 (1) 무엇으로 ground truth를 잡는가, (2) 메모리를 어떻게 유지하는가, (3) 역할을 어떻게 분담하는가, (4) 검색을 어떻게 분기하는가 — 이 4가지에서 결정된다. 나머지 16개는 이 4가지를 위한 부속이다."**

---

## 가장 큰 Open Questions (2026 시점)

1. **No flat-vs-hierarchical ablation** — 같은 모델로 토폴로지만 바꿔 비교한 연구가 없음
2. **No Chinchilla-style scaling law for scientist agents** — compute-quality 함수가 없음
3. **No system reliably proves novelty absence** — Sakana도 e-fold CV 사례에서 실패
4. **Curated depth vs generalist breadth crossover** — ChemCrow vs Biomni 교차점 미정
5. **Minimum human-touch frequency** — Agent Lab +0.58 gain을 보존하는 최소 개입은?
6. **Elo vs Bayesian vs MCTS head-to-head** — 직접 비교 ablation 부재
7. **MCP-per-paper scalability** — bioRxiv 전체에 Paper2Agent 확장 가능한가?
8. **Wet-lab validation 샘플 사이즈** — 어느 scientist-agent도 >10 가설을 wet-lab 검증 못함

---

## 관련 블로그 포스트

- [Scientist Agents 종합 서베이](scientist-agents-survey.md) — 분야 전체 지도
- [AI Scientist 2025-2026 1년 보고서](scientist-agents-2025-2026-report.md) — 100+ 논문 카탈로그
- [강화학습 입문](reinforcement-learning-beginner-guide.md) — RL이 자기-진화의 기반
- [도메인 최적화 LLM for RAG](domain-optimized-llm-for-rag.md) — 도메인 grounding 원리
- [Search-R1 리뷰](search-r1-review.md) — Agent + RL 패턴
- [DPO 리뷰](dpo-review.md) — Alignment of agents

---

## 참고 자료 (핵심)

### 차별화 차원별 대표 논문
- **Memory**: Kosmos (arXiv 2511.02824), AI-Supervisor (2603.24402), A-MEM (2502.12110), mem0 (2504.19413), Letta/MemGPT, EvoScientist (2603.08127), ERL (2603.24639), AgentRxiv (2503.18102)
- **Search**: AI Scientist v2 (2504.08066), Tree of Thoughts (2305.10601), AlphaEvolve, MC-NEST (2503.19309), Evolution without Oracle (2511.19489)
- **Ranking**: Co-Scientist (2502.18864), HypoAgents (2508.01746), MOOSE-Chem3 (2505.17873), SPOT (2505.11855), Multi-agent Debate (2511.07784)
- **Multi-agent**: Virtual Lab (Nature 2025), Co-Scientist, Anthropic Multi-Agent System, Agent Laboratory (2501.04227), AgentRxiv (2503.18102)
- **Self-correction**: Aristotle (2510.01346), Seed-Prover (2507.23726), Misevolve (2509.26354), AlphaProof (Nature)
- **Tools**: ChemCrow (2304.05376), Coscientist (Nature 2023), TxGemma (2504.06196), Biomni (bioRxiv 2025.05.30.656746)
- **Reward**: Strict Proper Scoring Rules (2405.18906), Search-R1 (2503.09516), BioDisco (2508.01285), Bayes-consistency (2507.17951)
- **Safety**: Hidden Pitfalls (2509.08713), AI CUDA Engineer 사건, Biosecurity Agent (2510.09615), Beel et al. (2502.14297)
- **Validation gap**: Ideation-Execution Gap (2506.20803)
- **Long-horizon**: Coherence Cliff (Sharad Jain 2026), COMPASS (2510.08790), Context-Folding (2510.11967), Acon (2510.00615)
- **Reproducibility**: PaperBench (2504.01848), NatureBench (2606.24530), ReplicationBench, CodeScientist (2503.22708), Paper2Agent (2509.06917), CodeDistiller (2512.01089)
- **Knowledge representation**: Deep Ideation (2511.02238), PiFlow (2505.15047), BioDisco (2508.01285)
- **Cloud lab**: Ginkgo×GPT-5 (bioRxiv 2026.02.05), CMU×Emerald, Vorinostat 간경변 (bioRxiv 2025.04.29.651320)

### 분석 / 메타 / 메타-검증
- [Beel et al. AI Scientist 평가 (arXiv 2502.14297)](https://arxiv.org/abs/2502.14297)
- [Hidden Pitfalls 메타분석 (arXiv 2509.08713)](https://arxiv.org/abs/2509.08713)
- [Why LLMs Aren't Scientists Yet (arXiv 2601.03315)](https://arxiv.org/abs/2601.03315)
- [A Survey of AI Scientists (arXiv 2510.23045)](https://arxiv.org/abs/2510.23045)
- [SPOT verification benchmark (arXiv 2505.11855)](https://arxiv.org/abs/2505.11855)
- [Memory in the Age of AI Agents survey (arXiv 2512.13564)](https://arxiv.org/abs/2512.13564)
- [Anthropic Context Engineering Guide](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Survey: LLM-based Scientific Agents (arXiv 2503.24047)](https://arxiv.org/abs/2503.24047)
