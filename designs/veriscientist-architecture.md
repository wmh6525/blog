# VeriScientist — 검증 우선 Scientist Agent 아키텍처 설계서

> **One-liner**: 100+ 시스템 리서치에서 도출된 "rigorous ablation 6가지"를 모두 만족하는 verifier-first Scientist Agent 청사진. 구현 가이드가 아니라 **설계 청사진(blueprint)**.

작성: 2026-06-26
상태: Draft (구현은 별도 진행)
근거 리서치:
- [Scientist Agents 1년 종합 보고서](/content/posts/scientist-agents-2025-2026-report.md)
- [Scientist Agents 차별화 설계 차원 20가지](/content/posts/scientist-agents-design-dimensions.md)

---

## 0. 설계 철학 — 5가지 원칙

| # | 원칙 | 출처 |
|---|------|------|
| 1 | **Verifier-First** — 자체 평가 절대 금지, 모든 claim은 외부 verifier 통과 필수 | Sakana CUDA Engineer 폭로, MLR-Bench 80% fabrication |
| 2 | **Tournament-based, never absolute** — 가설은 항상 pairwise로 평가 | AI Co-Scientist Elo, HypoAgents +116.3 Elo |
| 3 | **Layered memory, not flat context** — episodic + semantic + procedural 분리 | A-MEM ablation, semantic layer load-bearing |
| 4 | **Code-first reproducibility** — 모든 결과는 컨테이너화된 코드로 재현 가능 | CodeScientist 6/19, Paper2Agent 100% |
| 5 | **Co-pilot by default** — 인간 hook 항상 활성, fully-autonomous 모드는 명시적 flag | Agent Lab +0.58 quality gain |

### 명시적 NON-목표

- "논문 자동 생성" (저자가 가장 약함, fabrication 위험)
- Fully autonomous research (Marketing only)
- 모든 도메인 일반화 (도메인별 verifier가 다름)

---

## 1. 시스템 개요

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       VeriScientist System Boundary                       │
│                                                                           │
│   ┌─────────────┐      ┌──────────────────┐      ┌──────────────────┐    │
│   │  Human PI   │◄────►│   Orchestrator   │◄────►│  Memory Subsys.  │    │
│   │  (co-pilot) │      │   (Planner/PI)   │      │ (3-tier + KG)    │    │
│   └─────────────┘      └─────────┬────────┘      └──────────────────┘    │
│                                  │                                        │
│        ┌─────────────────────────┼─────────────────────────┐             │
│        ▼                         ▼                         ▼             │
│   ┌──────────┐            ┌──────────────┐         ┌──────────────┐      │
│   │ Hypothes.│            │  Tournament  │         │   Executor   │      │
│   │ Generator│◄──────────►│   Ranking    │◄───────►│   (Code +    │      │
│   │  (×N)    │   Elo      │   (Critic +  │ candid. │   Lab API)   │      │
│   └──────────┘            │   Reflect.)  │         └──────┬───────┘      │
│                           └──────┬───────┘                │              │
│                                  │ winner                 │ artifact     │
│                                  ▼                        ▼              │
│                           ┌──────────────────────────────────────┐       │
│                           │   VERIFIER LAYER (MUST-PASS gate)     │       │
│                           │ ┌──────┬──────┬─────┬────┬─────────┐ │       │
│                           │ │Formal│ Sim. │ Wet │Code│ Citation│ │       │
│                           │ │(Lean)│(LAMM)│ Lab │Test│Hallucin.│ │       │
│                           │ └──────┴──────┴─────┴────┴─────────┘ │       │
│                           └──────────────────┬───────────────────┘       │
│                                              │ verified                  │
│                                              ▼                           │
│                           ┌──────────────────────────────────────┐       │
│                           │ Output: Hypothesis / Code / MCP / etc│       │
│                           └──────────────────────────────────────┘       │
│                                                                           │
│   ┌────────────────────────── Safety Layer ─────────────────────────┐    │
│   │  Sandbox (Firecracker) │ Novelty Check │ Cost Cap │ Audit Log   │    │
│   └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1.1 핵심 컴포넌트 (7개)

1. **Orchestrator** — PI 역할, 계획·디스패치·상태 관리
2. **Memory Subsystem** — Episodic / Semantic / Procedural 3계층 + KG
3. **Hypothesis Generator (×N)** — 다양성 강제 (다른 prompt, 다른 seed, 다른 backbone)
4. **Tournament Ranker** — Elo pairwise + Critic + Reflection
5. **Executor** — Code/Simulator/Lab API 실행
6. **Verifier Layer** — 5가지 외부 검증 채널 (MUST-PASS gate)
7. **Safety Layer** — Sandbox + Novelty + Cost + Audit (cross-cutting)

---

## 2. 컴포넌트별 상세 설계

### 2.1 Orchestrator

**역할**: 인간 PI를 모방. 계획·역할 분담·상태 추적·체크포인트.

```
입력: 연구 질문 (자연어)
출력: 단계별 작업 그래프 + 각 단계 책임 에이전트

상태:
  - Research goal (immutable)
  - Current phase (PLAN | IDEATE | RANK | EXECUTE | VERIFY | WRITE)
  - Active sub-tasks
  - Budget remaining (token + cost + time)
  - Human-checkpoint pending queue
```

**구현 고려**:
- 상태는 모두 영구 스토리지에 (resume 가능)
- 페이즈 전환마다 Human PI에게 notify (default), opt-out 가능
- Budget 초과 시 자동 escalate (Human 결정)

**Anti-pattern 회피**:
- ❌ "fully autonomous" 모드를 default로 두지 않음 (Agent Lab +0.58 교훈)
- ❌ phase 전환 자동, 인간 모름 (Hidden Pitfalls 권고: trace log mandatory)

---

### 2.2 Memory Subsystem — 3계층 + KG

> **가장 자주 over-engineering 되는 부분.** Kosmos "world model"의 component-level ablation은 없음. **A-MEM과 PiFlow만 component-level 효과 증명**. 따라서 이 두 가지 패턴을 차용.

```
┌──────────────────────────────────────────────────────────────┐
│  Tier 1: EPISODIC (이번 run에서 일어난 모든 사건)            │
│   - Tool call 로그 (input, output, timestamp)                │
│   - Hypothesis 생성·기각 이력                                │
│   - Verifier verdict                                         │
│   - 영구 스토리지 (run UUID로 인덱싱)                        │
├──────────────────────────────────────────────────────────────┤
│  Tier 2: SEMANTIC (학습된 사실 + 외부 지식 + KG)             │
│   - 도메인 KG (개념, 관계, 인용 네트워크)                    │
│   - 검증된 메소드/데이터/툴 카탈로그                         │
│   - A-MEM 스타일: 노트 + 자동 link 생성 + memory evolution   │
│   - Cross-run 누적 (AgentRxiv 패턴)                          │
├──────────────────────────────────────────────────────────────┤
│  Tier 3: PROCEDURAL (재사용 가능한 heuristic + failure mode)  │
│   - "이런 실험에선 baseline X로 시작하라"                    │
│   - "이런 가설은 wet-lab 비용 비싸니까 simulator 먼저"       │
│   - ERL 스타일: 성공/실패 trajectory에서 정수만 distill      │
└──────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────────────┐
│  Retrieval API (모든 에이전트가 사용)                         │
│   - recall_episode(query)  → 비슷한 과거 사건                │
│   - lookup_fact(entity)    → KG에서 정확한 사실               │
│   - get_heuristic(context) → 적용 가능한 절차                │
└──────────────────────────────────────────────────────────────┘
```

**핵심 설계 결정**:
- **Tier 2 (semantic) 제거 시 가장 큰 성능 하락** (arXiv 2603.29194 ablation) → 절대 빼지 말 것
- **Link generation + memory evolution 모두 load-bearing** (A-MEM ablation) → 둘 다 구현
- **mem0 패턴**: extract-store-retrieve를 epoch 단위로 (latency 91% 감소 사례)

**Anti-pattern 회피**:
- ❌ context window에 모든 history dump (Coherence Cliff)
- ❌ 단일 unstructured 텍스트 메모리 (PiFlow 대비 -27% 토큰 효율 차이)

---

### 2.3 Hypothesis Generator (×N)

**철학**: **단일 generator는 본질적으로 mode collapse**. 다양성을 구조적으로 강제.

```
입력: Research goal + Memory context
출력: K개 가설 (서로 다른 angle)

다양성 강제 메커니즘 (3중):
  1. Prompt 다양화: 같은 goal을 K개 다른 framing
     - "혁신 우선" / "실용 우선" / "리스크 최소" / "데이터 가용성 우선"
  2. Backbone 다양화: 가능한 경우 Claude/Gemini/o3 등 섞기
  3. Seed 다양화: temperature/top-p 다양화

각 가설은 schema 강제:
  {
    "claim": str,                    # 단일 검증 가능한 명제
    "rationale": str,                # 왜 그럴 것 같은지
    "test_protocol": str,            # 검증 방법
    "predicted_outcome": str,        # 사전 등록된 예측
    "fallback_if_fails": str,        # 실패 시 다음 가설
    "novelty_evidence": [citation],  # 부재 증명 자료
    "cost_estimate": {tokens, $, time}
  }
```

**핵심 결정**:
- 가설은 항상 **사전 등록된 예측**을 포함 (post-hoc selection bias 차단, Hidden Pitfalls)
- **novelty_evidence는 필수 필드** — Sakana e-fold CV 실패 교훈
- Schema 강제 → free-text 환각 차단

---

### 2.4 Tournament Ranker — Elo + Critic + Reflection

```
입력: K개 가설 후보
출력: 1개 winner + 순위표

라운드:
  for round in 1..R:
    pairs = swiss_pair(candidates)  # 균등 페어링
    for (a, b) in pairs:
      # Critic 에이전트 N명이 독립적으로 판정
      verdicts = [critic_i.judge(a, b) for i in 1..N]
      # 다수결 (반-편향: order swap)
      winner = majority(verdicts + [critic_i.judge(b, a) for i in 1..N])
      update_elo(a, b, winner)
    if elo_converged(threshold): break

  # Reflection 단계 (Co-Scientist 패턴)
  for top_k in candidates[:K_REFINE]:
    top_k = Reflection.improve(top_k, weakness_list)
```

**핵심 함정 회피**:
- Position bias → order swap으로 mitigate
- Same-model judge bias → critic은 generator와 **다른 backbone** 강제
- SPOT 경고: 높은 Elo ≠ 올바른 과학 → **반드시 verifier로 final gate**

**Confidence**: Co-Scientist의 Elo↔정확도 monotonic 상관은 본문 only (Medium-Low). 우리는 가정하지 말고 측정.

---

### 2.5 Executor

**책임**: 코드 실행, 시뮬레이션, lab API 호출.

```
입력: Winner 가설 + test_protocol
출력: 실험 artifact (data, code, log, plot)

3가지 실행 백엔드:
  1. Code Sandbox (Python REPL in Firecracker microVM)
     - 모든 코드 실행 격리
     - Network 정책: read-only API allow-list
     - Cost cap 강제
  2. Simulator Wrapper (LAMMPS/Lean/etc.)
     - 도메인별 wrapper script
     - 파일 포맷 변환 자동화
  3. Lab API (Opentrons/Emerald/Strateos)
     - 모든 명령에 Human approval gate (default ON)
     - Dual-use 안전 체크 (compound safety filter)
```

**산업 표준 따르기**:
- Firecracker microVM (E2B 패턴) — Docker만으론 부족
- 모든 lab API 명령은 PI 확인 (Coscientist 36% jailbreak 교훈)

**Anti-pattern**:
- ❌ `bash -c "$llm_output"` 직접 실행
- ❌ wet lab 실험을 LLM 단독 결정으로

---

### 2.6 Verifier Layer — **MUST-PASS Gate** (시스템의 핵심)

> **이 layer가 VeriScientist를 다른 시스템과 차별화하는 결정적 요소**. 모든 claim은 이 gate를 통과해야 함.

```
입력: 실험 artifact + claim
출력: PASS | FAIL (구체적 reason 포함)

5개 검증 채널 (도메인에 따라 활성화):
```

| 채널 | 적용 | 신뢰도 | 비용 |
|------|------|------|------|
| **C1. Formal Verifier** | Math/Code | 100% | 낮음 |
| **C2. Simulator** | Chem/Materials/Physics | 높음 | 중 |
| **C3. Wet Lab** | Bio/Chem | 최고 | **매우 높음** |
| **C4. Code Test** | ML/SW | 높음 | 낮음 |
| **C5. Citation/Fact Check** | 모든 도메인 | 중 | 낮음 |

```
verify(claim, artifact):
  results = []
  for channel in applicable_channels(claim.domain):
    r = channel.verify(claim, artifact)
    results.append(r)
  # 정책: 적용 가능한 모든 채널이 PASS여야 PASS
  return all(r == PASS for r in results), results
```

#### C1. Formal Verifier (Lean / Type checker / Z3)
- Math claim → Lean 4
- Code claim → static type check + property test
- Cost ~0, 신뢰도 100% — **가능하면 무조건 켜기**

#### C2. Simulator Verifier
- 화학: RDKit + RXN4Chemistry 합성 가능성 체크
- 재료: LAMMPS MD 시뮬레이션
- 물리: 수치 시뮬레이션 결과와 일치 확인

#### C3. Wet Lab Verifier
- Cost: $250k+ (Emerald 진입), $100k+ (Strateos)
- **항상 Human PI 승인 필요**
- 사전 등록된 predicted_outcome과 비교
- n≥3 replicate 강제

#### C4. Code Test Verifier
- pytest / mypy / coverage 강제
- "Test-driven hypothesis" — 가설을 코드 테스트로 표현
- Paper2Agent 패턴: MCP 서버 자동 빌드 + 100% 정확도 case study

#### C5. Citation & Fact Check
- **Semantic Scholar API + Google Scholar** 교차 검증
- 모든 인용에 대해:
  - 실제로 존재하는가? (Sakana 1997→2016 오류 교훈)
  - 인용한 내용이 실제 그 논문에 있는가?
  - 출간연도 정확한가?
- 환각 인용 1건이라도 발견되면 **즉시 FAIL**

**핵심 정책**: **하나라도 FAIL → 전체 FAIL → 재생성 또는 escalate**.

---

### 2.7 Safety Layer (Cross-cutting)

```
4 모듈:
  1. Sandbox: Firecracker microVM (E2B Gateway 패턴)
  2. Novelty Guardrail:
     - Semantic Scholar API (Sakana보다 깊이 검색)
     - FutureHouse Owl 스타일: "Has anyone done X?" 명시 query
     - 다중 검색 엔진 cross-check
     - Embedding similarity (의미적 중복도)
  3. Cost Cap:
     - Per-task token budget
     - Per-task $ budget
     - Per-task wall-clock budget
     - 초과 시 자동 stop + PI notify
  4. Audit Log:
     - 모든 tool call, 모든 LLM call, 모든 verifier verdict
     - Mandatory artifact 표준 (Hidden Pitfalls 권고)
     - Tamper-evident (append-only)
```

**핵심 발견 반영**:
- Misevolve 데이터: self-training이 안전 거부율 70% 감소 → **VeriScientist는 self-training 안 함** (정책 결정)
- Citation 환각 → C5 verifier로 차단
- Prompt injection (peer review attack 5편) → 외부 텍스트 입력은 모두 sanitize

---

## 3. 데이터 플로우 — 한 사이클

```
Phase 0: PLAN
  Human PI: "X를 연구해줘"
  Orchestrator: goal 파싱 → 단계 그래프 생성
  Human PI: 그래프 승인 (default checkpoint)

Phase 1: IDEATE
  Memory: 관련 사전 지식 retrieve
  Hypothesis Generator (×K): K개 가설 생성 (다양성 강제)
  Safety: novelty check (각 가설)
  → K개 가설 (schema 검증됨)

Phase 2: RANK
  Tournament Ranker: R 라운드 Elo + Reflection
  → 1개 winner + 순위표

Phase 3: EXECUTE
  Executor: winner.test_protocol 실행
  Cost Cap 모니터링
  Human approval: wet lab 명령에 대해
  → Artifact (data, code, log)

Phase 4: VERIFY (MUST-PASS GATE)
  Verifier Layer: 적용 가능한 모든 채널
  PASS → Phase 5
  FAIL → escalate
    (a) 자동 재생성 (재시도 N회)
    (b) Critic이 가설 보완
    (c) Human PI에 escalate

Phase 5: WRITE
  Output Specializer: 도메인에 맞는 form으로 출력
    - Math → Lean proof + Markdown 설명
    - Bio → 가설 + 검증 protocol (논문은 인간이)
    - ML → 코드 + 결과 + MCP server
  Citation Check: C5 (최종 한 번 더)
  → Output

Phase 6: REFLECT
  Memory: episodic → semantic → procedural distill
    - 성공한 trajectory에서 heuristic 추출
    - 실패한 trajectory에서 failure mode 추출
  AgentRxiv: 결과를 shared corpus에 publish (opt-in)
```

---

## 4. 출력 타입별 specialization

> **"논문"이 가장 약하고, "정리 증명" / "분자 설계"가 가장 강함** — 100+ 시스템 분석 결론.

VeriScientist는 출력 타입을 명시적으로 specialize:

| 타입 | Verifier | Output |
|------|---------|--------|
| **Theorem (math)** | C1 (Lean) | Lean proof + explanation |
| **Molecule (chem)** | C2 (RDKit) + C3 (wet lab) | SMILES + synthesis protocol + 검증 결과 |
| **Code/Algorithm** | C1 (type) + C4 (test) | repo + test suite + MCP server |
| **Hypothesis (bio)** | C5 (citation) + (optional C3) | 가설 + 검증 protocol (실험은 인간이) |
| **Reproduction** | C4 (test) | container + match report |
| **Reference paper** ⚠️ | **모든 channel + 외부 peer review 권장** | Markdown + 검증 trail |

**중요**: "논문" output은 가장 위험. Default로 disable. 명시적 flag로만 enable + 외부 peer review 권장.

---

## 5. Failure Mode 처리

리서치에서 식별된 6가지 (Lossfunk 2026):

| Failure | 우리의 mitigation |
|---------|------------------|
| Training-data 편향 | Diversity-forced hypothesis generator + 다중 backbone |
| Implementation drift | Code test verifier (C4) + Sandbox 격리 |
| Memory/context 열화 | 3-tier memory (단일 컨텍스트 회피) |
| Overexcitement | MUST-PASS verifier gate (자체 평가 금지) |
| 약한 도메인 지능 | Tier 2 semantic memory (도메인 KG) + 전문 tool 통합 |
| 약한 scientific taste | Tournament에서 다양한 critic angle + Human PI checkpoint |

추가 mitigation:
| 위험 | mitigation |
|------|----------|
| Reward hacking | 독립 verifier + audit log |
| Citation 환각 | C5 강제 |
| Self-modification | 정책: agent는 자기 코드/메모리 직접 수정 불가 |
| Prompt injection | 외부 텍스트 sanitize + plan/exec 분리 |

---

## 6. 인간-AI Co-pilot 인터페이스

**Default 정책**: Human-in-loop는 ON. Fully autonomous는 명시적 opt-in (위험 경고 포함).

```
Checkpoint 위치 (default ON):
  - Phase 0 → 1: 계획 승인
  - Phase 2 후: top-3 가설 검토
  - Phase 3 wet-lab 명령: 각 실험 승인
  - Phase 4 verifier FAIL: escalation 결정
  - Phase 5 output 발행: 최종 검토

Notification 채널:
  - CLI / TUI (개발용)
  - Slack / Email (배포용)
  - Audit log는 항상 read-only으로 노출

Human override:
  - Skip checkpoint (per-session)
  - Force re-rank (Elo 무시)
  - Inject hypothesis (외부 가설 추가)
  - Kill switch (즉시 stop, 모든 외부 명령 취소)
```

---

## 7. 디렉토리 구조 (제안)

```
veriscientist/
├── core/
│   ├── orchestrator.py        # Phase 관리, 상태 머신
│   ├── memory/
│   │   ├── episodic.py        # Run 단위 이벤트 로그
│   │   ├── semantic.py        # KG + A-MEM 스타일
│   │   ├── procedural.py      # ERL 스타일 heuristic distill
│   │   └── retrieval.py       # 통합 retrieval API
│   ├── hypothesis/
│   │   ├── generator.py       # 다양성 강제 K-generator
│   │   └── schema.py          # 가설 schema 검증
│   ├── tournament/
│   │   ├── ranker.py          # Elo + Swiss pairing
│   │   ├── critic.py          # Multi-angle critic
│   │   └── reflection.py      # Co-Scientist 스타일
│   ├── executor/
│   │   ├── sandbox.py         # Firecracker wrapper
│   │   ├── simulator.py       # LAMMPS/Lean/etc.
│   │   └── lab_api.py         # Opentrons/Emerald
│   └── verifier/
│       ├── formal.py          # C1
│       ├── simulator.py       # C2
│       ├── wet_lab.py         # C3
│       ├── code_test.py       # C4
│       └── citation.py        # C5
├── safety/
│   ├── sandbox.py
│   ├── novelty.py
│   ├── cost_cap.py
│   └── audit.py
├── domains/                   # 도메인별 plugin
│   ├── math/
│   ├── chemistry/
│   ├── biology/
│   └── ml_research/
├── interfaces/
│   ├── cli.py
│   ├── tui.py
│   └── slack_bot.py
├── tools/                     # MCP servers
│   ├── semantic_scholar.py
│   ├── github.py
│   ├── arxiv.py
│   └── ...
├── tests/
└── configs/
    ├── default.yaml
    ├── math_domain.yaml
    └── bio_domain.yaml
```

---

## 8. 기술 스택 (제안)

| 영역 | 선택 | 이유 |
|------|-----|------|
| Orchestration | LangGraph or 자체 state machine | 명시적 상태 머신 필요 |
| LLM 백본 | Multi-vendor (Claude/Gemini/o3) | Critic 편향 회피 |
| 메모리 KG | Neo4j or NetworkX + Pickle | Tier 2 |
| 벡터 검색 | Qdrant / FAISS | Tier 1 episodic |
| Sandbox | E2B + Firecracker | 산업 표준 |
| Lab API | Opentrons HTTP, Emerald SDK | (도메인 활성화 시) |
| Formal | Lean 4 | 수학 |
| Sim | RDKit, LAMMPS Python wrapper | 화학/재료 |
| Citation API | Semantic Scholar + OpenAlex | C5 |
| Audit log | Append-only Parquet on S3/local | Tamper-evident |
| 모니터링 | Langfuse / Phoenix | 모든 trace 기록 |

---

## 9. 첫 milestone — MVP scope

> **모든 도메인 한 번에 하지 말 것.** 한 도메인 + verifier 하나로 시작.

### MVP-1: Math 도메인 + Lean Verifier만

이유:
- C1 (Lean) verifier 가장 신뢰도 높음 (100%)
- Wet lab 비용 없음 (Cloud lab 진입 $250k 회피)
- 결과가 binary (proved or not) → 평가 명확
- AlphaProof / Seed-Prover / Aristotle 선행 사례 풍부

### MVP-1 범위
- Orchestrator (단순화: 직선 phase)
- Hypothesis Generator (K=3)
- Tournament (R=2 라운드)
- Lean Verifier만 (C1)
- Memory Tier 1만 (episodic)
- Safety: sandbox + audit log
- Human checkpoint: 모든 phase

### MVP-1 평가
- miniF2F 벤치마크에서 베이스라인 LLM 대비 측정
- 목표: pass@1을 X% 개선 (구체 숫자는 baseline 측정 후)

### MVP-2 이후 확장 (priority order)
1. Tier 2 semantic memory + KG
2. C4 code test verifier + Python 코드 가설
3. C5 citation verifier
4. Tier 3 procedural memory + heuristic distill
5. Multi-domain (chem/bio 단계적)
6. C3 wet lab (lab partnership 필요)

---

## 10. 의도적으로 채택하지 않은 것

리서치에서 본 일부 인기 패턴은 **의도적으로 배제**:

| 패턴 | 채택 안 한 이유 |
|------|---------------|
| "World Model" (Kosmos 식) | Component ablation 없음. 효과 isolation 불가. 우리는 명시적 3-tier memory로 |
| Self-evolving / self-training | Misevolve 데이터: 70% 안전 거부율 손실 |
| "Fully autonomous" 모드 default | Agent Lab +0.58 / Sakana 환각 교훈 |
| "Paper writing" as primary output | 가장 fabrication 위험 큼 |
| 단일 LLM judge | LLM judge가 tool-grounded agent를 sabotage (ChemCrow paradox) |
| 너무 많은 tool (50+) | 도구 품질 > 양 — 도메인 핵심 1-3개로 시작 |
| MCTS + Nash 등 복잡 search | Linear vs tree 차이만 verify됨, 그 이상은 over-engineering |

---

## 11. 측정 metric

VeriScientist의 성공은 다음으로 측정:

### Primary
- **Verified-output rate**: verifier gate 통과율 / total claims
- **Reproducibility rate**: 외부에서 우리 결과 재현 성공률 (target: >50%)
- **Citation hallucination rate**: C5에서 잡힌 환각 / total citation (target: <1%)

### Secondary
- **Novelty validation**: 우리가 novel이라 claim한 것 중 실제로 novel인 비율
- **Cost per verified claim**: $ / passed claim
- **Time per verified claim**: hours / passed claim
- **Human checkpoint friction**: PI가 거부한 비율

### Anti-metric (의도적으로 추적 안 함)
- **"논문 통과율"** — Goodhart 위험 큼 (Sakana ICLR workshop 교훈)
- **자체 평가 점수** — Sakana CUDA 교훈

---

## 12. Open Questions (구현 시 결정 필요)

리서치에 답이 없는 결정들:

1. **Hypothesis Generator의 K값** — 비용 vs 다양성 trade-off. 시작: K=5
2. **Tournament 라운드 수 R** — Co-Scientist는 수렴까지. 시작: R=3, 수렴 체크
3. **Critic 수 N** — 다수결을 위한 N. 시작: N=3, 다른 backbone
4. **Memory Tier 2 KG 스키마** — 도메인별로 다름. 도메인 plugin이 정의
5. **Co-pilot frequency** — 너무 많으면 인간 피로, 너무 적으면 fabrication. 시작: phase별
6. **Cost cap 기본값** — 시작: $50 / hypothesis (PI 조정 가능)
7. **Audit log retention** — 시작: 영구 (S3 archive)
8. **Cross-run memory 공유 정책** — AgentRxiv 패턴 자체 lab 내에서만 (opt-in)

---

## 13. 참고 문헌 (설계 근거)

### Verifier-First 근거
- Sakana AI CUDA Engineer 사건 (2025.02)
- MLR-Bench 80% fabrication (NeurIPS 2025, arXiv 2505.19955)
- Hidden Pitfalls (NeurIPS 2025 Spotlight, arXiv 2509.08713)

### Multi-Agent / Tournament 근거
- AI Co-Scientist (arXiv 2502.18864) + Nature May 2026
- HypoAgents (arXiv 2508.01746)
- Stanford Virtual Lab (Nature 2025)
- Anthropic Multi-Agent System (+90.2% 내부 평가)

### Memory 근거
- A-MEM (arXiv 2502.12110) — link + evolution 모두 load-bearing
- mem0 (arXiv 2504.19413) — 91% latency cut
- PiFlow (arXiv 2505.15047) — +31-42% efficiency
- ERL (ICLR 2026, arXiv 2603.24639) — heuristic distill

### Safety 근거
- Misevolve (arXiv 2509.26354) — self-training 70% safety loss
- Beel et al. (arXiv 2502.14297) — citation 환각 정량
- Biosecurity Agent (arXiv 2510.09615)
- 5편 peer review prompt injection (2508.20863, 2509.10248, 2511.01287, 2512.10449, 2601.06884)

### Reproducibility 근거
- PaperBench (arXiv 2504.01848) — 21% 천장
- NatureBench (arXiv 2606.24530) — 17.8% 천장
- CodeScientist (arXiv 2503.22708)
- Paper2Agent (arXiv 2509.06917) — 100% genomics

### Co-pilot 근거
- Agent Laboratory (arXiv 2501.04227) — +0.58 quality

---

## 14. 한 줄 정리

> **VeriScientist는 "또 하나의 AI Scientist"가 아니다. 100+ 시스템 리서치에서 (1) Verifier 없으면 fabrication 80%, (2) Self-eval은 reward hacking 보장, (3) Citation 환각이 만연하다는 결론을 받아들이고 — 검증 gate를 architectural 1순위로 둔 시스템.**

> **"빠르게 논문 찍는" 시스템이 아니라, "검증된 claim만 천천히 내놓는" 시스템.**

---

## Appendix A: 구현 우선순위 한 줄 요약

```
1. Sandbox + Audit log         (안전 기반)
2. Lean verifier (C1)          (수학으로 MVP)
3. Single hypothesis generator (baseline)
4. Citation verifier (C5)      (모든 도메인에 적용)
5. Tournament + multi-critic   (품질 ↑)
6. Tier 1 episodic memory      (resume 가능)
7. Diversity-forced K-gen      (mode collapse 방지)
8. Tier 2 semantic memory + KG (도메인 확장 기반)
9. Code test verifier (C4)     (ML 도메인 확장)
10. Co-pilot Slack/Email 통합   (배포 가능)
... (이후 도메인별 plugin)
```

---

설계자 노트: 이 문서는 청사진이지 spec이 아니다. 구현 중 학습한 것은 이 문서에 피드백 — 특히 Open Questions의 결정 결과는 명시적으로 업데이트.
