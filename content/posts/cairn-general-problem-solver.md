---
title: "[리뷰] Cairn — 범용 문제 해결 엔진 (54/54 Tencent 해킹 챌린지 우승 시스템)"
date: 2026-06-26
tags: ["Cairn", "Agent", "Blackboard", "OODA", "Pentesting", "GitHub"]
categories: ["ML/AI"]
summary: "Cairn은 '기점·목표·미지의 경로'가 있는 문제라면 무엇이든 풀도록 설계된 directed search 엔진. Tencent Cloud 해킹 챌린지 (2회)에서 54/54 완전 정답 (유일)으로 종합 3위. Blackboard 아키텍처 + OODA 루프 + stigmergy 조율. Fact/Intent/Hint 3개 primitive로 에이전트 간 직접 통신 없이 협업. 아키텍처 해부 + Scientist Agents 설계 관점 비교."
math: true
toc: true
draft: false
---

## 한 줄 소개

> **"Blackboard + OODA + stigmergy"** — 에이전트끼리 직접 대화하지 않고 공유 그래프에 사실만 쓴다. 사전 학습·MCP 도구·RAG·역할 정의 **하나 없이** Tencent Cloud 해킹 챌린지 **54문제 전부 solve** (참가팀 610팀 중 유일), 종합 3위.

- **레포**: [github.com/oritera/Cairn](https://github.com/oritera/Cairn) (1.9k stars, 273 forks)
- **언어**: Python 62.9%, HTML 36.0%, Dockerfile 1.1%
- **라이선스**: AGPLv3 (상업용 별도)
- **최신 릴리스**: v0.2.1 (2026-05-10)

---

## 1. Cairn이 뭐고 왜 재미있는가?

Cairn은 **범용 문제 해결 엔진** (general-purpose problem-solving engine)이다. 저자의 명확한 정의:

> **"기점(origin)이 정의되고, 목표(goal)가 정의되어 있으며, 그 사이의 경로가 미지인 문제"**

이 정의로 다음 태스크들이 모두 같은 시스템으로 풀 수 있다:
- 침투 테스트 (initial validation domain)
- 취약점 리서치
- CTF 챌린지
- 수학 증명
- 그 외 "찾아가야 하는" 문제 전반

**흥미로운 지점**: [Scientist Agents 설계 차원 20가지](scientist-agents-design-dimensions.md)에서 정리했던 원칙들 — external verifier, 다양성, tournament, 공유 메모리 — 여러 개가 Cairn에도 등장한다. 서로 독립적으로 도달한 결론이라는 게 시사적이다.

---

## 2. 검증 이력 — Tencent Cloud 해킹 챌린지 2회

- **참가**: 610팀 / 1,345명
- **결과**: **54/54 문제 완전 solve — 유일한 만점 팀**
- **종합 순위**: 3위
- **주목할 점**: 저자 표현 그대로 — **"zero MCP tools, zero RAG, zero predefined agent roles"** 로 대회 시작
- 즉, 도메인 특화 없이 범용 엔진 그 자체로 완주

이게 왜 놀라운가? 침투 테스트는 원래 **도구·워크플로우 특화**가 정답이라 여겨진 도메인. 여기서 "도구 없이 범용 시스템으로 만점"은 아키텍처가 정말로 도메인-무관하게 동작함을 의미한다.

---

## 3. 아키텍처 — Blackboard + OODA + Stigmergy

### 3.1 Blackboard 3-Primitive

Cairn의 핵심 발상은 **모든 에이전트가 공유하는 그래프 (Blackboard)**에 3종의 원소만 쓴다는 것이다:

| Primitive | 의미 |
|-----------|-----|
| **Fact** | 확인된 객관적 발견 (verified finding) |
| **Intent** | 선언된 탐색 방향 (아직 실행 안 됨) |
| **Hint** | 런타임에 인간이 주입하는 판단 |

**핵심**: 에이전트끼리 **직접 통신하지 않는다**. 오직 이 그래프에 읽고 쓰기만 한다. 이걸 stigmergy(스티그머지 — 흔적 기반 협업)라 부른다. 개미가 페로몬으로 협업하는 방식.

### 3.2 OODA 루프

각 워커(Agent Worker)는 군사 이론의 **OODA** 루프로 동작:

```
Observe   → 그래프 상태 관찰
Orient    → 현재 위치 판단
Decide    → 다음 Intent 결정
Act       → 실행하고 Fact 기록
```

이 루프가 여러 워커에서 동시에 돈다. 조정은 오직 Blackboard.

### 3.3 3가지 Task 타입 (동일 Worker 클래스가 실행)

| Task | 역할 |
|------|-----|
| **Bootstrap** | 프로젝트 시작 시 direct problem-solving 시도 |
| **Reason** | 그래프 전체 분석 → 목표 달성 여부, 다음 Intent 도출 |
| **Explore** | Claimed Intent 실행, 결과를 Fact로 보고 |

**설계적 우아함**: 세 task가 다른 코드가 아니라 **같은 Worker의 다른 진입점**. Bootstrap은 사실상 "선택적" — 실패해도 Reason이 이어받음.

---

## 4. 시스템 컴포넌트 3층 구조

```
┌────────────────────────────────────────────────────────┐
│  Cairn Server                                          │
│    - Blackboard 그래프 consistency 유지                  │
│    - Fact/Intent/Hint 저장소                             │
└────────────────┬───────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│  Cairn Dispatcher                                       │
│    - 그래프 읽고 Task 스케줄링                            │
│    - 컨테이너 라이프사이클 관리                            │
│    - 프로토콜 로그 기록                                   │
└────────────────┬───────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│  Worker Containers (프로젝트별 격리)                       │
│    - 프로젝트당 여러 Agent Worker 동시 실행                 │
│    - Claude Code / Codex / Pi 백엔드 지원                 │
└────────────────────────────────────────────────────────┘
```

### 지원 LLM 백엔드
- **Claude Code**
- **Codex** (OpenAI)
- **Pi** (Alibaba Qwen 계열)

`dispatch.yaml`에서 여러 백엔드를 동시에 등록해 병행 실행 가능.

---

## 5. 실전 설정 — `dispatch.yaml` 해부

Cairn의 진짜 재미는 config에서 드러난다:

```yaml
runtime:
  interval: 3                   # 스케줄러 tick (초)
  max_workers: 8                # 전역 동시 워커 상한
  max_running_projects: 3       # 동시에 활성 프로젝트 수
  max_project_workers: 4        # 프로젝트당 워커 상한
  worker_healthcheck: "startup_only"
  prompt_group: "default"

tasks:
  bootstrap:
    timeout: 300
    conclude_timeout: 90        # 실패 시 요약용 fallback
  reason:
    timeout: 300
    max_intents: 2              # 한 단계에서 만들 수 있는 새 Intent 수 hard cap
  explore:
    timeout: 300
    conclude_timeout: 90
```

**설계적 통찰**:

### 5.1 `max_intents: 2` — 폭발 방지
Reason이 한 번 돌 때마다 새 Intent 최대 **2개**. 무한 확장 방지 — 이는 [Scientist Agents 리서치](scientist-agents-2025-2026-report.md)에서 본 `ResearchGym` failure mode ("impatience, poor resource mgmt")에 대한 명시적 방어.

### 5.2 `conclude_timeout` — Fallback 예산 분리
Main task가 timeout으로 실패하면 별도 예산으로 **결과 요약**만 시도. LLM이 무한 루프 대신 "여기까지 알아낸 것" 반환.

### 5.3 여러 백엔드 병렬 등록
```yaml
workers:
  - name: "claudecode_deepseek-v4-pro"
    type: "claudecode"
    env:
      ANTHROPIC_MODEL: "deepseek-v4-pro"
      ANTHROPIC_BASE_URL: "https://api.deepseek.com/anthropic"
  - name: "codex_qwen3.6-plus"
    type: "codex"
    env:
      CODEX_MODEL: "qwen3.6-plus"
  - name: "pi_qwen3.6-plus"
    type: "pi"
    env:
      PI_MODEL: "qwen3.6-plus"
```
- **Claude Code + DeepSeek-v4-Pro** (Anthropic 호환 endpoint 사용)
- **Codex + Qwen3.6-Plus** (OpenAI 호환)
- **Pi + Qwen3.6-Plus** (openai-completions provider)

→ **동일 backbone(Qwen)이지만 다른 harness (Codex vs Pi)** — 재미있게도 harness 다양성 자체가 다양성으로 기능

### 5.4 우선순위와 per-worker concurrency
```yaml
- max_running: 2
- priority: 0        # 낮은 숫자가 먼저 선택
```
- Pi 같은 느린 backend는 `max_running`을 낮게 설정 권장

---

## 6. 설치 & 실행

### 6.1 요구사항
- macOS 또는 Linux
- Python ≥ 3.12
- Docker
- UV (Astral 패키지 매니저)

### 6.2 Docker Compose (권장)
```bash
docker pull --platform=linux/amd64 ghcr.io/oritera/cairn-worker-container:latest
docker pull ghcr.io/astral-sh/uv:python3.13-trixie
cp dispatch.example.yaml dispatch.yaml
# API 키·엔드포인트 채우기
docker compose up --build
```
- 서버가 8000 포트
- 데이터는 `./datas/cairn/`에 영구 저장

### 6.3 수동 실행
```bash
# 서버
uv run --project cairn cairn serve

# 디스패처
uv run --project cairn cairn dispatch --config dispatch.yaml

# 헬스체크만
uv run --project cairn cairn dispatch --config dispatch.yaml --startup-healthcheck-only

# 테스트
uv run --project cairn --group dev pytest
```

---

## 7. 설계 관점 분석 — 왜 잘 될까?

지난 [Scientist Agents 설계 차원 20가지](scientist-agents-design-dimensions.md) 리서치를 렌즈로 Cairn을 해부하면:

### 7.1 검증됐던 6가지 인과 요소 매칭

| 차원 | Cairn의 구현 | 강도 |
|------|-----------|------|
| **External verifier** | Fact는 "확인된 발견"으로 정의 → Explore가 Intent를 실제 실행해서 얻은 결과만 Fact | 강함 |
| **Tree/tournament search** | Intent 확산 (max_intents로 제어) + Reason이 최적 경로 선택 | 중간 |
| **Test-time compute scaling** | Worker 수·interval·timeout으로 명시적 조절 | 강함 |
| **Tool grounding** | 침투 테스트 자체가 실 세계 검증 (target 시스템이 ground truth) | 매우 강함 |
| **Shared corpora** | Blackboard 자체가 공유 그래프 | 매우 강함 |
| **Memory component** | Fact/Intent/Hint 명시 분리 | 강함 |

→ **6개 중 4개를 강하게 만족**. 나머지도 중간 이상.

### 7.2 Stigmergy의 우수성

Cairn이 **직접 에이전트 간 통신을 배제**한 것은 우연이 아니다:
- [Wynn et al. 2025 "Talk Isn't Always Cheap"](llm-harness-optimization-errors.md) — multi-agent debate가 오히려 정확도를 −12 pp 낮춤 (sycophantic conformity)
- Cemri et al. MAST taxonomy — "reasoning-action mismatch 13.2%", "task derailment 7.4%" 등 상당수가 에이전트 간 대화에서 발생

→ **직접 통신을 없애면 이 실패 모드가 원천 차단**. Blackboard가 있으면 굳이 대화할 이유가 없다.

### 7.3 `max_intents: 2`가 큰 발견
장기 horizon 에이전트의 dominant failure mode인 "impatience + explosion"에 대한 architectural 답. Config 한 줄로 해결하는 게 우아하다.

### 7.4 Hint의 존재
**인간이 런타임에 판단 주입 가능** — [Scientist Agents 리서치](scientist-agents-design-dimensions.md)의 "Co-pilot by default" 원칙과 정확히 일치. Agent Lab의 +0.58 quality gain 근거.

---

## 8. 흥미로운 트레이드오프

### 8.1 도메인 무관 vs 도메인 특화
- **장점**: 침투 테스트, 수학, CTF 모두 같은 코드로
- **단점**: 도메인 특화 도구 없음 → 도메인 최적화 여지 큼
- Cairn은 명확히 전자 선택 — "MCP 없이 만점"이 증명

### 8.2 Blackboard의 오버헤드
- 매 tick마다 그래프 상태 조회 → 그래프 커지면 병목 가능
- `max_workers: 8`, `max_project_workers: 4` 로 자연스럽게 제한

### 8.3 세 백엔드 병렬의 정치학
- Claude Code, Codex, Pi 각각 다른 도구 접근 방식
- 같은 Intent를 다르게 해석할 수 있음 → 다양성 이득
- 하지만 debugging 시 원인 추적이 복잡

---

## 9. 한계와 미해결 지점 (필자 관점)

1. **Fact의 진위**: "확인된 발견"이라지만 LLM이 "확인했다"고 주장한 것을 어떻게 신뢰? — Cairn 문서에 명시적 verifier 언급 부족. 침투 테스트에선 exploit 성공이 자체 ground truth지만 수학·CTF에선 명확치 않음.
2. **Intent 중복 감지**: 여러 워커가 동시에 비슷한 Intent 만들면? — max_intents는 총량만 제한
3. **Hint 활용도**: 인간이 얼마나 자주 개입해야 효과적인지 데이터 없음
4. **재현성**: 병렬 워커 + LLM 비결정성 → 같은 문제 두 번 solve 궤적이 완전히 다를 가능성

이런 지점들이 오히려 흥미로운 후속 연구 소재.

---

## 10. 응용 아이디어

Cairn 아키텍처를 다른 도메인에 매핑하면:

| 도메인 | Fact | Intent | Hint |
|--------|-----|--------|------|
| **버그 헌팅** | 확인된 취약점 | "함수 X에 fuzz" | "이 CVE 봐봐" |
| **수학 증명** | 검증된 lemma | "이 lemma로 정리 X 시도" | "귀납법 써봐" |
| **논문 리뷰** | 검증된 사실 | "관련 논문 X 조사" | "이 저자 논문 우선" |
| **데이터 분석** | 통계적 발견 | "이상치 X 조사" | "이 column 먼저" |
| **디버깅** | 재현된 버그 | "이 코드 경로 조사" | "이 커밋부터 봐" |

**설계 재사용성이 매우 높음** — Cairn의 real value는 여기 있는지도 모른다.

---

## 11. 결론 — 두 문장

> **Cairn은 "에이전트끼리 대화하지 않는 멀티 에이전트"라는 급진적 발상으로, 침투 테스트 도메인에서 만점을 낸 verified 시스템이다.** MCP·RAG·역할 정의 없이 범용 엔진 그 자체로.

> **Blackboard + OODA + stigmergy는 [Scientist Agents 설계 원칙](scientist-agents-design-dimensions.md)에서 도출한 6가지 rigorous 요소를 상당수 만족한다.** 서로 다른 도메인에서 독립적으로 도달한 결론이 겹친다는 게 이 아키텍처의 정당성을 시사한다.

---

## 12. ⚠️ 사용 시 주의

Cairn은 침투 테스트 도구를 포함한다. README의 명시적 경고:

> **"권한이 명확히 부여된 시스템에만 사용하라."**
> **"사용자가 인가에 대한 모든 책임을 진다."**

무단 사용은 대부분 국가에서 **불법**. CTF 환경, 자신의 시스템, 명시적 승인이 있는 pentest에서만 사용.

---

## 관련 블로그 포스트

- [Scientist Agents 설계 차원 20가지](scientist-agents-design-dimensions.md) — Cairn 아키텍처를 뒷받침하는 6가지 검증된 설계 원칙
- [Scientist Agents 1년 종합 보고서](scientist-agents-2025-2026-report.md) — 100+ 시스템 비교 지도
- [LLM Harness 최적화·오류](llm-harness-optimization-errors.md) — MAST taxonomy, Wynn "Talk Isn't Always Cheap" 등 Cairn stigmergy 정당화 근거
- [VeriScientist 설계 청사진 (외부 문서)](../../designs/veriscientist-architecture.md) — verifier-first Scientist Agent 설계

---

## 참고 자료

- **GitHub**: [oritera/Cairn](https://github.com/oritera/Cairn)
- **워커 컨테이너**: `ghcr.io/oritera/cairn-worker-container:latest`
- **v0.2.1 릴리스**: 2026-05-10
- **Tencent Cloud Hackathon 2회 결과**: 54/54 solve, 종합 3위 (참가 610팀, 1,345명)
- **저자 X**: @le1xia0
- **관련 글 (중국어)**:
  - "The Strongest AI Penetration Testing Agent: Postmortem..."
  - "The Pathless Path: Cairn AI from Penetration Testing to General Problem Solving"
