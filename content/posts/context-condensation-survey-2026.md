---
title: "[서베이] Context Condensation 연구 동향 (2026) — 에이전트 컴팩션·RL 학습·안전성·표현 압축·이론"
date: 2026-09-16
tags: ["연구노트", "서베이", "에이전트", "LLM", "메모리", "ContextCompression", "2026"]
categories: ["ML/AI"]
summary: "2026년 기준 LLM 컨텍스트 압축(context condensation / compaction) 연구 동향 총정리. 에이전트 이력 컴팩션의 5개 흐름(RL 학습·외부 관리자·주소 지정 회수·메타인지·서빙), 컴팩션이 안전 제약을 지우는 문제, soft token·KV·테스트타임 메모리 압축, 그리고 communication complexity와 rate-distortion 기반 이론화까지. Titans/HOPE 재현 연구와의 접점도 정리."
math: true
toc: true
draft: false
---

## Executive Summary

**Context condensation**은 LLM이 소비하는 컨텍스트(대화 이력·도구 관측·긴 문서)를 더 짧은 표현으로 줄이는 모든 기법을 가리킨다. 에이전트 프레임워크에서는 *compaction*, *condenser*, *summarization*, 아키텍처 쪽에서는 *context compression*, *compressive memory*, 서빙 쪽에서는 *KV cache compression*이라고 부르지만, 2026년 들어 이들을 **하나의 문제**로 보는 시각이 자리잡았다.

2026년 동향을 한 문장으로 요약하면 다음과 같다.

> 무게중심이 "요약 프롬프트를 잘 쓰는 문제"에서 **"압축을 학습하고, 검증하고, 이론화하는 문제"**로 이동했다.

핵심 변화 다섯 가지:

| # | 변화 | 대표 연구 |
|---|------|----------|
| 1 | 컴팩션을 **RL 정책 안에** 학습 (요약을 에이전트가 생성, 압축된 궤적에서 학습) | CompactionRL(Zhipu), Context-Folding, AgeMem, AdaCoM |
| 2 | 손실 요약 대신 **주소 지정 가능한 무손실 회수** + 에이전트 자기 인식 | ARC, VISTA |
| 3 | 컴팩션이 **안전 제약을 조용히 지운다**는 실증 → 타입별 보존 정책 | Governance Decay, Compaction Cliff, TRACE |
| 4 | **대규모 사전학습된 soft-token 압축기**와 테스트타임 최적화형 메모리 | LCLM(End-to-End Compression at Scale), GradMem, NestedKV |
| 5 | 컴팩션의 **이론화**: one-way communication complexity, rate-distortion | Context Compaction Theory(Anthropic·Harvard), Rate–Distortion View |

논문 대부분이 2026년 5~8월에 집중되어 있어, 지금이 이 분야의 "형성기"라고 볼 수 있다.

---

## 1. 문제 정의와 분류

### 1.1 왜 지금 문제가 되는가

장기 실행 에이전트(코딩·딥리서치·웹 탐색)는 스텝마다 도구 관측이 누적되어 컨텍스트가 단조 증가한다. 컨텍스트 윈도우가 1M 토큰이어도 두 가지 문제가 남는다.

- **비용·지연**: 컨텍스트 길이에 비례해 prefill 비용과 KV 메모리가 증가한다. 요약 호출 자체가 수십 초 블로킹된다.
- **Context rot**: 토큰이 늘수록 모델의 회상 정확도가 떨어진다. LOCA-bench는 환경 상태가 복잡해질수록 에이전트 성능이 일반적으로 하락함을 통제 실험으로 보였다.

즉 압축은 "윈도우에 안 들어가서" 하는 것이 아니라, **들어가더라도 해야 하는** 것이 되었다.

### 1.2 세 개의 층

Rate–Distortion View 논문의 분류를 빌리면, 컨텍스트 압축은 스택의 세 층에서 동시에 일어난다.

| 층 | 대상 | 대표 기법 |
|----|------|----------|
| **에이전트 층** | 대화 이력·도구 관측 (자연어) | 슬라이딩 윈도우, 롤링 요약, 계층 요약, 상태 폴딩(JSON), RAG 오프로딩 |
| **표현 층** | 토큰 시퀀스 → 잠재 벡터 | gist/soft token, 인코더-디코더 압축기, 버퍼 토큰 |
| **아키텍처·서빙 층** | KV 캐시·순환 상태·fast weight | KV 축출/양자화, SSM 상태, Titans/HOPE형 연상 메모리 |

세 층의 공통 실패는 동일하다: **질의를 알기 전에** attention 크기나 최근성으로 정보를 버린다.

---

## 2. 에이전트 층: 컴팩션의 5개 흐름

### 2.1 RL로 컴팩션을 정책 안에 학습

가장 큰 패러다임 전환이다. 기존에는 컨텍스트가 임계치에 닿으면 별도 프롬프트로 요약을 생성하고, 에이전트는 그 요약을 "주어진 것"으로 받았다. 2026년 흐름은 요약 생성도 에이전트의 행동으로 두고, **압축된 궤적에서 학습이 가능하도록 RL 목적함수를 고친다**.

**CompactionRL (Zhipu, 2026.07)**
- 태스크 수행과 요약 생성을 공동 최적화. 핵심 기술은 (a) 토큰 단위 손실 정규화, (b) 궤적 간(cross-trajectory) GAE — 컴팩션 전후 세그먼트가 서로 다른 시퀀스가 되어도 advantage가 일관되게 전파되도록 한다.
- GLM-5.2(750B) 학습에 실제 투입.

| 모델 | SWE-bench Verified | Terminal-Bench 2.0 |
|------|-------------------|-------------------|
| GLM-4.5-Air (106B) | 66.8 (+7.0) | 24.5 (+3.1) |
| GLM-4.7-Flash (30B) | 56.0 (+5.5) | 20.2 (+6.8) |

**계열 논문**
- **Context-Folding**: 하위과제로 브랜치 → 완료 시 요약으로 폴드. 분해와 컨텍스트 관리 자체에 보상.
- **SUPO / ReSum**: 요약 기반 컨텍스트 관리를 멀티턴 RL에 넣거나(SUPO), 주기적 외부 요약 도구와 세그먼트 궤적으로 요약 조건부 에이전트를 학습(ReSum).
- **AgeMem**: store / retrieve / update / summarize / discard 다섯 메모리 연산을 도구로 노출하고 3단계(SFT 웜업 → 태스크 RL → 스텝 단위 GRPO)로 학습.
- **Dynamic Long Context Reasoning over Compressed Memory (2026.02)**: 청크 압축기 + 게이팅 + 추론기를 end-to-end RL로 공동 학습. 7K → 1.75M 토큰 외삽, MemAgent 대비 6× 속도.

### 2.2 에이전트는 고정, 외부 관리자를 학습

폐쇄형 모델을 쓰는 실무에서는 에이전트 자체를 재학습할 수 없다. 이 제약을 받아들이는 흐름이다.

**AdaCoM — Learning Agent-Compatible Context Management (2026.05)**
- 별도 LLM이 frozen 에이전트의 컨텍스트를 유연한 편집 행동으로 관리하도록 end-to-end RL.
- 주목할 발견: **강한 에이전트는 고충실도 보존이, 약한 에이전트는 공격적 압축이 유리**하다. 전이는 비슷한 능력 수준 사이에서 가장 잘 된다. 즉 "최적 압축률"은 모델 능력의 함수다.

**ACON — Optimizing Context Compression for Long-horizon Agents (Microsoft)**
- 학습 없이 에이전트 실패 분석으로 자연어 압축 지침을 반복 정제. 최적화된 압축기를 소형 모델로 증류.
- 피크 토큰 26~54% 절감, 소형 모델 성능 최대 46% 향상 (AppWorld·OfficeBench·Multi-objective QA).

### 2.3 손실 요약 대신 주소 지정 가능한 무손실 회수

"요약은 본질적으로 손실이다"라는 전제에서 **아카이브와 표시를 분리**하는 설계다.

**ARC — Addressable Recall Compaction (2026.07)**
- 도구 관측을 append-only, ID 주소 지정 로그에 보관. 컨텍스트 한계에 닿으면 오래된 관측을 짧은 인용(citation)으로 치환. 필요하면 ID로 즉시 되찾는다 — 도구 재실행도, 유사도 검색도 필요 없다.

| 벤치마크 | ARC | 최고 베이스라인 |
|---------|-----|---------------|
| Needle-in-a-Haystack (exact) | 99.40 | 88.12 |
| LongBench-v2 Hard | 29.97 | 28.25 |

**VISTA — LLM Agents Are Latent Context Managers (2026.06)**
- 문제 진단: 모델은 자기 컨텍스트 한계를 **지각하지 못한다**(proprioceptively blind).
- 워킹 메모리를 타입 있는 주소 지정 블록으로 표현하고, 토큰 사용량·최근성·아카이브 상태·잔여 예산을 **대시보드**로 노출. 아카이브 결정은 모델이 한다.
- LOCA-Bench에서 Gemini-3-Flash 22.7 → 50.7. 10K~1M 토큰 궤적에 걸쳐 모델 간 전이.

두 논문은 학습이 필요 없고, 압축 정보를 버리지 않는다는 점에서 2.1과 정반대 극단에 있다. 실무에서는 이 둘의 결합(무손실 아카이브 + 학습된 요약)이 자연스러운 다음 단계다.

### 2.4 언제 압축할지의 메타인지

**SelfCompact — Self-Compacting Language Model Agents (2026.06)**
- 파인튜닝 없이 두 가지만 준다: (1) 호출 가능한 compaction 도구, (2) 트리거 루브릭 — *하위과제 종료·궤적 수렴 시 압축, 유도 도중·막힘 상태에서는 억제*.
- 수학 최대 +18.1점, 에이전트 검색 +5~9점, 질문당 토큰 30~70% 절감. **고정 주기 요약보다 낫다**.
- 핵심 통찰은 "메타인지 갭": 모델은 스스로 컨텍스트 저하를 인식하지 못하지만, 간단한 루브릭만으로 그 갭을 메울 수 있다.

### 2.5 서빙 관점

**Parallel Context Compaction (2026.05)**
- LLM 요약의 세 문제: 수십 초 블로킹, 프롬프트로 지시해도 요약 분량이 예측 불가, 실행마다 보존 정보가 흔들림.
- 이력을 블록으로 나눠 병렬 요약 → 동일 디코드 볼륨에서 wall time 감소, 요약 분량의 세밀한 제어. 8B~120B, dense/MoE 4개 백본, HotpotQA·LoCoMo에서 검증.

---

## 3. 경고 신호: 컴팩션은 안전성·안정성 실패 표면

2026년 여름 가장 눈에 띄는 하위 흐름이다. 세 논문이 독립적으로 **"요약이 제약을 조용히 지운다"**를 실증했다.

| 논문 | 발견 | 핵심 수치 |
|------|------|----------|
| **Governance Decay** (2026.06) | 정책이 컨텍스트에 보이면 준수, 컴팩션 후 위반 | 위반율 0% → 30% (일부 모델 59%). 1,323 에피소드에서 제약이 요약에 살아남으면 0%, 사라지면 38% |
| **Compaction Cliff** (2026.08) | 안전 규칙은 정확한 문구가 필요해 요약에 취약 | Claude Sonnet 4.6이 1회 컴팩션 후 안전 규칙 53%, 5회 후 10% 보존 |
| **TRACE** (2026.08) | 압축이 최근 상호작용 영향력을 약화 | 차단 행동·반복 탐색 증가, 실행 간 불안정성 증가 (AppWorld) |

Governance Decay는 여기에 **Compaction-Eviction Attack**을 추가한다 — 적대적 in-context 콘텐츠가 요약기를 편향시켜 정당한 정책을 누락시키게 하며, 최적화된 삽입은 평가한 모든 모델을 격파했다. 컨텍스트 관리 층이 새로운 **공격 표면**임을 뜻한다.

### 대응: 타입별 보존 정책

세 논문의 처방은 수렴한다. 정보를 **타입으로 분류**하고 타입마다 다른 보존 정책을 적용하라.

- **Constraint Pinning** (Governance Decay): 거버넌스 제약을 손실 압축 바깥에 고정. 벤치마크에서 위반율 0% 복원.
- **Knowledge Triage** (Compaction Cliff): TypeCompact(타입별 충실도로 재작성), TypeDecompose(큰 토픽 분할 시 안전 규칙 복제), TypeRetrieve(안전 규칙 우선 회수). 5회 컴팩션 후 회수율 96%, recall@50 100%(베이스라인 73%).
- **경계 국소 평가** (TRACE): 동일 환경 상태에서 압축 전후 쌍(pair) 롤아웃을 비교해 컴팩션 이벤트 하나하나를 검증. 최종 태스크 점수만으로는 압축 품질을 잴 수 없다는 문제의식.

### 요약이 항상 이득은 아니다

과학발견 코딩 에이전트에서 8가지 condenser를 비교한 연구(DiscoveryBench 60과제 × GPT-4o, 480회)는 더 냉정한 결과를 냈다.

- LLM 요약기는 토큰 비용을 **24~94% 증가**시켰다 (요약 호출 자체의 비용).
- 가설 품질을 유의미하게 올린 전략은 **없었다**.
- 도구 출력 마스킹만 8.6% 순절감. 최적 condenser는 도메인·태스크 길이에 따라 달랐다.

즉 "요약을 넣으면 좋아진다"는 가정은 태스크 의존적이며, 특히 짧은 태스크에서는 요약 오버헤드가 이득을 상쇄한다.

---

## 4. 표현 층: soft token · 인코더-디코더 · 컴파일된 메모리

### 4.1 Gist / soft token의 한계와 돌파

Gist token 계열(2023~)은 특수 토큰이 self-attention으로 컨텍스트 정보를 흡수하도록 학습한다(LLM-as-compressor). 2024년 종합 연구는 이 방식이 full attention 대비 세부 정보 손실이 크다는 점을 지적했고, 2026년 논문들은 세 방향으로 대응한다.

- **ComprExIT — Context Compression via Explicit Information Transmission (2026.02)**: attention에 암묵적으로 맡기는 대신 정보 전달 경로를 명시적으로 설계.
- **DAST**: 정보 밀도가 균일하지 않다는 전제에서 soft token을 동적으로 할당.
- **Sentence-Anchored Gist**: 문장 경계에 gist를 앵커링해 구조 보존.

### 4.2 대규모 사전학습 압축기

**End-to-End Context Compression at Scale (Goldstein·Goldblum·Izmailov 외, 2026.06)**
- **Latent Context Language Model (LCLM)**: 0.6B 인코더가 긴 토큰 시퀀스를 짧은 잠재 임베딩으로 매핑, 4B 디코더가 소비. 아키텍처 탐색 후 350B+ 토큰으로 지속 사전학습.
- 압축률 1:4 / 1:8 / 1:16 모델 패밀리. 기존 KV 압축 대비 일반 태스크 성능·압축 속도·피크 메모리 모두 개선.
- 에이전트 사용법: 압축된 긴 컨텍스트를 **훑다가(skim) 필요한 구간만 펼친다(expand on demand)**. 프로덕션 추론 엔진 호환.

이 논문의 의미는 "soft-token 압축은 오프라인 전처리용 소규모 실험"이라는 인식을 깼다는 점이다. 압축률 16배가 실제 규모에서 작동한다.

### 4.3 컴파일된 이식 가능 메모리

**Latent Context Compilation (2026.01)**
- 일회용 LoRA를 "컴파일러"로 써서 긴 컨텍스트를 **버퍼 토큰**으로 변환. 버퍼 토큰은 stateless라 frozen 베이스 모델과 동시 서빙 호환.
- 합성 QA 없이 **무작위 질의로 재구성을 정규화**하는 self-aligned 최적화. Llama-3.1-8B에서 16× 압축, 세부 사실과 추론 능력 보존.

---

## 5. 아키텍처 층: 테스트타임 최적화형 메모리와 KV 압축

### 5.1 Titans → Nested Learning/HOPE → 2026 파생

Titans(2025.01)는 attention을 단기 메모리, 테스트타임에 경사하강으로 갱신되는 신경 메모리를 장기 메모리로 두고 2M+ 토큰까지 확장했다. Nested Learning(2025)은 모델을 "컨텍스트 흐름에 대한 중첩 최적화 문제"로 재정의하고, 다중 시간척도 **Continuum Memory System(CMS)**과 압축 유발 surprise에 적응하는 자기 수정 갱신 규칙을 가진 **HOPE**를 제시했다.

핵심 관점: 옵티마이저의 모멘텀은 그래디언트 스트림을 고정 크기로 압축하고, Titans의 장기 메모리는 토큰 스트림을 가중치로 압축한다. **둘은 같은 연산**이다. 이 시각에서 컨텍스트 압축은 "학습"의 한 형태가 된다.

2026년 파생 연구:

**GradMem — Learning to Write Context into Memory with Test-Time Gradient Descent (ICML 2026, 2026.03)**
- 모델 가중치는 고정, **prefix 메모리 토큰**에 자기지도 재구성 손실로 경사하강. 손실 주도 쓰기 + 반복 오류 수정.
- 연상 key-value 회수에서 같은 메모리 용량의 forward-only 방법을 능가. **gradient step 수가 반복 forward write보다 메모리 용량을 훨씬 효율적으로 키운다**. bAbI·SQuAD 변형에서 메모리 인코딩 정보만으로 전이.
- Titans/HOPE와의 차이: 쓰는 대상이 fast weight가 아니라 토큰이다. 갱신 사상(loss-driven write)은 동일.

**NestedKV — Nested Memory Routing for Long-Context KV Cache Compression (EMNLP 2026, 2026.05)**
- Nested Learning의 CMS에서 영감. **global · block · sliding-window** 다중 시간척도 앵커를 유지하고, 토큰을 "multi-time-scale cosine anomaly"로 점수화. head-adaptive mixing + **surprise-gated token routing**, per-head 적응 예산. 학습 불필요.

| 조건 (Qwen3-4B) | NestedKV | KeyDiff |
|----------------|----------|---------|
| r=0.75, RULER | +19.10 | — |
| r=0.75, LongBench | +19.29 | — |
| r=0.95, LongBench | 37.32 | 17.55 |

- **Federated Nested Learning (2026.05)**: 자기참조 메모리를 연합학습으로 협업 학습해 테스트타임 적응.

### 5.2 KV 캐시 압축

ACL 2026 서베이 *System-Aware KV Cache Optimization*이 분야 지도 역할을 한다. 2026년 경향:

- **추론(reasoning) 특화**: Information-Aware KV Compression for Long Reasoning, Adaptive Mass-Segmented KV Compression — 긴 CoT에서 중요 토큰 분포가 문서 QA와 다르다는 관찰.
- **고압축률 양자화**: VQKV(벡터 양자화), DepthWeave-KV(층 간 잔차 분해).
- **의미 단위 보존**: ChunkKV(청크 단위), PyramidKV(층별 피라미드 예산).

---

## 6. 이론화: 압축을 수학 문제로

### 6.1 Context Compaction Theory (Anthropic·Harvard, 2026.08)

컴팩션을 처음으로 형식적으로 분석한 논문이다.

- **주장 1**: 질의 집합에 답하기 위한 최소 컴팩션 예산은 해당 문제의 **one-way communication complexity**와 같다. 컴팩션은 "과거의 나"가 "미래의 나"에게 보내는 일방향 메시지이므로 통신 복잡도가 그대로 하한이 된다.
- **주장 2**: **선택(selection)** 기반(상태 부분집합 유지 = KV 축출·토큰 가지치기)은 제한된 프로토콜 클래스이며, **생성(generation)** 기반(요약)이 특정 질의 집합에서 엄격히 우월하다.
- 두 게임이론 모델을 제시하고 Anthropic의 컴팩션 엔드포인트를 집합 멤버십 질의로 벤치마크한 사례 연구를 포함한다.

실무적 함의: 어떤 태스크가 "선택으로 충분한지" 아니면 "생성이 필요한지"를 질의 집합의 통신 복잡도로 판단할 수 있다.

### 6.2 Rate–Distortion View (2026.07)

- KV 축출·프롬프트 가지치기·아키텍처 상태 유계화·에이전트 메모리 통합을 **하나의 rate-distortion 문제**로 통합: "자원 예산 하에서 어떤 정보를 어떤 충실도로 남겨 하류 태스크 효용을 보존할 것인가".
- 7축 분류 체계로 스택 전 층의 방법을 균일하게 분류. 서빙 스택 기법을 에이전트 메모리로 전이하는 식의 **층 간 메커니즘 전이**를 제안.
- 측정 공백 지적: 단일 턴 압축은 꼼꼼히 측정되지만, **에이전트가 실제로 하는 반복 컴팩션은 거의 측정되지 않는다**. 모든 층에 일관된 예산 제약을 유지하는 벤치마크가 없다.

---

## 7. 프로덕션 현황: 코딩 에이전트의 컴팩션

Claude Code · Codex CLI · OpenCode · Amp의 구현을 비교한 공개 분석을 요약하면 다음과 같다.

| 항목 | Claude Code | Codex CLI | OpenCode | Amp |
|------|------------|-----------|----------|-----|
| 트리거 | ~95% 자동 + `/compact` | 모델별 토큰 한계의 95% | context_limit − output_limit | 자동 없음 (수동 handoff) |
| 요약 내용 | 완료 작업·진행 중·파일·다음 단계·요청 | "handoff" 관점, 결정·제약 포함 | 기술적 결정과 **이유** 명시 보존 | 보조 모델이 필요 정보만 추출 |
| 도구 출력 가지치기 | — | — | **있음** (최근 40k 보호) | — |
| 최근 메시지 보존 | — | ~20k 토큰 | ~20k 토큰 | — |
| 품질 경고 | — | 다회 컴팩션 시 저하 경고 | 동일 | 긴 대화 자체를 비권장 |
| 커스터마이즈 | 커스텀 컴팩션 지침 | 고정 템플릿 | 고정 템플릿 | — |

OpenHands는 *Context Condenser*라는 이름으로 오래된 이력을 LLM 요약으로 대체하며 SWE 태스크에서 동등 이상 성능을 보고한다. 공통 관찰은 사용자들이 품질 우려로 자동 임계치보다 **일찍 수동 컴팩션**을 선호한다는 점이고, 이는 §2.4 SelfCompact의 "언제 압축할지" 문제가 실무에서도 미해결임을 보여준다.

---

## 8. Titans/HOPE 재현 연구와의 접점

필자가 진행 중인 HOPE(Nested Learning) fast-weight 메모리 재현은 §1.2의 **아키텍처 층**에 속한다. 2층 residual MLP 연상 메모리를 청크 병렬 DGD(delta gradient descent)로 갱신하고 보존 게이트 $\alpha_t$와 학습률 $\eta_t$를 토큰별로 두는 구조인데, 이번 서베이가 시사하는 실험 방향은 세 가지다.

**(1) 반복 압축 안정성 벤치마크.**
§6의 두 이론 논문이 공통으로 지적한 공백이다. fast-weight 메모리를 여러 청크 경계에 걸쳐 굴릴 때 초기 청크 정보의 보존율이 어떻게 감쇠하는지 측정하면, Compaction Cliff(§3)의 "5회 컴팩션 후 10%" 같은 수치와 직접 비교 가능한 결과가 된다. 게이트 $\alpha$의 누적곱 $A_C=\prod_s \alpha_s$가 바로 이 감쇠의 해석 가능한 지표다.

**(2) GradMem 대비: gradient step 수 vs forward write.**
GradMem은 같은 메모리 예산에서 gradient step을 늘리는 것이 forward write 반복보다 용량을 효율적으로 키운다고 보였다. 청크 병렬 갱신 구현에서는 청크당 grad를 한 번만 계산하므로, "청크 내 step 수"를 늘렸을 때 회수 정확도가 어떻게 스케일하는지 비교하는 실험이 바로 가능하다.

**(3) Surprise 게이트 × 타입별 보존.**
NestedKV의 surprise-gated routing과 Compaction Cliff의 타입별 보존을 결합하면, "규칙·제약 같은 정보는 낮은 망각(높은 $\alpha$)으로 보호하고, 일회성 관측은 빠르게 잊는" 게이트 설계가 자연스럽다. HOPE의 자기 수정 갱신 규칙이 이미 압축 유발 surprise에 반응하도록 설계되어 있으므로, 여기에 정보 타입 신호를 추가 입력으로 넣는 것이 최소 변경이다.

---

## 9. 정리: 2026년의 열린 문제

1. **반복 컴팩션 측정** — 에이전트가 실제로 하는 다회 압축을 일관된 예산으로 측정하는 벤치마크가 없다.
2. **질의 인식 압축** — 모든 층이 질의를 알기 전에 버린다. 무손실 아카이브(ARC) + 학습된 요약의 결합이 유력한 방향.
3. **안전 제약 보존** — 타입별 보존 정책이 초기 해답이지만, 적대적 삽입 공격에는 아직 취약.
4. **압축률과 모델 능력의 관계** — AdaCoM의 발견(강한 모델은 덜, 약한 모델은 더 압축)이 일반화되는지.
5. **층 간 통합** — 에이전트 요약·soft token·fast weight 메모리를 하나의 rate-distortion 프레임에서 공동 설계하는 시스템은 아직 없다.

---

## 10. 관련 블로그 포스트

- [해마 메모리와 attention 서베이](hippocampal-memory-attention-survey.md) — 생물학적 메모리 통합과 압축의 유사성
- [Bayesian Surprise](bayesian-surprise.md) — Titans/HOPE·NestedKV의 surprise 게이트 배경
- [최신 RAG 동향 (2026)](rag-survey-2026.md) — 외부 메모리 오프로딩 관점
- [AI Scientist 방법론 1년 종합](scientist-agents-2025-2026-report.md) — 장기 실행 연구 에이전트의 컨텍스트 문제

---

## 11. 참고 자료

### 에이전트 컴팩션 — RL 학습
- [CompactionRL: RL with Context Compaction for Long-Horizon Agents (arXiv:2607.05378)](https://arxiv.org/abs/2607.05378)
- [Learning Agent-Compatible Context Management (AdaCoM) (arXiv:2605.30785)](https://arxiv.org/abs/2605.30785)
- [ACON: Optimizing Context Compression for Long-horizon LLM Agents (arXiv:2510.00615)](https://arxiv.org/abs/2510.00615)
- [Dynamic Long Context Reasoning over Compressed Memory via End-to-End RL (arXiv:2602.08382)](https://arxiv.org/abs/2602.08382)
- [Remember When It Matters: Proactive Memory Agent (arXiv:2607.08716)](https://arxiv.org/abs/2607.08716)

### 에이전트 컴팩션 — 무손실 회수·메타인지·서빙
- [Addressable Recall Compaction (ARC) (arXiv:2607.25066)](https://arxiv.org/abs/2607.25066)
- [LLM Agents Are Latent Context Managers (VISTA) (arXiv:2606.30005)](https://arxiv.org/abs/2606.30005)
- [Self-Compacting Language Model Agents (arXiv:2606.23525)](https://arxiv.org/abs/2606.23525)
- [Parallel Context Compaction for Long-Horizon LLM Agent Serving (arXiv:2605.23296)](https://arxiv.org/abs/2605.23296)
- [TokenPilot: Cache-Efficient Context Management (arXiv:2606.17016)](https://arxiv.org/abs/2606.17016)

### 안전성·안정성
- [Governance Decay: How Context Compaction Silently Erases Safety Constraints (arXiv:2606.22528)](https://arxiv.org/abs/2606.22528)
- [The Compaction Cliff in Long-Running AI Agent Memory (arXiv:2608.22752)](https://arxiv.org/abs/2608.22752)
- [TRACE: Toward Reliable Context Compression for Long-Horizon Agents (arXiv:2608.06503)](https://arxiv.org/abs/2608.06503)
- [Evaluating Memory Condensation Strategies for Coding Agents in Scientific Discovery (arXiv:2605.18854)](https://arxiv.org/abs/2605.18854)

### 벤치마크·서베이
- [LOCA-bench: Language Agents Under Controllable Context Growth (arXiv:2602.07962)](https://arxiv.org/abs/2602.07962)
- [Memory for Autonomous LLM Agents: Mechanisms, Evaluation, Frontiers (arXiv:2603.07670)](https://arxiv.org/abs/2603.07670)
- [Rethinking Memory Mechanisms of Foundation Agents in the Second Half (arXiv:2602.06052)](https://arxiv.org/abs/2602.06052)
- [Towards Efficient LLM Serving: System-Aware KV Cache Optimization (ACL 2026)](https://github.com/jjiantong/Awesome-KV-Cache-Optimization)

### 표현 층 (soft token · 컴파일 메모리)
- [End-to-End Context Compression at Scale (LCLM) (arXiv:2606.09659)](https://arxiv.org/abs/2606.09659)
- [Latent Context Compilation (arXiv:2602.21221)](https://arxiv.org/abs/2602.21221)
- [Context Compression via Explicit Information Transmission (arXiv:2602.03784)](https://arxiv.org/abs/2602.03784)
- [DAST: Dynamic Allocation of Soft Tokens (arXiv:2502.11493)](https://arxiv.org/abs/2502.11493)
- [Sentence-Anchored Gist Compression (arXiv:2511.08128)](https://arxiv.org/abs/2511.08128)
- [A Silver Bullet or a Compromise for Full Attention? Gist Token Study (arXiv:2412.17483)](https://arxiv.org/abs/2412.17483)

### 아키텍처 층 (테스트타임 메모리 · KV)
- [Titans: Learning to Memorize at Test Time (arXiv:2501.00663)](https://arxiv.org/abs/2501.00663)
- [Test-time Regression: Unifying Framework for Associative Memory (arXiv:2501.12352)](https://arxiv.org/abs/2501.12352)
- [GradMem: Learning to Write Context into Memory with Test-Time GD (arXiv:2603.13875)](https://arxiv.org/abs/2603.13875)
- [NestedKV: Nested Memory Routing for KV Cache Compression (arXiv:2605.26678)](https://arxiv.org/abs/2605.26678)
- [Federated Nested Learning (arXiv:2605.16350)](https://arxiv.org/abs/2605.16350)
- [Information-Aware KV Cache Compression for Long Reasoning (arXiv:2606.26875)](https://arxiv.org/abs/2606.26875)
- [VQKV: Vector-Quantization KV Compression (arXiv:2603.16435)](https://arxiv.org/abs/2603.16435)

### 이론
- [Context Compaction Theory (arXiv:2608.01326)](https://arxiv.org/abs/2608.01326)
- [What to Keep, What to Forget: A Rate–Distortion View of Memory Compaction (arXiv:2607.08032)](https://arxiv.org/abs/2607.08032)
- [Recursive Language Models (arXiv:2512.24601)](https://arxiv.org/abs/2512.24601)

### 프로덕션
- [Context Compaction Research: Claude Code, Codex CLI, OpenCode, Amp (badlogic gist)](https://gist.github.com/badlogic/cd2ef65b0697c4dbe2d13fbecb0a0a5f)
- [Claude Cookbook — Context engineering: memory, compaction, tool clearing](https://platform.claude.com/cookbook/tool-use-context-engineering-context-engineering-tools)
- [OpenHands — Context Condenser](https://docs.openhands.dev/sdk/guides/context-condenser)
- [Dive into Claude Code: Design Space of AI Agent Systems (arXiv:2604.14228)](https://arxiv.org/abs/2604.14228)
