---
title: "[서베이] 저망각 파인튜닝 연구 동향 (2026) — 온폴리시·희소 갱신·학습률·측정"
date: 2026-09-16
tags: ["연구노트", "서베이", "파인튜닝", "연속학습", "CatastrophicForgetting", "LLM", "2026"]
categories: ["ML/AI"]
summary: "파인튜닝에서 파국적 망각(catastrophic forgetting)을 줄인다고 주장하는 2025~2026 논문 총정리. RL's Razor·SDFT 등 온폴리시 계열, Sparse Memory Finetuning·TFGN 등 갱신 격리 계열, FINCH·옵티마이저 일관성 등 최적화 계열, 샘플 단위 측정 계열로 나누고, 'RL도 결국 잊는다'는 2026년 여름 반론까지 정리. 모듈형 평생학습 어댑터 관점의 시사점 포함."
math: true
toc: true
draft: false
---

## Executive Summary

파인튜닝은 새 능력을 얻는 대신 기존 능력을 잃는다. 이 **파국적 망각(catastrophic forgetting)**은 연속학습의 오래된 문제지만, LLM 시대에 들어 "리플레이·EWC를 붙이는" 처방 위주에서 **"왜, 어떤 조건에서 잊는가"를 정량화하고 학습 목적·갱신 구조 자체를 바꾸는** 연구로 이동했다. 2025년 하반기부터 "저망각 파인튜닝"이 독립 주제로 자리잡았고, 논문은 크게 네 갈래로 나뉜다.

| 계열 | 핵심 주장 | 대표 논문 |
|------|----------|----------|
| **① 온폴리시 목적** | 망각은 알고리즘이 아니라 기저 모델과의 KL 거리로 결정되며, 온폴리시 학습이 KL 최소 해를 고른다 | RL's Razor, SDFT, RFT-mitigates, Mechanistic origins |
| **② 갱신 격리** | 망각의 원인은 공유 밀집 표현 덮어쓰기. 새 지식이 쓰이는 파라미터를 희소·모듈로 격리하면 구조적으로 억제 | Sparse Memory Finetuning(Meta), TFGN |
| **③ 최적화 하이퍼파라미터** | 학습률·옵티마이저·랭크 선택이 망각의 대부분을 설명. 스케줄만으로 90%대 감소 | FINCH, Optimizer-Model Consistency, LoRA 계열 |
| **④ 측정** | 과제 평균은 망각과 역전이를 상쇄해 숨긴다. 샘플 단위 전이로 재야 한다 | Mapping Forgetting at Scale, Spurious Forgetting |

한 문장 결론:

> **"온폴리시가 덜 잊는다"**는 가장 강한 합의이고, **"희소·모듈형 갱신으로 격리한다"**가 구조적으로 가장 깨끗하다. 단, 2026년 여름 **"RL도 결국 잊는다"**는 반론이 나와 원인(KL 페널티 vs 온폴리시 데이터)과 한계가 논쟁 중이다.

세 레버(온폴리시 목적 · 갱신 격리 · 손실 적응 학습률)는 서로 직교하므로 결합 가능하다.

---

## 1. 문제 재정의: 망각은 무엇으로 결정되는가

전통적 연속학습은 망각을 "과제 A 정확도가 과제 B 학습 후 얼마나 떨어지는가"로 재고, 리플레이·정규화(EWC)·그래디언트 투영·아키텍처 분리로 처방했다. 2025~2026 LLM 연구는 질문을 바꿨다.

- **무엇이 망각량을 예측하는가?** → 새 과제 분포에서 측정한 $D_{KL}(\pi_{\text{ft}} \| \pi_{\text{base}})$ (RL's Razor)
- **어느 파라미터가 망각을 일으키는가?** → 공유 밀집 표현, 특히 attention 회로 (Mechanistic origins, Mechanistic Analysis 2026.01)
- **어느 목적함수가 옛 분포를 보존하는가?** → forward-KL은 질량 붕괴, reverse-KL은 유계 드리프트 (Quantitative Characterization)
- **망각은 비가역적인가?** → 상당 부분은 회복 가능한 "드리프트" 또는 "과제 정렬 손실" (ES+AWD, Spurious Forgetting)

이 재정의가 아래 네 계열의 공통 배경이다.

---

## 2. 계열 ①: 온폴리시 목적이 망각을 줄인다

### 2.1 RL's Razor (MIT Improbable AI, NeurIPS 2025)

출발점이 된 논문. 새 과제 정확도를 **동일하게 맞춘** 상태에서 비교해도 RL은 SFT보다 일관되게 덜 잊는다. 핵심 주장은 두 가지다.

1. 망각 정도는 훈련 알고리즘이 아니라 **새 과제 분포에서 평가한 fine-tuned 정책과 base 정책의 KL 거리**로 결정된다.
2. 과제를 푸는 여러 해(solution) 중 온폴리시 RL은 **KL 최소 해**로 편향되고, SFT는 base에서 임의로 먼 분포로 수렴할 수 있다. 이것이 "RL's Razor"다.

LLM 실험과 통제된 toy 실험, 그리고 온폴리시 갱신이 작은 KL 변화를 낳는 이유에 대한 이론을 제시했다.

### 2.2 Self-Distillation Enables Continual Learning (SDFT, 2026.01, ICML 2026)

같은 그룹(Shenfeld·Damani·Hübotter·Agrawal)의 확장. "RL이 덜 잊는 건 온폴리시 덕분인데, 보상 없이 **데모 데이터만 있어도** 온폴리시로 학습할 수 있는가?"에 대한 답이다.

- 데모를 in-context로 조건화한 모델을 **교사**로 삼고, 학생(같은 모델, 조건 없음)의 온폴리시 샘플에 대해 **reverse-KL**로 교사에 맞춘다.
- SFT보다 새 과제 정확도가 높고 망각은 크게 줄어, 순차 학습에서 단일 모델이 여러 스킬을 성능 퇴행 없이 누적한다.
- 비용: 궤적 생성과 토큰별 손실 계산으로 FLOPs·wall-clock 증가.

이 논문의 의의는 "온폴리시 증류가 데모 기반 연속학습의 실용적 경로"임을 보인 것이다. SFT는 본질적으로 오프폴리시라는 점을 문제의 근원으로 지목한다.

### 2.3 Reinforcement Fine-Tuning Naturally Mitigates Forgetting (2025.07, 2026.06 개정)

Qwen2.5-VL-7B-Instruct 멀티모달 연속 후학습에서 RFT가 **멀티태스크 학습에 준하는 성능**을 유지하는 반면 SFT는 일반 능력을 심하게 망각. 원인을 KL 페널티가 아닌 **선택적 갱신 메커니즘**(보상 신호가 있는 샘플만 갱신)으로 돌리며, 학습 가능한 샘플을 우선하는 RIF-RFT 필터링을 제안한다.

### 2.4 기계론적·이론적 뒷받침

| 논문 | 기여 |
|------|------|
| **Mechanistic origins of catastrophic forgetting** (2026.05) | 헤드 단위 "differential circuit vulnerability" 지표. Qwen2.5-3B에서 SFT는 빠르게 적응하나 회로 파괴가 크고, RL은 느리지만 기저 회로를 더 보존. 행동 수준(KL) 설명을 회로 수준으로 확장 |
| **RL Fine-Tuning Heals OOD Forgetting in SFT** (2025.09) | SFT가 깨뜨린 OOD 능력을 후속 RL이 회복. 가중치 특이벡터의 **회전**이 관건, 회전 제어로 OOD 강건성 개선 가능 |
| **A Quantitative Characterization of Forgetting in Post-Training** (2026.03) | 가우시안 혼합 추상화. **forward-KL은 옛 분포 질량을 0으로 붕괴("mass forgetting")**, **reverse-KL은 질량 보존 + 모드 분리도에 지수적으로 감쇠하는 드리프트**만 발생. SDFT·TTT-Discover·OAPL이 옛 지식을 보존하는 명시적 조건 도출 |
| **Mechanistic Analysis of Catastrophic Forgetting** (2026.01) | 2026 초 SOTA 6개 아키텍처 분석. 세 메커니즘: attention 가중치의 그래디언트 간섭, 중간층 표현 드리프트, 손실 지형 평탄화 |

### 2.5 반론: RL도 잊는다

2026년 여름 두 논문이 "RL은 본질적으로 덜 잊는다"는 가정에 제동을 걸었다.

**RL Forgets! Towards Continual Policy Optimization (2026.07)**
- Qwen3-VL-8B 연속 후학습에서 표준 RL(GRPO 등)도 **유의미하게 망각**한다.
- 원인은 **목적 불일치**: KL 정규화가 *현재 과제* 데이터에서만 평가되는데, 망각은 *이전 과제* 분포에서의 행동 드리프트로 발생한다.
- 처방 CPO: 이전 과제 KL 제약(계산 불가)을 **희소 파라미터 이동 정규화**로 완화. 옛 데이터 저장 없이 망각 13.7% 감소, 사전학습 능력 7.0% 향상. MRCL 벤치마크 공개.

**Overcoming Catastrophic Forgetting in Visual Continual Learning with RFT (2026.05)**
- RFT가 SFT보다 낫지만 망각은 무시할 수 없는 수준으로 남는다.
- 병목은 **궤적 수준 드리프트 무지(Trajectory-level Drift Agnosticism)**: 같은 보상을 받는 롤아웃들 사이에서 이전 과제 정책과의 KL이 크게 다르고, 이 편차가 순차 과제 망각과 강하게 상관한다.

**논쟁의 현재 상태.** RL의 저망각 원인을 (a) KL 페널티(Shenfeld et al.), (b) 온폴리시 데이터 생성 자체(Zhang et al., Chen et al.)로 보는 두 입장이 공존하며, Chen·Lai 등의 경험적 평가는 KL 정규화의 역할이 미미하다고 보고한다. "RL이 **덜** 잊는다"는 재현되지만 "**안** 잊는다"는 아니다.

---

## 3. 계열 ②: 갱신을 격리한다

이 계열의 논리는 단순하다. 망각은 **공유 밀집 표현을 덮어쓰기** 때문에 생기므로, 새 지식이 쓰이는 파라미터 위치를 **라우팅으로 격리**하면 망각이 구조적으로 억제된다. 모듈형 평생학습 어댑터(비병합 저랭크 bypass + per-token 라우터) 설계와 가장 가까운 선행연구다.

### 3.1 Continual Learning via Sparse Memory Finetuning (Meta, 2025.10)

- 메모리 레이어(key-value 슬롯) 모델에서, 새 지식에 대해 **사전학습 데이터 대비 높게 활성화되는 슬롯만** 갱신한다.
- 같은 수준의 새 지식 습득에서 NaturalQuestions F1 하락:

| 방법 | NQ F1 하락 |
|------|-----------|
| Full finetuning | 89% |
| LoRA | 71% |
| **Sparse Memory Finetuning** | **11%** |

- 후속 연구
  - **Improving Sparse Memory Finetuning** (2026.04): KL 발산 기반의 이론적 슬롯 선택. 배경 분포 대비 정보량이 높은 토큰의 슬롯을 우선 갱신. Qwen-2.5-0.5B에 메모리 모듈을 **사후 장착(retrofit)**해도 최소 망각으로 사실 지식 습득.
  - **SMF as a Low-Forgetting Alternative to LoRA and Full FT** (2026.05): MedMCQA에서 SMF +2.5pt(망각 프로브는 기저 대비 ~1pt 이내), LoRA·Full FT는 더 큰 이득이지만 WikiText PPL·TriviaQA 모두 명확한 드리프트. KL vs TF-IDF 슬롯 선택이 두 망각 지표를 다르게 트레이드오프.

### 3.2 TFGN: Task-Free, Replay-Free Continual Pre-Training (2026.05)

- 트랜스포머 코어를 건드리지 않는 **입력 조건부 파라미터 효율 갱신 오버레이**. Read/Write 분해: forward는 완전 밀집, 도메인 간 갱신은 **이전 도메인 부분공간에 쓰이지 않도록** 구조화.
- 리플레이 버퍼·과제 ID·Fisher 페널티 없음. 398M / 739M / ~9B, 6개 텍스트 도메인, 단계당 1B 토큰.

| 지표 | 값 |
|------|-----|
| Backward transfer (LLaMA 3.1 8B) | −0.007 |
| 도메인 간 그래디언트 L2 직교성 | ≥ 99.59% |
| Forward transfer: Python 학습만으로 JavaScript PPL | −26.8% (8B), −62.0% (GPT-2 M) |
| 메타컨트롤 확장 시 추가 망각 감소 (398M) | 81% |

파인튜닝이 아닌 **연속 사전학습** 규모에서 격리 접근이 작동함을 보인 점이 다르다.

### 3.3 이 계열의 한계

- 이득의 크기가 밀집 갱신보다 작다(SMF의 +2.5pt vs LoRA의 더 큰 이득). "덜 배우고 덜 잊는" 트레이드오프의 구조적 버전.
- 슬롯/모듈 **선택 규칙**(활성화 대비, KL, TF-IDF)이 성능을 좌우하며 아직 표준이 없다.
- 라우터 자체의 망각·드리프트는 별도 문제로 남는다.

---

## 4. 계열 ③: 하이퍼파라미터가 망각을 결정한다

### 4.1 LoRA 논쟁

- **LoRA Learns Less and Forgets Less** (TMLR 2024): 표준 저랭크에서 LoRA는 Full FT보다 덜 배우지만, 타깃 도메인 밖 성능을 더 잘 유지하고 weight decay·dropout보다 망각을 잘 억제.
- **LoRA vs Full FT: An Illusion of Equivalence** (2025): **동일 적합도**에서도 LoRA가 덜 잊는다 → 단순 과소적합 효과가 아님. 단, LoRA는 사전학습 가중치에 없는 "침입 차원(intruder dimensions)"을 만든다.
- **Mitigating Forgetting in LoRA / LaLoRA** (2025.12): LoRA가 덜 잊는 것은 **보편적이지 않다**. 하이퍼파라미터에 따라 사전학습 가중치와 어긋난 고특이값 방향이 생겨 망각이 커진다. LoRA 가중치에만 Laplace 근사를 적용해 고곡률 방향 갱신을 제한(LaLoRA).
- **Optimizer-Model Consistency** (2026.05): **사전학습과 같은 옵티마이저**로 Full FT하면 LoRA보다 나은 학습·망각 트레이드오프. 옵티마이저는 암묵적 정규화자이며(Muon vs AdamW 비교, Muon은 소량 데이터에서 과도한 암기 경향), 기존 LoRA 논문들의 결론 차이는 **학습률 선택 차이**로 설명된다.

### 4.2 FINCH: Fine-Tuning Without Forgetting via Loss-Adaptive Learning Rates (2026.05)

- 이론: 망각은 $\eta \cdot \sqrt{\mathcal{L}_{\text{current}}}$ (학습률 × √현재 손실)로 유계.
- 처방: **고손실 배치에서 학습률을 낮추고 수렴할수록 올리는** 스케줄. 목적함수는 그대로. 고손실 토큰을 억제하는 기존 방법(Low-Perplexity Token Learning 등)과 달리 새 과제 학습 신호를 보존.

| 결과 | 값 |
|------|-----|
| 평균 망각 감소 | 93% |
| Qwen3-4B TruthfulQA 하락 | 5× 축소 |
| HaluEval | 하락 → 역전 |
| 새 과제 성능 | 표준 FT와 동등, 캘리브레이션 더 보존 |

### 4.3 회의적 결과들

- **Fine-tuning MLLMs Without Forgetting Is Easier Than You Think** (2026.03): 2×2(ID/OOD 이미지 × 텍스트) 실험에서 **낮은 학습률·파라미터 제약·데이터 혼합**만으로 복잡한 연속학습 방법을 이긴다. ID 이미지 + OOD 텍스트에서 나타나는 "과제 특화 과적합"이 별개 현상.
- **Overcoming Forgetting with Evolution Strategies + Anchored Weight Decay** (2026.05): 이전 과제 성능이 학습 중 **회복**되는 현상을 근거로 망각을 "비가역적 손실"이 아닌 **드리프트**로 재규정. 초기 파라미터로 끌어당기는 AWD가 큰 ES 집단 크기와 유사한 효과. RL에서도 같은 드리프트가 발생.
- **MoFO** (2024.07): 모멘텀 크기 상위 파라미터만 갱신하는 옵티마이저 수준 필터링. 계열 ②와 ③의 중간.

---

## 5. 계열 ④: 측정 — 저망각 주장을 어떻게 검증하나

**Mapping Post-Training Forgetting in Language Models at Scale (2025.10)**
- 과제 평균 대신 **샘플 단위 전이**: 1→0(전에 맞고 후에 틀림)=망각, 0→1=역전이(backward transfer). 다지선다에는 우연 보정 변형.
- 발견: 도메인 연속 사전학습은 중간 망각·최소 역전이. base 모델 RL/SFT는 수학·논리에서 중~대 역전이와 저~중 망각. **모델 머징은 망각을 안정적으로 줄이지 못함**. "not all forgetting is equal" — 평균은 반대 효과를 상쇄해 숨긴다.

**Spurious Forgetting in Continual Learning of Language Models (2025.01)**
- 초기 망각의 상당 부분은 지식 손실이 아닌 **과제 정렬(alignment) 손실**. 소량 재정렬로 회복되므로, 이를 "진짜 망각"으로 세면 방법 간 비교가 왜곡된다.

**검증 체크리스트** (저망각 주장을 볼 때)
1. 새 과제 정확도를 **매칭**한 상태에서 비교했는가? (RL's Razor 프로토콜)
2. 학습률 **스윕**을 했는가, 한 점만 봤는가? (Optimizer-Model Consistency의 비판)
3. 샘플 단위 1→0 전이로 재었는가, 과제 평균인가?
4. 소량 재정렬 후에도 손실이 남는가? (spurious vs genuine)
5. 옛 과제 분포에서의 KL/행동 드리프트를 직접 측정했는가? (RL Forgets!의 비판)

---

## 6. 정리: 세 개의 직교 레버

| 레버 | 무엇을 바꾸나 | 대표 | 비용 |
|------|-------------|------|------|
| **온폴리시 목적** | 손실 함수 (forward-KL → reverse-KL, 자기 샘플) | RL's Razor, SDFT | 궤적 생성 FLOPs |
| **갱신 격리** | 어느 파라미터에 쓰는가 (희소 슬롯·모듈·직교 부분공간) | SMF, TFGN | 이득 크기 감소, 선택 규칙 설계 |
| **손실 적응 학습률** | 얼마나 크게 쓰는가 | FINCH, 옵티마이저 일관성 | 거의 없음 |

셋은 서로 다른 축을 건드리므로 **결합**이 자연스럽다. 예컨대 모듈형 어댑터(격리) 위에 SDFT 목적(온폴리시)으로 학습하고 FINCH 스케줄(학습률)을 얹는 구성은 아직 논문에 없다.

### 모듈형 평생학습 어댑터 관점의 시사점

1. **구조적 선행연구**는 계열 ②(SMF, TFGN). "공유 밀집 표현을 덮어쓰지 않는다"는 논리가 동일하며, 라우터 기반 격리는 SMF의 활성화 기반 슬롯 선택을 일반화한 것으로 위치 지을 수 있다.
2. **어댑터 학습 목적**으로는 SDFT가 바로 얹을 수 있는 후보. 데모만 있어도 온폴리시로 학습해 어댑터 내부 망각(같은 어댑터에 여러 스킬 누적 시)을 줄일 수 있다.
3. **검증 프로토콜**은 계열 ④를 따라야 한다. 특히 "정확도 매칭 + 학습률 스윕 + 샘플 단위 전이"를 갖추지 않은 저망각 주장은 Optimizer-Model Consistency 논문의 비판에 그대로 노출된다.
4. **라우터 드리프트**는 미해결 문제다. RL Forgets!의 "현재 과제에서만 KL을 재는 목적 불일치" 지적은 라우터 학습에도 똑같이 적용된다.

---

## 7. 열린 문제

1. RL 저망각의 원인이 KL 페널티인가 온폴리시 데이터인가 — 실험 설계가 갈라놓지 못하고 있다.
2. 옛 과제 분포에서의 드리프트를 **옛 데이터 없이** 제어하는 방법 (CPO의 희소 이동 정규화가 첫 시도).
3. 격리 계열의 "덜 배움" 트레이드오프를 좁히는 슬롯/모듈 선택 규칙.
4. 스케일 의존성 — 대부분 0.5B~8B 결과. 계열 ④가 대규모 instruction tuning에서 "혼합 결과"를 보고한 것이 경고.
5. 세 레버를 결합한 체계적 어블레이션.

---

## 8. 관련 블로그 포스트

- [Context Condensation 연구 동향 (2026)](context-condensation-survey-2026.md) — 컨텍스트 압축도 "무엇을 잊을지"의 문제. 테스트타임 메모리 갱신과 연속학습의 접점
- [DPO 리뷰](dpo-review.md) — reverse-KL 목적의 배경
- [No-BP 방법 서베이](no-bp-methods-survey.md) — ES 등 비역전파 최적화
- [AI Scientist 방법론 1년 종합](scientist-agents-2025-2026-report.md)

---

## 9. 참고 자료

### 계열 ① 온폴리시 목적
- [RL's Razor: Why On-Policy RL Forgets Less (arXiv:2509.04259, NeurIPS 2025)](https://arxiv.org/abs/2509.04259)
- [Self-Distillation Enables Continual Learning (arXiv:2601.19897, ICML 2026)](https://arxiv.org/abs/2601.19897)
- [Reinforcement Fine-Tuning Naturally Mitigates Forgetting in Continual Post-Training (arXiv:2507.05386)](https://arxiv.org/abs/2507.05386)
- [Mechanistic origins of catastrophic forgetting: why RL preserves circuits better than SFT (arXiv:2605.28860)](https://arxiv.org/abs/2605.28860)
- [RL Fine-Tuning Heals OOD Forgetting in SFT (arXiv:2509.12235)](https://arxiv.org/abs/2509.12235)
- [A Quantitative Characterization of Forgetting in Post-Training (arXiv:2603.12163)](https://arxiv.org/abs/2603.12163)
- [Mechanistic Analysis of Catastrophic Forgetting in LLMs During Continual Fine-tuning (arXiv:2601.18699)](https://arxiv.org/abs/2601.18699)
- [Reinforcement Learning via Self-Distillation (arXiv:2601.20802)](https://arxiv.org/abs/2601.20802)

### 계열 ① 반론
- [RL Forgets! Towards Continual Policy Optimization (arXiv:2607.04364)](https://arxiv.org/abs/2607.04364)
- [Overcoming Catastrophic Forgetting in Visual Continual Learning with RFT (arXiv:2605.09640)](https://arxiv.org/abs/2605.09640)

### 계열 ② 갱신 격리
- [Continual Learning via Sparse Memory Finetuning (arXiv:2510.15103)](https://arxiv.org/abs/2510.15103)
- [Improving Sparse Memory Finetuning (arXiv:2604.05248)](https://arxiv.org/abs/2604.05248)
- [Sparse Memory Finetuning as a Low-Forgetting Alternative to LoRA and Full Finetuning (arXiv:2605.03229)](https://arxiv.org/abs/2605.03229)
- [TFGN: Task-Free, Replay-Free Continual Pre-Training (arXiv:2605.15053)](https://arxiv.org/abs/2605.15053)
- [MoFO: Momentum-Filtered Optimizer (arXiv:2407.20999)](https://arxiv.org/abs/2407.20999)

### 계열 ③ 최적화·하이퍼파라미터
- [LoRA Learns Less and Forgets Less (arXiv:2405.09673, TMLR 2024)](https://arxiv.org/abs/2405.09673)
- [LoRA vs Full Fine-tuning: An Illusion of Equivalence (arXiv:2410.21228)](https://arxiv.org/abs/2410.21228)
- [LaLoRA: Mitigating Forgetting in Low Rank Adaptation (arXiv:2512.17720)](https://arxiv.org/abs/2512.17720)
- [Optimizer-Model Consistency: Full Finetuning with the Same Optimizer as Pretraining Forgets Less (arXiv:2605.06654)](https://arxiv.org/abs/2605.06654)
- [FINCH: Fine-Tuning Without Forgetting via Loss-Adaptive Learning Rates (arXiv:2605.20005)](https://arxiv.org/abs/2605.20005)
- [Fine-tuning MLLMs Without Forgetting Is Easier Than You Think (arXiv:2603.14493)](https://arxiv.org/abs/2603.14493)
- [Overcoming Forgetting in LLM Fine-Tuning with Evolution Strategies (arXiv:2605.30148)](https://arxiv.org/abs/2605.30148)
- [Mitigating Forgetting in LLM Fine-Tuning via Low-Perplexity Token Learning (arXiv:2501.14315)](https://arxiv.org/abs/2501.14315)
- [Improved SFT for LLMs to Mitigate Catastrophic Forgetting (arXiv:2506.09428)](https://arxiv.org/abs/2506.09428)

### 계열 ④ 측정
- [Mapping Post-Training Forgetting in Language Models at Scale (arXiv:2510.17776)](https://arxiv.org/abs/2510.17776)
- [Spurious Forgetting in Continual Learning of Language Models (arXiv:2501.13453)](https://arxiv.org/abs/2501.13453)
- [Revisiting Catastrophic Forgetting in LLM Tuning (arXiv:2406.04836)](https://arxiv.org/abs/2406.04836)

### 연속 사전학습 (배경)
- [Revisiting Replay and Gradient Alignment for Continual Pre-Training (arXiv:2508.01908)](https://arxiv.org/abs/2508.01908)
- [The Finetuner's Fallacy: When to Pretrain with Your Finetuning Data (arXiv:2603.16177)](https://arxiv.org/abs/2603.16177)
- [Bring Your Own Knowledge: A Survey of LLM Knowledge Expansion (arXiv:2502.12598)](https://arxiv.org/abs/2502.12598)
