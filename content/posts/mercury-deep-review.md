---
title: "[논문 리뷰] Mercury 심화 — 최초의 상용 Diffusion LLM, 학습/추론과 벤치마크 완전 해부"
date: 2026-05-15
tags: ["논문리뷰", "Diffusion", "LLM", "Mercury", "InceptionLabs"]
categories: ["ML/AI"]
summary: "Mercury(Inception Labs) 심화 리뷰. 세계 최초 상용 규모 diffusion LLM. SEDD·MDLM 저자들이 만든 제품. 아키텍처, 학습(공개 범위), coarse-to-fine 병렬 추론, 1109 tok/s 속도, 코딩 벤치마크 4개 테이블 전 수치를 논문 기반으로 해부 — 비공개 사항도 솔직히 명시."
math: true
toc: true
draft: false
---

## 논문 정보

- **제목**: Mercury: Ultra-Fast Language Models Based on Diffusion
- **저자**: Inception Labs 팀 (Samar Khanna 외) + 창업자/어드바이저 **Stefano Ermon, Aditya Grover, Volodymyr Kuleshov**
- **arXiv**: 2506.17298 (2025.06)
- **제품 출시**: Mercury Coder는 2025.03.08 먼저 출시, 기술 보고서는 6월 공개

> 이 글은 [Diffusion LM 서베이](diffusion-language-models-survey.md)의 Mercury 심화 편이다. 상용 제품이라 비공개 항목이 많으므로, **공개된 것과 비공개된 것을 명확히 구분**한다.

---

## 1. 한 줄 요약

> **SEDD·MDLM 논문 저자들이 학술 알고리즘을 상용 제품으로 끌어올린 세계 최초의 상용 규모 디퓨전 LLM — H100에서 1109 tok/s.**

---

## 2. Mercury란 무엇인가

### 학술 → 상용의 다리

기존 디퓨전 LLM(D3PM, SEDD, MDLM, LLaDA)은 모두 학술 규모. Mercury는 **상용 제품으로 배포된 최초**.

### 창업자 배경 (핵심 서사)

| 창업자 | 배경 |
|--------|------|
| **Stefano Ermon** (Stanford) | 이미지 디퓨전 공동 발명자, **SEDD** 핵심 인물 |
| **Volodymyr Kuleshov** (Cornell) | **MDLM** 등 masked diffusion 주역 |
| **Aditya Grover** (UCLA) | 생성 모델·의사결정 모델 |

세 명이 10년+ 함께 연구. Inception Labs는 약 5천만 달러 펀딩.

> **의의**: SEDD + MDLM 저자들이 직접 학술 알고리즘을 상용화. "디퓨전 LM의 wall-clock 속도 개선 실패"라는 학계 미해결 과제를 실제 제품에서 해결.

### 라인업

- **Mercury Coder** — 코딩 특화, Mini / Small 두 변형
- **Mercury** — 일반 채팅 모델
- **Mercury 2** (2026.02 출시) — 추론 특화, 후속 세대

---

## 3. 아키텍처

| 항목 | 내용 |
|------|------|
| 백본 | 표준 **Transformer** (논문: "diffusion은 학습/생성 알고리즘일 뿐, 아키텍처 제약 없음") |
| AR과의 차이 | 여러 토큰 **병렬 예측**, coarse-to-fine 반복 정제 |
| 디퓨전 종류 | discrete diffusion (창업자 계보상 masked diffusion 계열로 추정 — 논문 미명시) |
| 모델 크기 | **비공개** (Mini/Small 상대 명칭만) |
| 컨텍스트 | 기본 32,768 토큰, 확장 시 최대 128K |

**학습 손실**:

$$-\mathbb{E}_t\left[\gamma(t) \cdot \mathbb{E}_{z_t \sim q} \log p_\theta(x | z_t)\right]$$

$\gamma(t)$는 노이즈 레벨별 가중 함수.

### AR과의 핵심 차이

```
AR 모델:    토큰을 좌→우 순차 1개씩 예측
Mercury:    노이즈에서 시작 → 여러 토큰 병렬 예측
            → coarse-to-fine 반복 정제
            → 전역 관점에서 실수/할루시네이션 수정 가능
```

---

## 4. 학습 (공개 범위 — 상당 부분 비공개)

### 공개된 사항

| 항목 | 내용 |
|------|------|
| 학습 데이터 규모 | "조 단위(trillions) 토큰" |
| 데이터 구성 | 웹 크롤 + 자체 엄선 실제·합성 데이터 |
| 학습 인프라 | NVIDIA H100 대규모 클러스터 |
| 학습 절차 | 선행 연구를 데이터·연산 측면에서 스케일업. **핵심 변경 = AR 손실을 denoising diffusion 손실로 대체** |
| 사후 학습 | instruction tuning, RLHF, DPO 지원 |

### 비공개 (proprietary) — 솔직히 명시

- 정확한 토큰 수, 학습 FLOPs, 학습 기간
- 데이터 소스 구체 구성·비율
- 정확한 파라미터 수, 레이어 수, hidden dim
- 디퓨전 스케줄 $\gamma(t)$의 구체 형태, 타임스텝 $T$
- 추론 디노이징 스텝 수

---

## 5. 추론 / 샘플링 (핵심 — 헤드라인)

### Coarse-to-Fine 병렬 디노이징

```
노이즈 → [거친 출력] → 반복 정제 → [정교한 출력]
          ↑ 매 스텝 여러 토큰 병렬 수정, 전역 품질 개선
```

조건부 생성 지원: zero-shot, few-shot, chain-of-thought 모두.

### 자체 추론 엔진

고효율 디퓨전 샘플링 구현:
- **동적 배칭(dynamically batched) 샘플링**
- **페이징 구현**
- **병렬 추론 워크로드용 커스텀 커널**
- OpenAI 표준 호환 API

### 속도 (헤드라인 — H100 기준)

| 모델 | 속도 |
|------|------|
| **Mercury Coder Mini** | **1,109 tok/s** |
| **Mercury Coder Small** | **737 tok/s** |

- 속도 최적화 frontier 모델 대비 **평균 최대 10배** 빠름
- 50 tok/s 미만 frontier 모델 대비 최대 20배
- 과거 커스텀 칩에서만 가능하던 1,000+ tok/s를 범용 H100에서 달성

**측정 방법론**: 제3자 **Artificial Analysis**가 측정 (약 1K 입력 + 1K 출력 코딩 프롬프트, 클라우드 간 중앙값 throughput).

**디노이징 스텝 수는 비공개** — 논문은 forward pass 반복을 줄이는 것이 최적화 전략이라 언급하나 구체 수치 미공개. 선행 연구가 "반복 줄이기는 보였으나 wall-clock 효율 개선엔 실패"했음을 지적하며, Mercury가 커스텀 커널/엔진으로 해결했다고 주장.

---

## 6. 벤치마크

### 6.1 주요 코딩 벤치마크 (pass@1) + 속도

| 모델 | HumanEval | MBPP | EvalPlus | MultiPL-E | LCB | Speed |
|------|-----------|------|----------|-----------|-----|-------|
| **Mercury Coder Mini** | 88.0 | 77.1 | 78.6 | 74.1 | 17.0 | **1109** |
| **Mercury Coder Small** | 90.0 | 76.6 | 80.4 | 76.2 | 25.0 | **737** |
| GPT-4o Mini | 88.0 | 74.6 | 78.5 | 72.0 | 23.0 | 59 |
| Claude 3.5 Haiku | 86.0 | 78.0 | 75.1 | 72.3 | 31.0 | 61 |
| Gemini 2.0 Flash Lite | 90.0 | 75.0 | 77.3 | 79.5 | 18.0 | 201 |
| Qwen 2.5 Coder 7B | 88.0 | 80.0 | 79.3 | 75.3 | 9.0 | 195 |
| GPT-4o | 90.2 | 82.2 | 82.4 | 77.6 | 31.0 | 61 |
| Claude 3.5 Sonnet | 90.2 | 81.2 | 77.3 | 81.9 | 38.0 | 76 |

(LCB = LiveCodeBench)

**해석**: HumanEval 88.0/90.0은 GPT-4o Mini와 동급, Gemini 2.0 Flash Lite와 타이. **속도는 6-20배 빠름**. 단 LiveCodeBench(고난도 경쟁 프로그래밍)에서는 DeepSeek V3(36.0), Claude 3.5 Sonnet(38.0)에 뒤짐.

### 6.2 Fill-in-the-Middle (FIM) — 디퓨전의 강점

| 모델 | FIM Single-Line | FIM Random-Span | 평균 |
|------|-----------------|-----------------|------|
| **Mercury Coder Small** | 93.1 | 76.5 | **84.8** |
| **Mercury Coder Mini** | 92.9 | 71.5 | 82.2 |
| Codestral 2501 (코드 편집 특화) | 93.0 | 72.0 | 82.5 |
| GPT-4o Mini | 74.8 | 47.0 | 60.9 |
| Claude 3.5 Haiku | 63.6 | 27.4 | 45.5 |

**핵심 결과**: Mercury Coder Small(84.8)이 코드 편집 특화 Codestral까지 능가. AR 모델(GPT-4o Mini 60.9)과 큰 격차 → **디퓨전의 양방향(전역) 컨텍스트 활용이 FIM에 본질적으로 유리**.

### 6.3 Copilot Arena (실사용자 평가)

| 모델 | 지연시간 | Elo |
|------|---------|-----|
| **Mercury Coder Mini** | **0.25s (1위)** | 993 (공동 2위) |
| Codestral | 0.31s | 992 |
| GPT-4o | 0.76s | 980 |
| GPT-4o Mini | 0.84s | 939 |

**핵심**: Mercury Coder Mini가 품질 Elo 공동 2위 + **지연시간 1위** (평균 25ms, GPT-4o Mini보다 ~4배 빠름).

---

## 7. 배포 / 접근성

| 플랫폼 | 내용 |
|--------|------|
| Inception API | platform.inceptionlabs.ai, OpenAI 호환 |
| Mercury Chat | chat.inceptionlabs.ai |
| Amazon Bedrock | Marketplace 배포, 오토스케일링, Agents/Guardrails 연동 |
| Amazon SageMaker JumpStart | Studio/SDK 배포, 권장 인스턴스 ml.p5.48xlarge (H100) |
| Azure AI Foundry | 최초의 상용 dLLM으로 Azure 등재 |

온프레미스 배포 + 파인튜닝(SFT, RLHF) 지원.

---

## 8. 한계 / 비공개 사항

### 비공개 (proprietary)

파라미터 수, 학습 토큰 수, FLOPs, 데이터 소스 구성, 추론 디노이징 스텝 수, 노이징 스케줄, 아키텍처 세부 — 상용 제품이므로 상당 부분 미공개.

### 명시된 한계

1. **디퓨전 LLM 스케일링 특성**이 AR만큼 잘 이해되지 않음 (논문 인정)
2. **고난도 벤치마크 약세** — LiveCodeBench 등에서 풀 frontier 모델 대비 격차
3. **출력 품질** — 초기 사용자들은 AR 대비 다소 "거칠다"는 평 (반복 정제로 완화)
4. **속도 수치 주의** — 1109/737 tok/s는 특정 코딩 프롬프트 조건의 중앙값, 워크로드에 따라 변동

---

## 9. 핵심 요약

| 질문 | 답 |
|------|-----|
| **무엇** | 세계 최초 상용 규모 디퓨전 LLM |
| **누가** | SEDD(Ermon) + MDLM(Kuleshov) 저자들 |
| **아키텍처** | 표준 Transformer + discrete diffusion |
| **학습** | 조 단위 토큰, H100 클러스터, AR 손실 → denoising 손실 (세부 비공개) |
| **추론** | coarse-to-fine 병렬 디노이징, 자체 엔진, 1109 tok/s |
| **결과** | HumanEval 90, FIM에서 Codestral 능가, 속도 6-20배 |
| **비공개** | 파라미터·토큰 수·스텝 수·스케줄 |

> **한 줄**: "학계의 SEDD·MDLM이 상용 제품 Mercury가 되었고, 디퓨전 LM이 AR보다 10배 빠를 수 있음을 증명했다."

---

## 10. 관련 블로그 포스트

- [Diffusion LM 서베이](diffusion-language-models-survey.md)
- [LLaDA 심화 리뷰](llada-review.md)
- [SEDD 심화 리뷰](sedd-deep-review.md) — Mercury 창업자 Ermon의 논문
- [MDLM 심화 리뷰](mdlm-deep-review.md) — Mercury 창업자 Kuleshov의 논문

---

## 참고 자료

- [Mercury (arXiv:2506.17298)](https://arxiv.org/abs/2506.17298)
- [Inception Labs 공식 블로그](https://www.inceptionlabs.ai/blog/introducing-mercury)
- [AWS — Mercury on Bedrock & SageMaker](https://aws.amazon.com/blogs/machine-learning/mercury-foundation-models-from-inception-labs-are-now-available-in-amazon-bedrock-marketplace-and-amazon-sagemaker-jumpstart/)
