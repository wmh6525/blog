---
title: "[서베이] 연속 의미 공간에서 예측하는 언어 모델 — Discrete Token을 넘어서 (LCM, Coconut, CALM, Diffusion LM, JEPA)"
date: 2026-05-06
tags: ["서베이", "LLM", "연속벡터", "LCM", "Coconut", "CALM", "Diffusion-LM", "JEPA"]
categories: ["ML/AI"]
summary: "다음 토큰 예측(NTP)을 넘어, 연속 벡터 공간에서 직접 예측하는 언어 모델 패러다임을 정리한다. Meta의 LCM(SONAR 임베딩), Coconut(연속 추론), CALM(연속 자회귀), Diffusion-LM, LLaDA, Mercury, 그리고 LeCun의 LLM-JEPA까지 — 30+ 논문 종합."
math: true
toc: true
draft: false
---

## 왜 연속 공간인가?

표준 LLM은 **이산 토큰**을 예측한다:

$$P(x_t | x_{<t}) = \text{softmax}(W \cdot h_t)$$

→ 어휘 V (50k~100k) 위의 확률 분포 → 한 토큰 샘플링

**하지만 인간의 사고는 단어 하나씩 떠오르는가?**

```
"추론" 또는 "사고"라는 행위는 단어 시퀀스의 형태로 일어나지 않는다.
이미지 생성도 픽셀 하나씩 분류 문제로 풀지 않고, 연속 공간에서 노이즈를 걷어낸다.
왜 언어만 이산 토큰에 갇혀 있어야 하는가?
```

이 질문에서 출발한 연구 흐름이 **연속 의미 공간 예측 LM**이다. 30+ 논문을 5가지 방향으로 정리한다:

1. **문장 수준 연속 예측** — LCM, SONAR
2. **토큰 묶음 연속 자회귀** — CALM
3. **연속 추론 (Latent CoT)** — Coconut, CoCoMix, Soft Thinking
4. **연속 확산 LM** — Diffusion-LM, LLaDA, Mercury
5. **JEPA 스타일 (LeCun)** — LLM-JEPA, VL-JEPA

---

## 1. 문장 수준 연속 예측

### 1.1 LCM (Large Concept Model) — Meta, 2024

- **arXiv**: 2412.08821
- **저자**: Loïc Barrault, Paul-Ambroise Duquenne, Holger Schwenk 외 (Meta FAIR)
- **코드**: [github.com/facebookresearch/large_concept_model](https://github.com/facebookresearch/large_concept_model)

**한 줄**: 토큰이 아니라 **문장(=concept) 단위로** 자회귀 예측.

#### 작동 방식

```
[1] 입력 문장 → SONAR 인코더 → 1024-dim 임베딩
[2] LCM이 다음 문장 임베딩을 자회귀로 예측
[3] SONAR 디코더 → 200개 언어 중 하나의 텍스트로 디코드
```

#### 4가지 변형

| 변형 | 손실 | 결과 |
|------|------|------|
| **Base-LCM** | MSE 회귀 | L2는 최고지만 다양성 최악 (averaging problem) |
| **One-Tower Diffusion** | Score matching | 단일 transformer로 컨텍스트+denoising |
| **Two-Tower Diffusion** | Score matching | 컨텍스트(5층) + Denoiser(14층) 분리, 7B까지 스케일 |
| **Quant-LCM** | Cross-entropy | SONAR 양자화 |

#### 왜 MSE가 실패하는가?

같은 컨텍스트에 대한 **유효한 다음 문장이 여러 개**인데, MSE는 평균값을 학습 → 의미 없는 평균 임베딩 생성.

→ Diffusion이 multimodal 분포를 자연스럽게 처리.

#### 강점

- **언어 무관 추론**: SONAR 공간이 200개 언어 공유 → zero-shot 다국어 요약
- **계층적 추론**: 토큰이 아닌 문장 단위 → 긴 문서 처리에 자연스러움

#### 약점

- 문장 단위 → 세밀한 표현 손실
- SONAR 디코더 품질에 종속

---

## 2. 토큰 묶음 연속 자회귀

### 2.1 CALM (Continuous Autoregressive Language Models) — Tencent, 2025

- **arXiv**: 2510.27688 (2025.10)
- **저자**: Chenze Shao, Darren Li, Fandong Meng, Jie Zhou (Tencent / WeChat AI)
- **코드**: [github.com/shaochenze/calm](https://github.com/shaochenze/calm)

**한 줄**: **K개 토큰 → 하나의 연속 벡터**로 자회귀 → 생성 스텝을 K배 단축.

#### 아키텍처

```
[1] 고품질 오토인코더 (75M)
    K=4 토큰 → 128-dim 연속 벡터 (>99.9% 복원)

[2] CALM (Transformer)
    연속 벡터를 자회귀로 예측

[3] 디코더
    벡터 → K개 토큰 복원 → 다시 임베딩 → 다음 예측 입력
```

#### Energy Score 손실 (핵심)

기존 diffusion/flow matching 헤드보다 **Energy Score**가 우수:

$$\mathcal{L}_{\text{energy}} = -\mathbb{E}\left[\| \hat{y}_1 - y \|\right] + \frac{1}{2}\mathbb{E}\left[\| \hat{y}_1 - \hat{y}_2 \|\right]$$

- 첫 항: 예측이 정답에 가까이
- 둘째 항: 예측 다양성 (mode collapse 방지)
- **Strictly proper scoring rule** — 다양성과 정확성 동시 보장

#### BrierLM — 새 평가 지표

연속 출력은 cross-entropy 계산 불가 → **Brier 점수의 기하 평균**으로 대체:

$$\text{BrierLM} = \left(\prod_{n=1}^{4} \text{Brier-n}\right)^{1/4}$$

CE와 거의 선형 상관 → CE 대체 가능.

#### 결과

| 모델 | BrierLM | 토큰/스텝 |
|------|---------|---------|
| Discrete AR baseline | 6.05 | 1 |
| **CALM-M (371M)** | 5.72 | **4** |
| **CALM-L (735M)** | 6.58 | **4** |
| **CALM-XL (1.82B)** | **8.53** | **4** |

→ **4배 빠른 생성**, 동등 이상 품질.

---

## 3. 연속 추론 (Latent CoT)

### 3.1 Coconut (Chain of Continuous Thought) — Meta, 2024

- **arXiv**: 2412.06769 (COLM/ICLR 2025)
- **저자**: Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Jason Weston, Yuandong Tian
- **코드**: [github.com/facebookresearch/coconut](https://github.com/facebookresearch/coconut)

**한 줄**: LLM의 **마지막 hidden state를 다음 입력으로 직접 사용** — 단어로 디코드 없이 연속 공간에서 추론.

#### 작동 방식

```
일반 CoT:  질문 → "단계 1: ..." → "단계 2: ..." → ... → 답
            ↓                            (각 단계가 토큰화됨)

Coconut:  질문 → <bot> [hidden_1] [hidden_2] ... <eot> → 답
                  ↑ 연속 벡터, 디코드 안 함
```

#### 학습: 다단계 커리큘럼

```
Stage 0: 일반 CoT 학습
Stage 1: 첫 추론 단계를 c개의 latent로 대체
Stage 2: 처음 두 추론 단계를 2c개의 latent로 대체
...
Stage k: 처음 k개 단계를 k·c개의 latent로 대체
```

**커리큘럼 없으면 학습 실패** — 너무 추상적.

#### 핵심 통찰: BFS-like Exploration

이산 CoT는 **하나의 단계만** 선택. 연속 thought는 **여러 가능한 다음 단계의 superposition** 표현 가능 → 동시에 여러 경로 탐색.

#### 결과

| 과제 | No CoT | 일반 CoT | **Coconut** |
|------|-------|--------|----------|
| ProntoQA | - | 99.8% | **99.9%** |
| ProsQA | 76.7% | 77.5% | **97.0%** |
| GSM8K | 16.5% | 42.9% | 34.1% |

ProsQA(계획 필요한 추론)에서 압도적 우세. 단순 산술(GSM8K)은 일반 CoT보다 약간 떨어짐.

### 3.2 CoCoMix — Meta FAIR, 2025 (ICLR 2026)

- **arXiv**: 2502.08524
- **저자**: Jihoon Tack, Jack Lanchantin, Shibo Hao, Yuandong Tian, Jason Weston, Xian Li 외
- **한 줄**: 사전학습된 SAE로 **개념을 추출**, 토큰 hidden과 **인터리빙하여 사전학습**.

```
표준 NTP:    [token] [token] [token] [token] ...
                ↓ 예측

CoCoMix:    [token] [concept] [token] [concept] ... 
                ↓ 토큰 예측 + 개념 예측 동시
```

**효과**: 같은 성능에 **21.5% 적은 토큰**, 지식 증류와 pause token보다 우수, 해석 가능성과 조정 가능성 추가.

### 3.3 Soft Thinking — UCSC + Microsoft (NeurIPS 2025)

- **arXiv**: 2505.15778
- **한 줄**: argmax/sampling을 **확률 가중 토큰 임베딩 mixture**로 대체.

```
일반: argmax(logits) → "the" → embedding("the") → 다음 입력

Soft Thinking:
  softmax(logits) = [0.3, 0.5, 0.1, ...]
  next_input = 0.3 · emb("the") + 0.5 · emb("a") + 0.1 · emb("an") + ...
```

**학습 불필요!** 추론 시 적용. Pass@1 +2.48%, 토큰 −22.4%.

### 3.4 Quiet-STaR — Stanford (2024)

- **arXiv**: 2403.09629
- 매 토큰마다 rationale 생성 → REINFORCE로 학습
- 학습 가능한 `<startofthought>` / `<endofthought>` 토큰
- GSM8K 5.9% → 10.9%, CommonsenseQA 36.3% → 47.2%

### 3.5 Pause Tokens — Google + CMU (ICLR 2024)

- **arXiv**: 2310.02226
- 학습 가능한 `<pause>` 토큰을 추가 → 출력 추출을 지연
- 같은 hidden 공간에서 추가 계산 시간 확보
- 1B 모델: SQuAD EM +18%

---

## 4. 연속 확산 언어 모델

### 4.1 Diffusion-LM — Stanford (NeurIPS 2022)

- **arXiv**: 2205.14217
- **저자**: Xiang Lisa Li, John Thickstun, Ishaan Gulrajani, Percy Liang, Tatsunori Hashimoto
- **한 줄**: 단어 임베딩에 가우시안 노이즈 → **점진적으로 denoising**하여 텍스트 생성.

```
표준 LM:  좌→우 자회귀 (한 번에 한 토큰)
Diffusion-LM: 노이즈 → 노이즈 → ... → 임베딩 → 토큰
              (병렬, 양방향, 제어 가능)
```

**강점**: gradient-based 제어 (특정 어휘/스타일/구조 강제).

### 4.2 LLaDA (Large Language Diffusion) — 2025

- **arXiv**: 2502.09992 (ICLR 2025)
- **저자**: Shen Nie, Fengqi Zhu 외 (Renmin Univ + Ant Group)
- **8B 규모 마스킹 확산 LLM** — LLaMA3 8B와 동등한 성능
- 역방향 시 시 완성에서 **GPT-4o 능가** (reversal curse 해결)
- LLaDA 2.0-flash: **100B까지 스케일**

### 4.3 Mercury — Inception Labs (2025)

- **arXiv**: 2506.17298
- **저자**: Inception Labs (Stefano Ermon, Aditya Grover, Volodymyr Kuleshov 공동 창업)
- **최초의 상용급 diffusion LLM**

#### 작동

```
표준 AR:    [t1] [t2] [t3] [t4] [t5] ... (순차)
Mercury:    [MASK] [MASK] [MASK] [MASK] [MASK]
           → [t1?] [t2?] [t3?] [t4?] [t5?]   (병렬 denoising)
           → ...
           → [t1] [t2] [t3] [t4] [t5]
```

#### 성능

- **Mercury Coder Mini: 1109 토큰/초**
- **Mercury Coder Small: 737 토큰/초** (H100)
- 속도 최적화 frontier 모델보다 **최대 10배 빠름**

Azure AI Foundry, AWS Bedrock에서 사용 가능.

### 4.4 SEDD (Score Entropy Discrete Diffusion) — Stanford (ICML 2024 Best Paper)

- **arXiv**: 2310.16834
- 이산이지만 비교 차원에서 중요
- 새로운 **score entropy** 손실 → score matching을 이산 공간으로 확장
- GPT-2 perplexity 25-75% 개선, 32배 적은 평가로 가능

---

## 5. JEPA 스타일 (LeCun의 비전)

### 5.1 LLM-JEPA — 2025

- **arXiv**: 2509.14252
- **저자**: Hai Huang, **Yann LeCun**, Randall Balestriero
- **한 줄**: JEPA(Joint Embedding Predictive Architecture)를 **언어**에 적용.

```
입력 공간 예측 (표준 LM):
  X → predict X' (실제 토큰)

임베딩 공간 예측 (JEPA):
  X → encode → z
  X' → encode → z'
  predict z' from z (임베딩 공간에서)
```

#### 결과

- Llama3, OpenELM, Gemma2, Olmo에서 표준 LM 학습 능가
- 과적합에 robust
- NL-RX, GSM8K, Spider, RottenTomatoes에서 검증

### 5.2 VL-JEPA — 2025.12

- **arXiv**: 2512.10942
- **저자**: Delong Chen, Yann LeCun, Pascale Fung 외
- Vision-Language JEPA — 타겟 텍스트의 **연속 임베딩 예측**
- **50% 적은 학습 파라미터**, 2.85배 적은 디코딩 연산
- 표준 VLM과 동등 성능 (1.6B 파라미터)

---

## 6. 종합 비교표

### 무엇을 예측하는가?

| 모델 | 예측 단위 | 예측 공간 | 출력 |
|------|--------|---------|------|
| 표준 LM | 토큰 | 이산 (V차원) | 토큰 |
| **LCM** | 문장 | SONAR (1024-d) | 문장 임베딩 → 디코드 |
| **CALM** | K개 토큰 묶음 | 128-d 잠재 | 벡터 → 디코드 |
| **Coconut** | hidden state | LLM hidden (4096-d) | 그대로 다음 입력 |
| **CoCoMix** | 토큰 + 개념 | 토큰 + SAE 개념 | 토큰 |
| **Soft Thinking** | 임베딩 mixture | 토큰 임베딩 가중평균 | 토큰 |
| **Diffusion-LM** | 단어 임베딩 | 임베딩 공간 | 임베딩 → rounding → 토큰 |
| **LLaDA** | 모든 토큰 동시 | 마스킹 분포 | 토큰 |
| **Mercury** | 모든 토큰 동시 | 마스킹 분포 | 토큰 |
| **LLM-JEPA** | 임베딩 (학습용) | 임베딩 공간 | 표준 토큰 (서빙) |

### 학습 손실

| 모델 | 손실 |
|------|------|
| 표준 LM | Cross-entropy |
| LCM (Base) | MSE (averaging problem) |
| LCM (Diffusion) | Score matching |
| CALM | **Energy Score** (proper scoring rule) |
| Coconut | CE on final token (latent은 gradient-free) |
| CoCoMix | CE 토큰 + 개념 예측 |
| Diffusion-LM | Diffusion ELBO |
| Mercury / LLaDA | Masked diffusion |
| LLM-JEPA | Embedding regression |

---

## 7. 왜 연속이 매력적인가? — 8가지 동기

1. **부드러운 그래디언트**: softmax bottleneck 회피
2. **높은 정보 대역폭**: 연속 벡터가 multiple alternatives 동시 표현 (Coconut의 BFS 논증)
3. **이산화 손실 없음**: 이미지/오디오 생성과 정렬
4. **계산 효율**: 적은 스텝 (CALM 4배, Mercury 10배)
5. **언어 무관 추론**: 문장/개념 수준은 언어 독립 (LCM 200개 언어)
6. **Reversal curse 해결**: 좌→우 편향 회피 (LLaDA)
7. **Softmax rank bottleneck 회피**: MoS 논증
8. **"인간 언어로 사고하지 않아도 됨"**: 자연어 대역폭 제약 탈피

---

## 8. 트레이드오프와 한계

| 문제 | 설명 |
|------|------|
| **명시적 likelihood 손실** | CE 계산 불가 → CALM이 BrierLM 발명 |
| **Averaging problem** | 다중 valid 답 → 평균하면 무의미 (Base-LCM 실패) |
| **커리큘럼/distillation 필요** | Coconut, CODI 모두 단계별 학습 필요 |
| **해석성 어려움** | 연속 latent는 디버깅 어려움 (CoCoMix가 SAE로 해결 시도) |
| **"진짜 추론하나?" 의심** | 암묵적 추론이 step-by-step인지 불분명 (Lin et al. 2024 비판) |
| **인프라 부재** | 기존 토큰 기반 인프라(검색, 캐싱)와 호환 어려움 |

---

## 9. 5가지 패러다임 요약

```
[표준 NTP]
  Discrete tokens, autoregressive softmax
  → GPT, LLaMA, Claude...

[연속 자회귀]
  ├── LCM: 문장 단위 SONAR
  └── CALM: K-token 묶음 (4배 가속)

[연속 추론]
  ├── Coconut: hidden을 그대로 사용
  ├── CoCoMix: 토큰+개념 인터리빙 (사전학습)
  ├── Soft Thinking: 임베딩 mixture (학습 불필요)
  └── CCoT: 압축된 contemplation token

[연속 확산]
  ├── Diffusion-LM: 임베딩 denoising
  ├── LLaDA: 100B 마스킹 확산 LLM
  ├── Mercury: 상용 10배 가속
  └── SEDD: 이산 확산 SOTA

[JEPA]
  ├── LLM-JEPA: 임베딩 예측 (LeCun)
  └── VL-JEPA: 비전+언어
```

---

## 10. 미래 전망

### 5년 후 예측

연속 LM이 이산 LM을 완전히 대체하지는 않을 것이지만:

1. **추론 작업**: Coconut/CoCoMix 계열이 표준화 (인간 언어로 사고 안 해도 됨)
2. **실시간 응답**: Mercury 같은 확산 LM이 latency-critical 영역 점령
3. **다국어/문서 요약**: LCM 계열이 sentence-level 추론 표준
4. **개념 수준 RAG**: 토큰이 아닌 의미 단위 검색
5. **하이브리드**: CALM처럼 K-token 묶음이 효율과 표현력의 sweet spot

### 핵심 미해결 문제

- **연속 공간의 평가 지표**: BrierLM 같은 새 메트릭 표준화 필요
- **연속 ↔ 이산 인터페이스**: 결국 인간은 텍스트를 읽음
- **해석 가능성**: SAE 등으로 연속 표현 해석 도구 발전 필요
- **이론적 이해**: 왜 연속이 어떤 작업에서 더 좋은지에 대한 이론 부재

---

## 11. 핵심 한 줄 요약

> **"인간의 사고는 단어 시퀀스가 아니다. 이미지 생성도 픽셀 분류가 아니다. 그렇다면 언어 모델도 토큰 분류만이 정답일 필요가 없다."**

이 직관에서 출발한 LCM, Coconut, CALM, Diffusion-LM, JEPA가 **다음 세대 언어 모델 패러다임**을 형성하고 있다.

---

## 12. 관련 블로그 포스트

- [DPO 리뷰](dpo-review.md) — RLHF의 단순화 (이산 정렬)
- [Search-R1 리뷰](search-r1-review.md) — RL 기반 검색 LLM
- [강화학습 입문](reinforcement-learning-beginner-guide.md) — RL 기초
- [Mamba 패밀리 서베이](mamba-family-complete-survey.md) — 다른 아키텍처 진화

---

## 참고 자료

### 핵심 논문
- [LCM (arXiv:2412.08821)](https://arxiv.org/abs/2412.08821) | [코드](https://github.com/facebookresearch/large_concept_model)
- [Coconut (arXiv:2412.06769)](https://arxiv.org/abs/2412.06769) | [코드](https://github.com/facebookresearch/coconut)
- [CALM (arXiv:2510.27688)](https://arxiv.org/abs/2510.27688) | [코드](https://github.com/shaochenze/calm)
- [CoCoMix (arXiv:2502.08524)](https://arxiv.org/abs/2502.08524)
- [Soft Thinking (arXiv:2505.15778)](https://arxiv.org/abs/2505.15778) | [코드](https://github.com/eric-ai-lab/Soft-Thinking)
- [Diffusion-LM (arXiv:2205.14217)](https://arxiv.org/abs/2205.14217) | [코드](https://github.com/XiangLi1999/Diffusion-LM)
- [LLaDA (arXiv:2502.09992)](https://arxiv.org/abs/2502.09992) | [코드](https://github.com/ML-GSAI/LLaDA)
- [Mercury (arXiv:2506.17298)](https://arxiv.org/abs/2506.17298)
- [SEDD (arXiv:2310.16834)](https://arxiv.org/abs/2310.16834)
- [LLM-JEPA (arXiv:2509.14252)](https://arxiv.org/abs/2509.14252)

### 보조 논문
- [Quiet-STaR (arXiv:2403.09629)](https://arxiv.org/abs/2403.09629)
- [Pause Tokens (arXiv:2310.02226)](https://arxiv.org/abs/2310.02226)
- [CCoT (arXiv:2412.13171)](https://arxiv.org/abs/2412.13171)
- [CODI (arXiv:2502.21074)](https://arxiv.org/abs/2502.21074)
- [SoftCoT (arXiv:2502.12134)](https://arxiv.org/abs/2502.12134)
- [SONAR (arXiv:2308.11466)](https://arxiv.org/abs/2308.11466)
- [VL-JEPA (arXiv:2512.10942)](https://arxiv.org/abs/2512.10942)
- [MoS (arXiv:1711.03953)](https://arxiv.org/abs/1711.03953)
- [MAR — 이미지 분야 (arXiv:2406.11838)](https://arxiv.org/abs/2406.11838)

### 서베이
- [Latent Reasoning Survey (arXiv:2507.06203)](https://arxiv.org/abs/2507.06203)
- [Latent CoT Survey (arXiv:2505.16782)](https://arxiv.org/abs/2505.16782)
- [NTP Alternatives Survey (arXiv:2509.24435)](https://arxiv.org/abs/2509.24435)
