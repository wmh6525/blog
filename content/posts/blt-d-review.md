---
title: "[논문 리뷰] BLT-D — Byte Latent Transformer에 블록 디퓨전을 결합해 추론을 92% 가속"
date: 2026-05-15
tags: ["논문리뷰", "Diffusion", "LLM", "BLT", "바이트", "토크나이저프리"]
categories: ["ML/AI"]
summary: "Fast Byte Latent Transformer 논문 상세 리뷰. BLT-D(BLT Diffusion), BLT-S(Self-speculation), BLT-DV(Diffusion+Verification) 세 기법으로 바이트 단위 LLM의 추론 메모리 대역폭을 최대 92% 절감. 엔트로피 패칭부터 블록 디퓨전 학습·추론까지 순서도 도식 포함 완전 해부."
math: true
toc: true
draft: false
---

## 논문 정보

- **제목**: Fast Byte Latent Transformer
- **저자**: Julie Kallini, Artidoro Pagnoni, Tomasz Limisiewicz, Gargi Ghosh, Luke Zettlemoyer, Christopher Potts, Xiaochuang Han, Srinivasan Iyer
- **소속**: Meta (FAIR) + Stanford + University of Washington
- **arXiv**: 2605.08044 (2026.05)
- **약어 정리**: 논문은 3가지 기법을 제안 — **BLT-D** (BLT Diffusion, 핵심), **BLT-S** (Self-speculation), **BLT-DV** (Diffusion + Verification)

> 참고: "BLT-D"는 단독 논문이 아니라 "Fast Byte Latent Transformer" 논문 안의 기법 이름이다. 코드는 아직 미공개 (2026.05 기준).

---

## 1. 한 줄 요약

> **토크나이저 없이 바이트를 직접 다루는 BLT는 학습은 빠르지만 추론이 느렸다. BLT-D는 로컬 디코더에 블록 디퓨전을 붙여 여러 바이트를 병렬 생성 — 추론 메모리 대역폭을 최대 92% 절감.**

---

## 2. 배경 — BLT(Byte Latent Transformer)란?

### 2.1 원조 BLT (Meta, 2024.12)

- **논문**: "Byte Latent Transformer: Patches Scale Better Than Tokens" (arXiv 2412.09871)
- **핵심**: **토크나이저 없이** 원시 바이트를 직접 처리하는 LLM
- 바이트를 **엔트로피 기반 패치(patch)**로 동적 그룹화 → 8B 파라미터 / 4T 바이트로 Llama 3 토크나이저 모델과 동등

### 2.2 BLT의 4단계 구조

```
원시 바이트 입력
   │
   ▼
┌──────────────────────────────┐
│ ① 엔트로피 패칭               │  작은 바이트 모델이 다음 바이트
│  (Entropy-based Patching)     │  불확실성(엔트로피) 측정
│                               │  → 예측 쉬운 구간 = 긴 패치
│                               │  → 예측 어려운 구간 = 짧은 패치
│                               │  (평균 ~4바이트, 최대 ~8바이트)
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ ② Local Encoder (ℰ)          │  경량 Transformer
│                               │  N개 바이트 → M ≈ N/4 잠재 토큰으로 압축
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ ③ Global Transformer (𝒢)     │  full-attention Transformer
│                               │  ~4배 적은 잠재 토큰에서 동작
│                               │  ← 대부분의 연산이 여기 집중
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ ④ Local Decoder (𝒟)          │  경량 Transformer
│                               │  잠재 토큰 → 출력 바이트로 디코드
│                               │  (자회귀, causal mask)
└──────────────┬───────────────┘
               ▼
          출력 바이트
```

### 2.3 BLT가 푼 문제 vs 못 푼 문제

| | 학습 시간 | 추론 시간 |
|--|---------|---------|
| **BLT가 해결** | ✅ 동적 패칭으로 글로벌 어텐션이 ~4배 적은 단위에서 동작 | ❌ |
| **BLT가 못 푼 것** | | **바이트를 한 번에 1개씩** 자회귀 디코드 → 한 토큰 = 여러 바이트이므로 토큰 모델보다 디코더 forward pass가 훨씬 많이 필요 |

→ **메모리 대역폭 병목** (디코더 가중치 + KV 캐시를 반복 스트리밍). FLOPs 병목이 아님.

> **Fast BLT의 질문**: BLT의 아키텍처를 유지하면서, 바이트를 **병렬 생성**할 수 없을까?

---

## 3. BLT-D — 핵심 기법

> **로컬 디코더에 보조 블록 디퓨전 목적함수를 추가하여, 미래 바이트 여러 개를 병렬로 unmask.**

### 3.1 학습 데이터 전처리

```
패치 경계 기준으로 블록 구성
   │
   ▼
┌────────────────────────────────────────┐
│ 블록(Block) = B개 연속 바이트           │
│ 각 패치 경계에서 시작                    │
│                                        │
│ ★ 핵심 트릭: 블록이 패치 경계를         │
│   "넘어서" 확장됨                       │
│   → 디코더가 평균 패치 길이를            │
│     넘는 바이트까지 예측하도록 학습      │
│   → 이것이 병렬 lookahead를 가능케 함    │
└────────────────────────────────────────┘
```

### 3.2 디퓨전 노이즈 주입 (absorbing-state 마스킹)

각 블록에 대해:
1. 타임스텝 $t \sim U(0, 1)$ 샘플
2. 블록 내 각 바이트를 독립적으로 확률 $t$로 `[MASK]` 치환
3. $t=0$ → 완전 clean, $t=1$ → 완전 마스킹

### 3.3 비대칭 어텐션 마스킹

| 영역 | 어텐션 방식 |
|------|-----------|
| Clean 시퀀스 | causal (인과적) |
| **손상된 블록 내부** | **양방향 (bidirectional)** ← 병렬성의 원천 |
| 손상된 블록 → 이전 clean 바이트 | causal |
| Cross-attention (clean) | 이전 잠재 토큰에 어텐드 |
| Cross-attention (손상) | **마지막 잠재 토큰**에 어텐드 |

### 3.4 학습 목적함수

전체 손실 = 표준 자회귀 손실 + 마스킹 디퓨전 손실

**Clean (자회귀) 손실**:

$$\mathcal{L}_{\text{clean}}(\theta) = -\sum_i \log p_\theta(x_i | x_{\lt i})$$

**마스킹 디퓨전 손실** (마스킹된 위치만 복원, $1/t$ 재가중):

$$\mathcal{L}_{\text{mask}}(\theta) = -\frac{1}{t} \sum_i \sum_k \mathbb{1}[b_{i,k}^t = [\text{MASK}]] \cdot \log p_\theta(x_{s_i + k} | b_i^t, x_{\lt s_i})$$

- $b_i^t$: 타임스텝 $t$에서 손상된 블록 $i$
- $s_i$: 블록 $i$의 시작 인덱스, $k$: 블록 내 오프셋
- $1/t$ 가중치 = absorbing diffusion(MDLM/D3PM 스타일) ELBO 스케일링 — 많이 마스킹된(어려운) 타임스텝을 up-weight

**결합 손실**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{clean}} + \mathcal{L}_{\text{mask}}$$

**핵심**: 같은 모델이 두 손실로 학습되므로 BLT-D는 **자회귀 모델로도 작동 가능** — 이 성질을 BLT-DV가 활용한다.

---

## 4. BLT-D 추론 순서도

블록 하나(B 바이트)를 생성하는 절차:

```
                  ┌─────────────────────────────┐
                  │  ① 프리픽스 인코딩            │
                  │  엔트로피 패처 → ℰ → 𝒢       │
                  │  잠재 토큰 생성 (캐시)        │
                  └──────────────┬──────────────┘
                                 ▼
                  ┌─────────────────────────────┐
                  │  ② 블록 초기화                │
                  │  B개 [MASK] 위치 생성         │
                  │  - 마지막 잠재 토큰에 어텐드   │
                  │  - 자기들끼리 양방향           │
                  └──────────────┬──────────────┘
                                 ▼
        ┌───────────▶┌─────────────────────────────┐
        │            │  ③ 병렬 unmasking 반복        │
        │            │  로컬 디코더 forward 1회      │
        │            │  → 모든 마스크 위치 동시 분포  │
        │            │                              │
        │            │  커밋할 위치 선택:            │
        │            │   (A) Confidence: p > α       │
        │            │   (B) Entropy-bounded: γ까지  │
        │            └──────────────┬──────────────┘
        │                           ▼
        │                    모든 위치 채워짐?
        │                     │            │
        │                  아니오          예
        └─────────────────────┘            ▼
                              ┌─────────────────────────────┐
                              │  ④ (선택) 검증 — BLT-DV       │
                              │  블록을 causal mask로 재인코딩 │
                              │  자회귀 예측과 대조 검증       │
                              └──────────────┬──────────────┘
                                             ▼
                              ┌─────────────────────────────┐
                              │  ⑤ 블록 확정 + 전진           │
                              │  커밋된 블록 추가 → 재인코딩   │
                              │  → 다음 블록으로              │
                              └─────────────────────────────┘
```

**핵심**: ③에서 매 forward pass마다 **여러 바이트가 동시 커밋** → B 바이트를 채우는 데 B번보다 훨씬 적은 forward pass.

### 4.1 두 가지 unmasking 전략 (추론 시 튜닝, 재학습 불필요)

| 전략 | 방식 |
|------|------|
| **Confidence 기반** | 확률 > 임계값 $\alpha$ (예: 0.7)인 모든 위치 unmask. 없으면 최고 확신도 1개만 (진행 보장) |
| **Entropy-bounded 샘플링** | 마스크 위치를 예측 엔트로피로 정렬, 누적 엔트로피가 $\gamma$ 초과할 때까지 커밋. $\gamma$↑ → 스텝당 더 많은 바이트 (빠름, 다양성↑) |

블록 크기 $B \in \lbrace 4, 8, 16 \rbrace$ → 모델명 **BLT-D-4 / BLT-D-8 / BLT-D-16**.

---

## 5. BLT-S — Self-Speculation (추가 학습 불필요)

BLT의 **기존 경량 로컬 디코더**를 speculative **드래프터**로 재활용:

```
┌─────────────────────────────────────────┐
│ ① 드래프트                               │
│   로컬 디코더가 k 바이트 자회귀 초안 작성  │
│   (k = 8 또는 16, 엔트로피 무시)          │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│ ② 검증                                   │
│   전체 모델(ℰ+𝒢+𝒟)이 재인코딩            │
│   → 검증된 다음 바이트 예측 (1 forward)    │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│ ③ 수용                                   │
│   드래프트 vs 검증을 바이트별 비교         │
│   첫 불일치 전까지 수용, 롤백, 재개        │
└─────────────────────────────────────────┘
```

**장점**: **추가 학습 0**. greedy 디코딩 시 **표준 BLT와 완전히 동일한 출력** → **품질 손실 없는 순수 가속**.

---

## 6. BLT-DV — Diffusion + Verification

BLT-D가 $\mathcal{L}_{\text{clean}}$도 학습했으므로 같은 가중치를 causal mask로 자회귀 실행 가능. BLT-DV는:

1. BLT-D 디퓨전 디코더가 병렬 unmasking으로 블록 제안
2. 전체 모델이 블록을 **causal 어텐션**으로 재인코딩
3. 자회귀 예측과 검증 → 검증된 prefix 수용, 첫 불일치에서 거부

**결과**: BLT-D의 품질 손실을 **회복**하면서 가속 대부분 유지 (BLT-D와 BLT-S의 중간).

---

## 7. 학습 설정

| 항목 | 값 |
|------|-----|
| 모델 | BLT(베이스라인), BLT-D-4/8/16, BLT-S, BLT-DV — **1B / 3B** |
| 학습 코퍼스 | **BLT-1T** — 1조 토큰 (Datacomp-LM 부분집합 포함, 원조 BLT와 동일 계보) |
| 학습 (1B) | 240K 스텝, 배치 $2^{19}$ 토큰/스텝 |
| 학습 (3B) | 480K 스텝, 배치 $2^{20}$ 토큰/스텝 |
| 옵티마이저 | AdamW ($\beta_1$=0.9, $\beta_2$=0.95) |
| 학습률 | peak 4e-4, cosine 스케줄 |
| Weight decay | 0.1 |
| 그래디언트 클리핑 | 1.0 |

**평가 과제**:
- 번역: FLORES-101 (Fr→En, De→En), SentencePiece BLEU (4-shot)
- 코드: HumanEval (0-shot), MBPP (3-shot)
- Likelihood: ARC-Easy/Challenge, PIQA, HellaSwag, MMLU

**효율 지표**: 디코더 NFE(network function evaluations), 인코더/글로벌 NFE, 추정 메모리 대역폭:

$$\text{Bandwidth} \approx \frac{b \cdot [N_{\text{dec}} \cdot P_{\text{dec}} + N_{\text{enc}} \cdot (P_{\text{enc}} + P_{\text{glob}})]}{10^9}$$

$b=2$ (16비트 가중치), $N$=호출 수, $P$=모듈 파라미터 수.

---

## 8. 실험 결과

### 8.1 메모리 대역폭 절감 (3B, confidence $\alpha \approx 0.7$, BLT 베이스라인 대비)

| 변형 | 대역폭 절감 | 품질 |
|------|----------|------|
| BLT-D-4 | ~50%+ | 거의 베이스라인 수준 |
| BLT-D-8 | ~72-73% | 번역에서 강함 |
| **BLT-D-16** | **87-92%** | 코드 과제에서 눈에 띄는 하락 |
| BLT-S (k=16) | 최대 ~77% | **품질 손실 없음** (greedy 시 동일 출력) |
| BLT-DV | 최대 ~81% | BLT-D 품질 회복, BLT-D보다 느림 |

**헤드라인**: 3가지 기법 모두 BLT 대비 **추정 메모리 대역폭 50%+ 절감**, 일부 구성에서 **최대 92%**.

### 8.2 Likelihood 벤치마크

BLT-D 변형은 BLT 대비 **소폭 하락 (~1-4점)** — $\mathcal{L}_{\text{clean}}$도 학습하므로 자회귀 능력 보존.

### 8.3 품질/속도 트레이드오프

```
블록 크기 B ↑  →  병렬성/속도 ↑  →  품질 손실 ↑

번역 과제:  큰 블록도 잘 견딤
코드 과제:  BLT-D-16에서 명확히 열화 (구조적/정확 출력에 민감)
```

**다양성 노브**: entropy-bounded 샘플링에서 디코더 호출을 늘리면 type-token ratio 상승 → **재학습 없이 다양성-효율 프론티어 튜닝 가능**.

---

## 9. 다른 디퓨전 LM과의 차별점

| | LLaDA / MDLM / Mercury | **BLT-D** |
|--|----------------------|-----------|
| 디퓨전 적용 범위 | 전체 시퀀스 | **로컬 블록 내부만** |
| 단위 | 토큰 | **바이트** |
| 토크나이저 | 필요 | **불필요** (토크나이저-프리) |
| AR과의 관계 | 별도 모델 | **같은 가중치가 AR으로도 작동** |
| 가장 가까운 친척 | — | Block Diffusion (AR↔디퓨전 보간) |

**핵심 신규성**: **바이트 수준 + 블록 디퓨전 + 계층적 BLT 구조**의 교차점. 전체 시퀀스 디퓨전이 아니라 **계층적 바이트 모델 내부의 로컬 블록**에만 absorbing 디퓨전을 적용한 semi-autoregressive 방식.

---

## 10. 한계

1. **NFE·메모리 대역폭은 프록시 지표** — 실제 wall-clock 속도는 커널 구현·배칭·KV 캐시 관리에 의존, 논문이 직접 벤치마크 안 함
2. **BLT-D-16은 코드 생성에서 열화** — 큰 블록이 구조적/정확 출력 과제에 불리
3. **3B까지만 평가** — 대규모 스케일링 미검증
4. **코드 미공개** (2026.05 기준)
5. Future work: 디코더 크기 확대, $\mathcal{L}_{\text{clean}}$/$\mathcal{L}_{\text{mask}}$ 재가중, 추가 사전학습

---

## 11. 핵심 요약

| 질문 | 답 |
|------|-----|
| **무엇** | BLT(바이트 LLM)에 블록 디퓨전을 결합해 추론 가속 |
| **3가지 기법** | BLT-D(디퓨전), BLT-S(self-speculation), BLT-DV(디퓨전+검증) |
| **핵심 손실** | $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{clean}} + \mathcal{L}_{\text{mask}}$ |
| **추론** | 블록 B바이트를 병렬 unmask (confidence $\alpha$ / entropy $\gamma$) |
| **데이터** | BLT-1T (1조 토큰), 1B/3B 모델 |
| **결과** | 메모리 대역폭 최대 **92% 절감**, BLT-S는 품질 손실 0 |

> **한 줄**: "토크나이저-프리 바이트 모델 BLT의 마지막 약점이던 추론 속도를, 블록 디퓨전으로 메웠다."

---

## 12. 관련 블로그 포스트

- [Diffusion LM 서베이](diffusion-language-models-survey.md) — 전체 맥락
- [MDLM 심화 리뷰](mdlm-deep-review.md) — BLT-D가 쓰는 absorbing 디퓨전의 원형
- [LLaDA 심화 리뷰](llada-review.md) — 전체 시퀀스 마스킹 디퓨전
- [SEDD 심화 리뷰](sedd-deep-review.md)
- [Mercury 심화 리뷰](mercury-deep-review.md)
- [BLT 등 바이트 수준 LM 서베이](blt-byte-level-lm-survey.md)

---

## 참고 자료

- [Fast Byte Latent Transformer (arXiv:2605.08044)](https://arxiv.org/abs/2605.08044)
- [원조 BLT (arXiv:2412.09871)](https://arxiv.org/abs/2412.09871)
- [BLT 코드: github.com/facebookresearch/blt](https://github.com/facebookresearch/blt)
