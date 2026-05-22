---
title: "[논문 리뷰] SEDD 심화 — Score Entropy 손실의 수식 유도와 학습/추론 완전 해부"
date: 2026-05-15
tags: ["논문리뷰", "Diffusion", "LLM", "SEDD", "ScoreEntropy"]
categories: ["ML/AI"]
summary: "SEDD(Score Entropy Discrete Diffusion) 심화 리뷰. ICML 2024 Best Paper. Concrete score, Score Entropy 손실의 Bregman 유도, Denoising Score Entropy, OpenWebText 학습 데이터·하이퍼파라미터, τ-leaping/analytic 샘플링, GPT-2 능가 perplexity까지 논문 기반 완전 해부."
math: true
toc: true
draft: false
---

## 논문 정보

- **제목**: Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution
- **저자**: Aaron Lou (Stanford), Chenlin Meng (Stanford / Pika), Stefano Ermon (Stanford)
- **학회**: **ICML 2024 Best Paper Award** (Oral)
- **arXiv**: 2310.16834
- **코드**: [github.com/louaaron/Score-Entropy-Discrete-Diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)

> 이 글은 [Diffusion LM 서베이](diffusion-language-models-survey.md)의 SEDD 심화 편이다. 수식 유도와 학습·추론 절차를 논문 그대로 해부한다.

---

## 1. 한 줄 요약

> **이산 확산을 "확률의 비율"로 모델링하는 Score Entropy 손실을 제안 — cross-entropy의 Bregman 일반화로, GPT-2를 perplexity에서 능가한 첫 비자회귀 모델.**

---

## 2. 이론적 배경

### 2.1 이산 확산 프로세스

- 상태 공간: 유한 지지 $\mathcal{X} = \lbrace 1, \ldots, N \rbrace$
- **Forward 프로세스**: 연속 시간 이산 마르코프 체인, 선형 ODE

$$\frac{dp_t}{dt} = Q_t p_t$$

- $Q_t$: 확산(rate) 행렬 — 비대각 원소 비음수, 열 합 0 (질량 보존)
- 보통 $Q_t = \sigma(t) Q$ (스칼라 노이즈 × 고정 $Q$)

- **Reverse 프로세스** (Kelly 1980):

$$\frac{dp_{T-t}}{dt} = \bar{Q}_{T-t} p_{T-t}, \qquad \bar{Q}_t(y, x) = \frac{p_t(y)}{p_t(x)} Q_t(x, y)$$

### 2.2 Concrete Score

비율 $p_t(y)/p_t(x)$가 **concrete score** — 연속의 $\nabla_x \log p_t$의 이산 유사물. 스코어 네트워크가 학습하는 것:

$$s_\theta(x, t)_y \approx \frac{p_t(y)}{p_t(x)} \quad (y \neq x)$$

### 2.3 Score Entropy 손실 (Definition 3.1)

$$\mathcal{L}_{SE} = \mathbb{E}_{x \sim p}\left[\sum_{y \neq x} w_{xy}\left(s_\theta(x)_y - \frac{p(y)}{p(x)} \log s_\theta(x)_y + K\left(\frac{p(y)}{p(x)}\right)\right)\right]$$

여기서 $K(a) = a(\log a - 1)$는 $\mathcal{L}_{SE} \geq 0$을 보장하는 정규화 함수.

### 2.4 왜 Bregman인가? (Fisher 대신)

기존 **Concrete Score Matching (CSM)**은 Fisher 발산 스타일 $\ell^2$ 손실:

$$\mathcal{L}_{CSM} = \frac{1}{2}\mathbb{E}\left[\sum_{y \neq x}\left(s_\theta(x)_y - \frac{p(y)}{p(x)}\right)^2\right]$$

**문제**: 비율 $p(y)/p(x)$는 **양수**여야 하는데, $\ell^2$ 손실은 음수/0 예측을 페널티하지 않아 실제로 발산. (논문 Appendix D: $\ell^2$ 손실로 바꾸면 likelihood 손실이 3-4배, perplexity는 ~10,000배 폭발)

Score Entropy는 **Bregman 발산** $D_F(s(x), p(y)/p(x))$ ($F = -\log$ 볼록 함수)에서 유도:
- 비음수, 대칭, 볼록
- cross-entropy를 양의 실수값으로 일반화 (확률 단순체에 국한 안 됨 → "entropy"라는 이름)
- **자연스러운 log-barrier** — $s_\theta \geq 0$을 유지

### 2.5 Denoising Score Entropy — 실용 버전 (Theorem 3.4)

$p$가 $p_0$의 변형이면 ($p(x) = \sum_{x_0} p(x|x_0) p_0(x_0)$):

$$\mathcal{L}_{DSE} = \mathbb{E}_{x_0 \sim p_0, x \sim p(\cdot|x_0)}\left[\sum_{y \neq x} w_{xy}\left(s_\theta(x)_y - \frac{p(y|x_0)}{p(x|x_0)} \log s_\theta(x)_y\right)\right]$$

확장 가능 — Monte Carlo가 $s_\theta(x)$ 한 번 평가만 필요.

### 2.6 Likelihood Bound (Theorem 3.6)

$$-\log p_0^\theta(x_0) \leq \mathcal{L}_{DWDSE}(x_0) + D_{KL}(p_{T|0}(\cdot|x_0) \| p_{\text{base}})$$

$\mathcal{L}_{DWDSE}$는 **diffusion weighted denoising score entropy** (forward 확산으로 가중된 DSE의 시간 적분). 이것이 ELBO를 형성 → max-likelihood 학습 가능.

### 2.7 Absorbing vs Uniform 전이

| | Uniform | Absorbing |
|--|---------|-----------|
| 정의 | 비대각 = 1, 대각 = $1-N$ | 대각 = -1, 마지막 행 = 1, MASK 열 = 0 |
| $p_{\text{base}}$ | 균등 분포 | MASK 상태 |
| 직관 | 모든 토큰으로 전이 | BERT 마스킹 |

---

## 3. 아키텍처

- **백본**: Diffusion Transformer (**DiT**) — bidirectional Transformer + 시간 컨디셔닝 (adaLN-zero)
- 내부 클래스명: `DDiT` (Diffusion DiT)
- **RoPE** 위치 인코딩, **FlashAttention**
- 시간 컨디셔닝: 전체 노이즈 레벨 $\sigma$로 파라미터화. `TimestepEmbedder` (sinusoidal → MLP → SiLU → adaLN)
- **출력 후처리**: 네트워크 출력을 **지수화**하여 $s_\theta$ 형성 (양수성 유지)

### 모델 크기

| | 히든 | 레이어 | 헤드 | 시퀀스 | 파라미터 |
|--|------|-------|------|--------|---------|
| **SEDD Small** | 768 | 12 | 12 | 1024 | ~90M |
| **SEDD Medium** | 1024 | 24 | 16 | 1024 | ~320M |

시간 컨디셔닝 네트워크 때문에 일반 Transformer보다 5-10% 더 많은 파라미터.

---

## 4. 학습 데이터

GPT-2 스케일 헤드라인 결과는 **OpenWebText**로 학습:

| 항목 | 값 |
|------|-----|
| 코퍼스 | **OpenWebText** (GPT-2 WebText의 오픈 복제) |
| 토크나이저 | **GPT-2 BPE**, 어휘 50,257 (absorbing은 +MASK = 50,258) |
| 시퀀스 길이 | **1024** |
| 데이터 준비 | sentence packing — 문서 토큰화, EOS 추가, 연결, 1024 블록으로 분할 |

추가 평가: text8 (문자 수준), One Billion Words (LM1B).

---

## 5. 학습 절차

### 5.1 손실

**Diffusion-weighted denoising score entropy** $\hat{\mathcal{L}}_{DWDSE}$ 최소화. 코드:

```python
t ∼ U[sampling_eps, 1]      # sampling_eps = 1e-3
(σ, dσ) = noise(t)
perturbed = graph.sample_transition(batch, σ)
log_score = log_score_fn(perturbed, σ)
loss = (dσ · score_entropy).sum(-1)
```

학습 iteration은 표준 AR 학습과 거의 같은 속도·메모리.

### 5.2 노이즈 스케줄

| 스케줄 | 정의 | 사용 |
|-------|------|------|
| **Log-linear** | $\bar{\sigma}(t) = -\log(1 - (1-\epsilon)t)$ | SEDD Absorb (기본) |
| **Geometric** | $\bar{\sigma}(t) = \sigma_{min}^{1-t} \sigma_{max}^t$ | SEDD Uniform |

Geometric은 $\sigma_{min} = 10^{-4}$, $\sigma_{max} = 20$ 사이 보간.

### 5.3 하이퍼파라미터

| 항목 | 값 |
|------|-----|
| 배치 크기 | **512** |
| 옵티마이저 | AdamW ($\beta_1=0.9$, $\beta_2=0.999$, $\epsilon$=1e-8) |
| 학습률 | **3e-4** |
| Weight decay | 0 |
| LR warmup | 2500 스텝 선형 |
| 그래디언트 클리핑 | 글로벌 norm 1.0 |
| 학습 iteration | **1,300,001** 스텝 |
| EMA | 0.9999 |
| 정밀도 | bfloat16 |

### 5.4 GPU 설정

- 8× A100 80GB 또는 16× A100 40GB 노드 (PyTorch DDP)
- SEDD medium은 그래디언트 누적 사용
- **하이퍼파라미터·아키텍처 탐색 없음** — 표준 레시피 그대로 (DiT, RoPE, "3e-4 LR, 0.9999 EMA")

---

## 6. 추론 / 샘플링 절차

Reverse 확산을 $x_T \sim p_{\text{base}}$에서 $x_0$까지 시뮬레이션. **Predictor-Corrector** 방식.

### 6.1 τ-leaping

Euler 스텝은 한 위치만 변경 (비효율). **τ-leaping**은 **모든 위치를 동시에** 업데이트:

$$\delta_{x_t^i}(x_{t-\Delta t}^i) + \Delta t \cdot Q_t(x_t^i, x_{t-\Delta t}^i) \cdot s_\theta(x_t, t)_{i, x_{t-\Delta t}^i}$$

### 6.2 Tweedie τ-leaping (Theorem 4.2)

모든 $p_t(y)/p_t(x)$ 비율을 활용해 최적 denoising — 이산 Tweedie 정리. $s_\theta$가 완벽히 학습됐다고 가정하면 모든 τ-leaping 전략 중 **최적** (참 reverse에 대한 KL 발산 최소화).

### 6.3 Analytic Predictor-Corrector

`analytic` predictor — staggered score $\exp(-d\sigma E) s_\theta$ 적용 후 범주 샘플링. **Analytic이 일반적으로 Euler 샘플링보다 우수** (특히 uniform 모델).

### 6.4 샘플링 스텝

- 기본 config: `predictor: euler`, `steps: 128`, `noise_removal: True`
- perplexity 평가 시 **32 ~ 2048 스텝**으로 시뮬레이션 (시퀀스 길이 1024)

### 6.5 Perplexity 계산

- ELBO 상한 $\mathcal{L}_{DWDSE}$로 계산 → 모든 perplexity는 **상한** ($\leq$)
- likelihood 적분을 **1000 타임스텝 랜덤 샘플링**으로 Monte Carlo 추정

---

## 7. 벤치마크

### 7.1 Zero-shot Unconditional Perplexity (↓) — Small

| 모델 | LAMBADA | WikiText2 | PTB | WikiText103 | 1BW |
|------|---------|-----------|-----|-------------|-----|
| GPT-2 | **45.04** | 42.43 | 138.43 | 41.60 | **75.20** |
| **SEDD Absorb** | ≤50.92 | **≤41.84** | **≤114.24** | **≤40.62** | ≤79.29 |
| SEDD Uniform | ≤65.40 | ≤50.27 | ≤140.12 | ≤49.60 | ≤101.37 |
| D3PM | ≤93.47 | ≤77.28 | ≤200.82 | ≤75.16 | ≤138.92 |
| PLAID | ≤57.28 | ≤51.80 | ≤142.60 | ≤50.86 | ≤91.12 |

### 7.2 Zero-shot — Medium

| 모델 | LAMBADA | WikiText2 | PTB | WikiText103 | 1BW |
|------|---------|-----------|-----|-------------|-----|
| GPT-2 | **35.66** | 31.80 | 123.14 | 31.39 | **55.72** |
| **SEDD Absorb** | ≤42.77 | **≤31.04** | **≤87.12** | **≤29.98** | ≤61.19 |

**핵심**:
- **SEDD Absorb가 GPT-2를 5개 중 3개 과제에서 능가** (WikiText2, PTB, WikiText103) — 비자회귀 모델이 현대 AR 모델과 perplexity에서 처음 경쟁
- 기존 디퓨전(D3PM, PLAID)은 전 과제에서 압도
- **25-75% perplexity 감소** = SEDD가 기존 디퓨전 베이스라인 능가

### 7.3 One Billion Words (↓)

| 방법 | Perplexity |
|------|-----------|
| Transformer (AR) | **31.98** |
| D3PM Absorb | ≤77.50 |
| Diffusion-LM | ≤118.62 |
| **SEDD Absorb** | **≤32.79** |

SEDD가 모든 이산 디퓨전을 **2배 이상** 능가, AR과 **1 perplexity 이내**.

### 7.4 생성 품질

- SEDD가 GPT-2를 일관되게 능가 (un-annealed 샘플링)
- GPT-2 품질을 **32배 적은 네트워크 평가**로 달성
- 같은 스텝 수에서 **6-8배** 더 나은 generative perplexity

---

## 8. 한계

1. **KV 캐시 불가** — 양방향 forward, 스텝 비교가 apples-to-apples 아님
2. **현대 대형 LLM과 격차** — GPT-2 스케일 경쟁이지 SOTA는 아님
3. **노이즈 스케줄 미탐색** — 체계적 탐색 안 함, 개선 여지
4. **Uniform이 Absorb보다 열등** — uniform은 선형 step/quality 곡선 부재
5. **Perplexity는 상한** — 정확한 likelihood가 아님

---

## 9. 핵심 요약

| 질문 | 답 |
|------|-----|
| **무엇** | 이산 확산을 확률 비율(concrete score)로 모델링 |
| **손실** | Score Entropy — cross-entropy의 Bregman 일반화 |
| **데이터** | OpenWebText, GPT-2 BPE, 시퀀스 1024 |
| **학습** | 배치 512, LR 3e-4, 1.3M 스텝, 8× A100 |
| **추론** | τ-leaping / Tweedie / analytic, 32-2048 스텝 |
| **결과** | GPT-2를 perplexity에서 능가한 첫 비AR 모델 |

---

## 10. 관련 블로그 포스트

- [Diffusion LM 서베이](diffusion-language-models-survey.md)
- [LLaDA 심화 리뷰](llada-review.md)
- [MDLM 심화 리뷰](mdlm-deep-review.md)
- [Mercury 심화 리뷰](mercury-deep-review.md)

---

## 참고 자료

- [SEDD (arXiv:2310.16834)](https://arxiv.org/abs/2310.16834)
- [코드: github.com/louaaron/Score-Entropy-Discrete-Diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)
- [Aaron Lou 블로그 (discrete diffusion)](https://aaronlou.com/blog/2024/discrete-diffusion/)
