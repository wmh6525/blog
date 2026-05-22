---
title: "[논문 리뷰] MDLM 심화 — '마스킹 디퓨전 = 가중 MLM' 정리의 완전 유도와 학습/추론 해부"
date: 2026-05-15
tags: ["논문리뷰", "Diffusion", "LLM", "MDLM", "마스킹디퓨전"]
categories: ["ML/AI"]
summary: "MDLM(Masked Diffusion Language Models) 심화 리뷰. NeurIPS 2024. Rao-Blackwellized ELBO가 가중 MLM 손실로 단순화되는 핵심 정리, SUBS 파라미터화, OpenWebText/LM1B 학습 데이터·하이퍼파라미터, 캐싱 샘플러, SEDD 대비 17% 개선까지 논문 기반 완전 해부."
math: true
toc: true
draft: false
---

## 논문 정보

- **제목**: Simple and Effective Masked Diffusion Language Models
- **저자**: Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, Volodymyr Kuleshov
- **소속**: Cornell Tech
- **학회**: NeurIPS 2024
- **arXiv**: 2406.07524
- **코드**: [github.com/kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm)

> 이 글은 [Diffusion LM 서베이](diffusion-language-models-survey.md)의 MDLM 심화 편이다. 핵심 정리의 유도와 학습·추론 절차를 논문 그대로 해부한다.

---

## 1. 한 줄 요약

> **마스킹 디퓨전의 ELBO가 BERT MLM 손실의 시간 가중 평균으로 정확히 단순화됨을 증명 — "BERT가 사실은 generator였다".**

---

## 2. 핵심 이론 (논문의 중심 기여)

### 2.1 Forward 프로세스 — Interpolating Discrete Diffusion

토큰은 $K$개 범주의 원핫 벡터 ($K$번째는 $[\text{MASK}]$ 토큰 $m$). Forward 마진:

$$q(z_t | x) = \text{Cat}(z_t; \alpha_t x + (1-\alpha_t) \pi)$$

$\alpha_t$는 단조 감소, $\alpha_0 \approx 1$, $\alpha_1 \approx 0$. **마스킹 디퓨전**에서는 $\pi = m$.

마스킹되면 영원히 마스킹: $q(z_t | z_{t'} = m) = \text{Cat}(z_t; m)$.

### 2.2 단순화된 Posterior

$$q(z_s | z_t, x) = \begin{cases} \text{Cat}(z_s; z_t) & z_t \neq m \\ \text{Cat}\left(z_s; \frac{(1-\alpha_s) m + (\alpha_s - \alpha_t) x}{1 - \alpha_t}\right) & z_t = m \end{cases}$$

### 2.3 SUBS 파라미터화 (기여 1)

디노이징 네트워크 $x_\theta(z_t, t)$가 $x$를 근사. SUBS는 두 가지 구조적 성질을 **출력 치환으로** 강제 (아키텍처 변경 없음):

1. **Zero Masking Probabilities**: $\langle x, m \rangle = 0$이므로 **MASK 로짓을 $-\infty$로 치환** → 모델이 마스크 토큰을 절대 예측 안 함
2. **Carry-Over Unmasking**: $z_t$가 이미 비마스킹이면 $x_\theta(z_t, t) = z_t$ → 비마스킹 입력을 출력으로 직접 복사. 드러난 토큰은 이후 동결

### 2.4 ★ 핵심 정리: Rao-Blackwellized ELBO → 가중 MLM (기여 2)

SUBS 하에서, 연속 시간 극한($T \to \infty$)의 NELBO:

$$\mathcal{L}_{\text{NELBO}}^\infty = \mathbb{E}_q \int_{t=0}^{1} \frac{\alpha_t'}{1 - \alpha_t} \cdot \log\langle x_\theta(z_t, t), x \rangle \, dt$$

시퀀스 $x^{1:L}$에 대해 (forward 노이징이 토큰별 독립):

$$\mathcal{L}_{\text{NELBO}}^\infty = \mathbb{E}_q \int_{t=0}^{1} \frac{\alpha_t'}{1 - \alpha_t} \sum_\ell \log\langle x_\theta^\ell(z_t^{1:L}, t), x^\ell \rangle \, dt$$

> **이것은 정확히 고전 MLM(BERT 스타일 마스킹 cross-entropy) 손실의 시간 가중 평균이다.**

**함의**:
- 생성 디퓨전 모델 ↔ encoder-only BERT 모델 연결
- 원칙적인 랜덤 마스킹 비율 제공
- BERT 스타일 모델에 원칙적 생성 능력 부여

**왜 "Rao-Blackwellization"인가?** $\langle x_\theta, m \rangle = 0$ 같은 기댓값을 *학습하지 않고 해석적으로* 계산. 분산도 경험적으로 감소.

**참고**: 손실이 모든 토큰에 대해 쓰여 있지만, **비마스킹 토큰은 정확히 0 기여** (carry-over가 복사 → $\log\langle x_\theta^\ell, x^\ell \rangle = \log 1 = 0$).

### 2.5 노이즈 스케줄 불변성

변수 치환 $\gamma \equiv \log(1-\alpha_t)$를 하면 손실이 **$\alpha_t$의 함수 형태와 무관**해진다. 경험적 검증: log-linear, cosine, linear 모두 같은 likelihood (3.30 BPD)지만 **log-linear가 가장 낮은 분산** (1.81 vs cosine 3.30 vs linear 7.57).

---

## 3. 아키텍처

- **백본**: Diffusion Transformer (DiT) + **RoPE**. SEDD의 Transformer 아키텍처 사용
- **시간 컨디셔닝**: 선택적, **제거 가능**. OpenWebText에서 제거해도 perplexity 거의 동일 (23.21 vs 23.05) → **제거하면 2배 추론 가속 + 캐싱 가능** (config 기본값 `time_conditioning: False`)

### 모델 크기

| | 히든 | 레이어 | 헤드 | 시퀀스 |
|--|------|-------|------|--------|
| **small** (~110M, GPT-2 small) | 768 | 12 | 12 | 1024 |
| **medium** | 1024 | 24 | 16 | 1024 |

워드 임베딩 untied. 백본으로 DiT 외 AR Transformer, Mamba(`dimamba`)도 지원.

---

## 4. 학습 데이터

| 항목 | LM1B | OpenWebText (OWT) |
|------|------|-------------------|
| 코퍼스 | One Billion Words | OpenWebText |
| 토크나이저 | `bert-base-uncased` | **GPT-2 토크나이저** |
| 시퀀스 길이 | **128** | **1024** |
| 시퀀스 처리 | pad & truncate | 연결 & wrap, 문서 사이 EOS |
| 1M 스텝 학습 토큰 | ~33B | ~262B (token-position 524B) |

**디퓨전 토큰 수 공식**: 매 스텝 마스킹 비율 $p_m = t$ ($\mathbb{E}[t] = 0.5$) → 디퓨전 모델이 보는 토큰 = 스텝 × 배치 × 컨텍스트 × **0.5**. AR 베이스라인은 토큰 수 맞추려 절반 스텝만 학습.

---

## 5. 학습 절차

### 5.1 목적 함수

가중 MLM NELBO (마스킹 cross-entropy).

### 5.2 마스킹 스케줄

Log-linear: $\sigma(t) = -\log(1-t)$ → $\alpha_t = 1-t$, 마스킹 확률 $p_m = t$. config: `sigma_min: 1e-4`, `sigma_max: 20`.

### 5.3 학습 알고리즘 (Algorithm 1)

```
반복:
1. 문장 x^{1:L} ~ q(x) 샘플
2. t ~ U[0,1] 샘플
3. 각 토큰 독립 마스킹: z_t^ℓ ~ Cat(z_t^ℓ; α_t·x^ℓ + (1-α_t)·m)
4. 그래디언트 스텝: ∇_θ (α_t'/(1-α_t)) Σ_ℓ log⟨x_θ^ℓ(z_t, t), x^ℓ⟩
```

### 5.4 하이퍼파라미터

| 항목 | 값 |
|------|-----|
| 옵티마이저 | AdamW ($\beta_1$=0.9, $\beta_2$=0.999, $\epsilon$=1e-8), weight decay **0** |
| 학습률 | **3e-4**, 상수, 2500 스텝 선형 warmup |
| 글로벌 배치 크기 | **512** |
| Dropout | 0.1 |
| 그래디언트 클리핑 | 1.0 |
| 정밀도 | bf16 |
| 학습 스텝 | OWT 1M, LM1B 1M/5M/10M |
| EMA | 0.9999 |
| 시간 샘플러 | **Low-discrepancy (antithetic)** — $[0,1]$ 분할, 샘플 $i$는 $U[(i-1)/N, i/N]$에서 → ELBO 분산 감소 |

### 5.5 엔지니어링 레시피 (가장 큰 기여)

논문이 강조 — "(3)이 가장 큰 기여":
1. **토크나이저가 가장 중요** — 작은 어휘(D3PM의 8k)는 장거리 의존성 유발 → 현대 GPT-2/BERT 토크나이저 사용
2. **수치적으로 안정한 손실** — 전체 $\bar{Q}_t$ 전이 행렬 대신 마스킹 인덱스에서만 KL 평가
3. **현대 아키텍처** — DiT + RoPE (D3PM의 T5 대신)
4. **Low-discrepancy 샘플러** — 분산 감소

---

## 6. 추론 / 샘플링 절차

### 6.1 효율적 Ancestral Sampling

완전 마스킹 시퀀스에서 시작 → reverse를 $T$ 스텝 이산화 → 각 스텝마다 마스크 토큰을 실제 토큰으로 점진 교체. 비마스킹 토큰은 동결 (carry-over).

### 6.2 캐싱 샘플러 (`ddpm_cache`, 핵심 기여)

네트워크가 **시간 컨디셔닝 없으면**, 그리고 한 스텝에서 새로 비마스킹되는 토큰이 없으면 (큰 $T$에서 초기 디노이징 시 흔함), $z_{s-1/T}$를 **캐시된** $x_\theta(z_t)$에서 샘플 가능 → **네트워크 forward pass 완전 스킵**.

SEDD는 시간 의존 rate를 모델링하므로 캐싱 불가.

### 6.3 Semi-autoregressive (SAR) — 임의 길이 생성

$L$개 토큰 생성 후 $L'$ 더 확장하려면: 이전 블록의 마지막 $L-L'$ 토큰을 **prefix**로 (마스크 대신 clean 토큰으로 초기화) → 새 $L'$ 위치 생성. Carry-over로 prefix는 매 스텝 복사. 무한 반복 가능.

### 6.4 속도 (64 샘플, A5000, T=10k)

| 샘플러 | 시간 |
|--------|------|
| SEDD | 229.3s |
| MDLM + `ddpm` | 206.6s |
| **MDLM + `ddpm_cache`** | **60.4s** (SEDD 대비 ~3.8배 빠름) |

---

## 7. 벤치마크

### 7.1 LM1B Test Perplexity (110M, ↓)

| 모델 | PPL |
|------|-----|
| D3PM absorb | ≤76.90 |
| DiffusionBERT | ≤63.78 |
| **SEDD (33B 토큰)** | ≤32.79 |
| AR Transformer (33B) | 22.32 |
| **MDLM (33B 토큰)** | **≤27.04** |
| **MDLM (327B 토큰)** | **≤23.00** |

- **SEDD 대비 17% 개선**: ≤27.04 vs ≤32.79 (같은 33B 토큰)
- **AR과 14% 이내**: ≤27.04 vs AR 22.32. 327B 학습 시 ≤23.00 → AR 327B(20.86)에 근접

### 7.2 OpenWebText Test Perplexity (↓)

| 모델 | PPL |
|------|-----|
| AR Transformer | 17.54 |
| SEDD | ≤24.10 |
| **MDLM** | **≤23.21** |

### 7.3 Zero-shot Perplexity (OWT 학습 모델, ↓)

| 모델 | PTB | WikiText | LM1B | Lambada | Pubmed | Arxiv |
|------|-----|----------|------|---------|--------|-------|
| AR | 82.05 | 25.75 | 51.25 | 51.28 | 49.01 | 41.73 |
| SEDD | 100.09 | 34.28 | 68.20 | 49.86 | 44.53 | 38.48 |
| **MDLM** | **95.26** | **32.83** | **67.01** | **47.52** | **41.89** | **37.37** |

MDLM이 SEDD를 전 데이터셋에서 능가. **Lambada, Pubmed, Arxiv에서는 AR도 능가** (OWT에서 먼 도메인 → 디퓨전의 마스킹 목적이 더 강건).

### 7.4 Ablation (LM1B, T=1000, 5 시드)

| 구성 | PPL |
|------|-----|
| MDLM (전체) | 27.04 |
| -연속 시간 (이산 T=1000) | 27.19 |
| & -carry-over | 28.56 |
| & -zero-masking (= D3PM) | 28.51 |

→ **Carry-over unmasking이 ~1.5 PPL 기여**.

---

## 8. 영향 — MDLM이 가능하게 한 것

- **ByteDance Seed Diffusion** — "MDLM이 ByteDance의 Seed Diffusion(가장 빠른 산업급 디퓨전 LLM)을 구동"
- **NVIDIA GenMol** — 분자 생성 모델
- LLaDA, Mercury 등 스케일된 MDLM 물결의 정전적 레퍼런스

---

## 9. 한계

1. **Perplexity는 상한** (ELBO), AR의 정확한 PPL과 직접 비교 불가
2. **AR과 in-domain 격차** (OWT 23.21 vs 17.54)
3. **고품질 샘플링은 많은 스텝 필요** (T=1000-10000), few-step은 급격히 열화 (T=10 → 42.18)
4. **캐싱 가속은 시간 컨디셔닝 제거 필요**
5. **GPT-2-small / ~110M 스케일만 평가** — 대규모 스케일링 연구 없음

---

## 10. 핵심 요약

| 질문 | 답 |
|------|-----|
| **무엇** | 마스킹 디퓨전 ELBO = 가중 MLM 증명 |
| **핵심 정리** | $\mathcal{L} = \int \frac{\alpha_t'}{1-\alpha_t} \sum_\ell \log\langle x_\theta^\ell, x^\ell\rangle dt$ |
| **SUBS** | Zero-masking + Carry-over (출력 치환) |
| **데이터** | OWT (GPT-2 토크나이저, 1024), LM1B (BERT, 128) |
| **학습** | 배치 512, LR 3e-4, 1M 스텝, low-discrepancy 샘플러 |
| **추론** | `ddpm_cache` 샘플러 (SEDD 대비 3.8배 빠름), SAR로 임의 길이 |
| **결과** | SEDD 대비 17% 개선, AR과 14% 이내 |

---

## 11. 관련 블로그 포스트

- [Diffusion LM 서베이](diffusion-language-models-survey.md)
- [LLaDA 심화 리뷰](llada-review.md)
- [SEDD 심화 리뷰](sedd-deep-review.md)
- [Mercury 심화 리뷰](mercury-deep-review.md)

---

## 참고 자료

- [MDLM (arXiv:2406.07524)](https://arxiv.org/abs/2406.07524)
- [프로젝트 페이지: s-sahoo.com/mdlm](https://s-sahoo.com/mdlm/)
- [코드: github.com/kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm)
