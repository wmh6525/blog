---
title: "[심화 리뷰] CALM + Tencent WeChat AI 시리즈 — Scoring Rules · Patch Training · EAR · CALM 모든 수식 완전 해부"
date: 2026-05-28
tags: ["논문리뷰", "CALM", "연속벡터", "Tencent", "EnergyScore", "BrierLM", "Patch-Level"]
categories: ["ML/AI"]
summary: "CALM(Continuous Autoregressive Language Models) 심화 리뷰. Tencent WeChat AI(Chenze Shao 그룹)의 4편 시리즈 — Scoring Rules for LM, Patch-Level Training, EAR (Continuous Visual AR), CALM — 의 수학적 계보. Energy Score, BrierLM, Rejection-based Temperature Sampling의 모든 수식을 직관→유도→ablation으로 완전 분해."
math: true
toc: true
draft: false
---

## 들어가며 — Tencent WeChat AI의 4편 시리즈

CALM은 단독 논문이 아니라 **Chenze Shao, Fandong Meng, Jie Zhou** (Tencent WeChat AI Pattern Recognition Center) 그룹의 **4편 시리즈가 합류한 정점**이다:

```
[1] Scoring Rules for LM (ICML 2024, arXiv 2405.18906)
       ↓ Brier Score를 LM에 도입 → 학습 목적 함수의 일반화
[2] Patch-Level Training (ICLR 2025, arXiv 2407.12665)
       ↓ K개 토큰을 평균하여 patch 단위 학습 → 비용 1/K
[3] EAR — Continuous Visual AR (ICML 2025, arXiv 2505.07812)
       ↓ Energy Score를 비전 도메인에서 도입
[4] ★ CALM (2025.10, arXiv 2510.27688)
        Scoring Rules + Patch + Energy Score를 언어에서 통합
```

병행 작품: **SLED** (NeurIPS 2025, 음성 도메인) — 같은 그룹의 cross-modality 확장.

이 글은 이 5편을 차례로 다루되, **모든 수식을 (1) 직관 → (2) 수식 → (3) 왜 그렇게 설계 → (4) ablation** 순으로 분해한다.

---

# Part 1 — CALM (Continuous Autoregressive Language Models)

## 1. 핵심 동기

### 1.1 이산 토큰의 정보 천장

표준 LLM은 토큰당 정보량이 다음으로 제한된다:

$$\text{토큰당 정보} \leq \log_2 |V|$$

- 32K vocab → 15 bits
- 256K vocab → 18 bits

Vocabulary를 키워야 정보량이 늘지만 **지수적으로만** 증가 → 비실용적.

### 1.2 연속 벡터의 정보 확장성

차원 $l$의 연속 벡터는 (양자화 정밀도 b bits 기준):

$$\text{벡터당 정보} \approx l \cdot b$$

→ **차원에 대해 선형** 확장.

### 1.3 K배 단계 감소

K개 토큰을 단일 벡터로 묶으면 시퀀스 길이가 1/K → autoregressive step도 1/K.

### 1.4 도전 — Likelihood-Free 환경

연속 도메인에서는 softmax와 likelihood가 없다. 따라서:
- 학습 (loss는 어떻게?)
- 평가 (perplexity는 어떻게?)
- 온도 샘플링 (softmax temperature는 어떻게?)

→ CALM은 이 셋을 한꺼번에 해결하는 **likelihood-free toolkit**을 제안한다.

---

## 2. 아키텍처

### 2.1 Autoencoder (75M params)

```
Encoder:  K개 토큰 임베딩 → 토큰별 FFN → flatten Kd → d로 압축 → FFN → linear → l차원 latent
Decoder:  미러 구조, linear로 d → Kd 확장 → reshape → FFN → tied embedding projection → argmax
```

| 항목 | 값 |
|------|-----|
| K (chunk size) | **4** (기본) |
| l (latent 차원) | 128 |
| d (hidden) | 512 |
| 재구성 정확도 | **>99.9%** |
| Noise tolerance | $\sigma \approx 0.3$ (VAE-style) |
| Latent dropout | 0.15 |
| Token dropout | 0.15 |
| KL clip threshold | $\lambda_{KL} = 0.5$ |

### 2.2 CALM 본체 모델 크기

| 모델 | Layers | Hidden | Params | Generative Head |
|------|--------|--------|--------|----------------|
| CALM-M | 16 | 1024 | **371M** | 4 blocks |
| CALM-L | 16 | 1536 | **735M** | 4 blocks |
| CALM-XL | 16 | 2560 | **1.82B** | 4 blocks |

### 2.3 Energy-Based Generative Head

```
입력: Transformer hidden state h_{i-1} (d-dim)
     + 균등 노이즈 ε ∼ U[-0.5, 0.5] (d-dim)
       ↓
L개 residual MLP 블록 (기본 L=3, transformer layer의 ~1/4)
  각 블록: Linear + SwiGLU + residual + linear projection
       ↓
출력: l차원 연속 벡터 z_i
```

**핵심**: 전체 모델 파라미터의 **약 10%**만 차지, **단 1 step forward**로 sampling (diffusion 100 step, flow matching 4 step과 대비).

---

## 3. 수학적 정식화 — 모든 수식 완전 해부

### 3.1 Autoencoder 손실 (Eq. 1)

$$\mathcal{L}_{ae}(x_{1:K}) = -\sum_{i=1}^{K} \log p_{dec}(x_i \mid z = f_{enc}(x_{1:K}))$$

**직관**: K개 토큰을 latent $z$로 압축한 뒤, decoder가 각 위치에서 원본 토큰의 log-probability를 최대화. 표준 categorical reconstruction.

**왜?**: K개 정보를 단일 벡터에 무손실 압축해야 다운스트림 AR이 의미 있다 → >99.9% 재구성이 목표.

### 3.2 전체 손실 (Eq. 2)

$$\mathcal{L}_{total} = \mathcal{L}_{ae} + \beta \cdot \mathcal{L}_{KL}, \quad \beta = 0.001$$

**왜 $\beta$가 이렇게 작은가?**: 재구성을 최우선시. 표준 VAE($\beta=1$)는 latent를 prior에 강하게 맞추지만 재구성이 망가짐. CALM은 **재구성 우선 + 약한 매니폴드 정규화**.

### 3.3 KL Divergence (Eq. 3)

$$\mathcal{L}_{KL} = -\frac{1}{2} \sum_{i=1}^{l} (1 + \log \sigma_i^2 - \sigma_i^2 - \mu_i^2)$$

**직관**: 가우시안 prior $\mathcal{N}(0, I)$에 대한 closed-form KL. Encoder가 $(\mu, \log \sigma^2)$를 출력, reparameterization $z = \mu + \sigma \odot \epsilon$.

### 3.4 ★ KL Clipping (Eq. 4) — 핵심 트릭

$$\mathcal{L}_{KL}^{\text{clip}} = \sum_{i=1}^{l} \max(\lambda_{KL}, \mathcal{L}_{KL,i}), \quad \lambda_{KL} = 0.5$$

**문제**: $\beta=0.001$로 매우 약한데도, 어떤 차원이 **완전히 prior에 붕괴**(posterior collapse)하면 그 차원은 정보 흐름 없음.

**해결**: 차원별 KL이 임계값 $\lambda_{KL}$ 미만이면 gradient 차단 → 차원당 **최소 0.5 nats**의 정보 흐름 강제.

**Ablation 효과**:
- KL only: 3.48 BrierLM
- + KL clip: **4.13** (+0.65)
- + token dropout: 4.55
- + latent dropout: **4.70** (전체 +1.22)

### 3.5 Autoregressive 목적 (Eq. 5, 6)

$$Z = (z_1, \ldots, z_L), \quad z_i = f_{enc}(x_{(i-1)K+1}, \ldots, x_{iK})$$

$$p(Z) = \prod_{i=1}^{L} p(z_i \mid z_{\lt i}), \quad L = T/K$$

**해석**: 토큰 시퀀스 길이 $T$를 vector 시퀀스 길이 $L = T/K$로 압축 후 표준 AR factorization.

**문제**: $p(z_i \mid z_{\lt i})$는 **연속 분포**이므로 softmax 불가 → 새로운 학습 목적이 필요하다.

### 3.6 ★ Energy Score (Eq. 9) — CALM의 심장

$$S(P, y) = \mathbb{E}_{x', x'' \sim P}[\| x' - x'' \|^\alpha] - 2 \, \mathbb{E}_{x \sim P}[\| x - y \|^\alpha]$$

$\alpha \in (0, 2)$일 때 **strictly proper scoring rule** (Gneiting & Raftery 2007).

**두 항의 직관**:

| 항 | 의미 | 보상 대상 |
|----|------|---------|
| $\mathbb{E}_{x', x'' \sim P}[\| x' - x'' \|^\alpha]$ | 분포 $P$ 내 두 독립 샘플 간 평균 거리 | **다양성** |
| $-2 \, \mathbb{E}_{x \sim P}[\| x - y \|^\alpha]$ | 샘플과 정답 $y$ 사이 거리 (음수) | **정확도** |

**왜 strictly proper인가?**: $S(P, Q) \leq S(Q, Q)$는 $P = Q$일 때만 등호 → 두 항의 균형이 정확히 데이터 분포에서 최대화.

### 3.7 ★ 왜 $\alpha = 1$인가? (CALM의 결정적 선택)

**$\alpha \to 2$인 경우**: $\| \cdot \|^2$가 되며 cross term이 전개되면

$$S(P, y) \propto -\| \mathbb{E}_P[x] - y \|^2$$

→ **평균값 예측에만 의존** → mean collapse, 분포 정보 소실. 여전히 proper지만 **strictly가 아님**.

**$\alpha \lt  1$인 경우**: 그래디언트 폭발 ($\alpha = 0.75$에서 학습 실패).

**$\alpha = 1$**: 안정적이고 최고 성능 (BrierLM **4.70**). $\alpha = 1.25$에서 4.42로 하락.

**Likelihood-free의 핵심**: 이 손실은 $p(x)$ 평가가 필요 없음 — **모델에서 샘플만 뽑으면 됨** → 연속 latent에 완벽 적합.

### 3.8 Energy Loss Monte-Carlo 추정자 (Eq. 10)

실용적으로 계산하려면:

$$\mathcal{L}_{energy} = \sum_{i=1}^{L} \left[ \frac{2}{NM} \sum_{n=1}^{N} \sum_{m=1}^{M} \| z_{i,m} - \tilde{z}_{i,n} \| - \frac{1}{N(N-1)} \sum_{n \neq k} \| \tilde{z}_{i,n} - \tilde{z}_{i,k} \| \right]$$

| 기호 | 의미 |
|------|------|
| $\tilde{z}_{i,n}$ ($n=1..N$) | 모델 head가 생성한 $N$개 샘플 ($N=8$) |
| $z_{i,m}$ ($m=1..M$) | Target에서 뽑은 $M$개 샘플 ($M=100$, stochastic AE 출력) |

**왜 N=8, M=100?**: 비용 대비 성능 최적점. N=12, M=200은 성능 향상 미미.

### 3.9 ★ BrierLM 평가 메트릭 (Eq. 12, 14, 15) — 두 번째 핵심

표준 Brier score (Eq. 12):

$$\text{Brier}(P, y) = 2 P(y) - \sum_x P(x)^2$$

이는 strictly proper. 다만 $\sum_x P(x)^2$가 **vocab 전체 합산 필요** → likelihood-free 모델에선 평가 불가.

### 3.10 ★ Unbiased 추정자 (Eq. 14) — 마법

단 **2개의 샘플**만으로 Brier를 unbiased로 추정:

$$\text{Brier}(P, y) \approx \mathbb{1}[x_1 = y] + \mathbb{1}[x_2 = y] - \mathbb{1}[x_1 = x_2], \quad x_1, x_2 \sim P \text{ i.i.d.}$$

**유도** — 각 항의 기댓값을 보자:

1. $\mathbb{E}[\mathbb{1}[x_1 = y]] = P(y)$
2. $\mathbb{E}[\mathbb{1}[x_2 = y]] = P(y)$
3. $\mathbb{E}[\mathbb{1}[x_1 = x_2]] = \sum_x P(x) \cdot P(x) = \sum_x P(x)^2$ ← **collision probability**!

따라서:

$$\mathbb{E}[\text{추정자}] = 2P(y) - \sum_x P(x)^2 = \text{Brier}(P, y) \checkmark$$

**핵심**: 두 i.i.d. 샘플이 같을 확률 = $\sum P(x)^2$. 이 collision probability trick으로 $|V|$ 합산 없이 unbiased.

### 3.11 BrierLM 조합 (Eq. 15)

n-gram 단위 Brier-n을 계산한 후:

$$\text{BrierLM} = 100 \cdot \left( \prod_{n=1}^{4} \text{Brier-}n \right)^{1/4}$$

**기하평균** → 한 n에서 성능이 0이면 전체 0 (엄격 평가).

**Cross-entropy와의 상관**: Pearson $r = -0.966$ — perplexity의 sample-only 대체재로서의 정당성 입증.

### 3.12 ★ Rejection-based Temperature Sampling (Algorithm 1)

목표 분포:

$$P_T(x) = \frac{P(x)^{1/T}}{Z_T}, \quad Z_T = \sum_x P(x)^{1/T}$$

$T \in (0, 1)$일 때 $1/T > 1$. 문제: $P(x)$를 계산할 수 없음.

**분해**: $\frac{1}{T} = n + \alpha$, $n = \lfloor 1/T \rfloor$, $\alpha = 1/T - n \in [0, 1)$.

**Stage 1 (정수부 $n$)**:
- 모델에서 $n$개 i.i.d. 샘플 추출
- **전부 동일하면 채택** (그 값을 $x$로), 아니면 기각하고 재시작
- 채택 확률 $\propto P(x)^n$ ($n$개가 다 $x$일 확률)

**Stage 2 (분수부 $\alpha$ — Bernoulli Factory)**:
- 추가 샘플들로 $P(x)^\alpha$에 비례하는 채택률 시뮬레이션
- 핵심 트릭: $i$번째 시도에서 $\alpha/i$ 확률 검사 → 정확히 $P(x)^\alpha$에 비례

**Theorem 5.1 (정확성)**: Algorithm 1의 채택 분포 = $P_T$.

**Theorem 5.2 (기대 비용)**:

$$\mathbb{E}[N_{total}] = \frac{n + \mathbb{1}[\alpha \gt  0] \cdot \sum_x P(x)^{1/T - 1}}{Z_T}$$

**Corollary 5.2 (상한)**:

$$\mathbb{E}[N_{total}] \leq \begin{cases} (1 + n) / Z_T & 0 \lt  T \leq 0.5 \\ (1 + |X|^{2 - 1/T}) / Z_T & 0.5 \lt  T \lt  1 \end{cases}$$

$T \to 1$에서 vocab 크기 $|X|$에 의존 → 고온 영역에서 비싸짐.

### 3.13 Batch Approximation (Algorithm 2, 실제 사용)

- $N \gg n$인 큰 배치 샘플 추출
- 각 unique 샘플의 등장 횟수 $c_x$ 카운트
- $c_x \geq n$인 후보를 $\binom{c_x}{n}$ 가중치로 선택
- 부족 시 매칭 요건을 점진 완화

**Theorem 5.3 (Asymptotic Unbiasedness)**:

$$\lim_{N \to \infty} P_{alg}(x; N) = P_T(x)$$

실험적으로 $N \approx 100$ ≈ $T \approx 0.6$, $N \approx 500$ ≈ $T \approx 0.5$에 해당.

---

## 4. 학습 절차

### Stage 1 — Autoencoder Pretraining

| 항목 | 값 |
|------|-----|
| 데이터 | Pile-uncopyrighted **15B 토큰** (files 00-01) |
| Optimizer | AdamW ($\beta_1$=0.9, $\beta_2$=0.95, $\epsilon$=1e-8) |
| LR | 3e-4 constant, warmup 2000 |
| Weight decay | 0.1 |
| Grad clip | 1.0 |
| Batch size | 512K tokens |
| Steps | 30K |
| Tokenizer | LLaMA-3 BPE |
| 차원 | $d=512$, $l=128$ (K=4) |

### Stage 2 — CALM LM Training

| 항목 | 값 |
|------|-----|
| 데이터 | Pile 나머지 **~215B tokens** (files 02-29) |
| Validation | WikiText document-level |
| Batch size | **2M tokens** |
| Steps | 250K |
| Context | 8192 tokens (= 2048 vector steps for K=4) |
| Energy loss | $N=8$, $M=100$, $\alpha=1$ |

**Compute**: CALM-M = 3.7e20 FLOPs (Transformer-S baseline 6.6e20 대비 **-44%**).

---

## 5. 추론 절차

매 step $i$:
1. 직전 vector $z_{i-1}$ → decoder → K개 discrete 토큰 (argmax)
2. K개 토큰 임베딩 → 2-layer MLP 압축 → Transformer 입력
3. Transformer → hidden state $h_{i-1}$
4. Energy head $(h_{i-1}, \epsilon \sim U[-0.5, 0.5])$ → 단일 forward → $z_i$
5. K개 토큰을 한 번에 출력

**왜 discrete 재인코딩?**: Table 5 결과 — discrete 입력 BrierLM 4.70, 연속 latent 직접 입력 3.25, 결합 4.40. **토큰 공간에서의 grounding이 필수**(에러 누적 방지).

---

## 6. 벤치마크

### Table 1 (메인)

| 모델 | Params | Train FLOPs | Infer FLOPs | BrierLM |
|------|--------|------------|------------|---------|
| Transformer-S | 281M | 6.6e20 | 4.4e8 | 6.05 |
| Transformer-M | 465M | 11.9e20 | 7.9e8 | 7.07 |
| Transformer-L | 849M | 22.5e20 | 15.0e8 | 8.98 |
| **CALM-M** | 371M | **3.7e20** | **2.9e8** | **5.72** |
| **CALM-L** | 735M | **7.7e20** | **4.6e8** | **6.58** |
| **CALM-XL** | 1.82B | **19.5e20** | **9.4e8** | **8.53** |

CALM-M = Transformer-S 대비 학습 **-44%**, 추론 **-34%** FLOPs.

### Generative Head 비교

| Head | BrierLM | Steps |
|------|---------|-------|
| **Energy** | **4.7** | **1** |
| Flow matching | 4.4 | 4 |
| Diffusion | 3.8 | 100 |

**Energy가 단일 step에서 최강**.

### K Ablation

K=4가 sweet spot. K=8은 모델 capacity 한계로 하락. K=1은 연속 자체가 어려워 불리.

---

## 7. 한계

1. Autoencoder가 **context-agnostic** — 청크를 독립 처리. context-aware AE는 future work.
2. **High-T 비용** — $T \to 1$에서 rejection 비용 $|V|^K$로 폭발.
3. **Low-T rejection rate** — $T=0.2$면 n=5개 동일 샘플 필요 → batch approx 의존.
4. **K=8 capacity** — 더 큰 모델로 해결 필요.
5. **K=1의 연속 단점** — 이산 task보다 어렵고 초기 수렴 느림.
6. **Discrete grounding 필수** — pure continuous는 underperform.

---

# Part 2 — 같은 그룹의 관련 4편

## 8. Language Generation with Strictly Proper Scoring Rules (ICML 2024)

- **arXiv**: 2405.18906
- **저자**: Chenze Shao, Fandong Meng, Yijin Liu, Jie Zhou
- **코드**: [github.com/shaochenze/ScoringRulesLM](https://github.com/shaochenze/ScoringRulesLM)

### 핵심 기여

Log-likelihood(logarithmic score) **외의** strictly proper scoring rule을 LM에 도입.

### 두 가지 새 손실

**Brier score loss**:

$$\mathcal{L}_{\text{Brier}} = -\mathbb{E}_x \sum_t \left[ 2 p_\theta(x_t \mid x_{\lt t}) - \| p_\theta(\cdot \mid x_{\lt t}) \|^2 \right]$$

**해석**: 정답 토큰의 확률 2배에서, 분포의 전체 norm 제곱을 뺀다. norm 제곱은 분포가 한 점에 집중될수록 커지므로 → 모델이 한 토큰에만 확률 몰아주는 것을 방지하면서 정답을 맞추도록.

**Spherical score loss**:

$$\mathcal{L}_{\text{Spherical}} = -\mathbb{E}_x \sum_t \frac{p_\theta(x_t \mid x_{\lt t})}{\| p_\theta(\cdot \mid x_{\lt t}) \|}$$

**해석**: 정답 확률을 분포의 L2 norm으로 정규화 → 분포의 "방향"이 정답을 가리키는지 측정.

### Token-level Decomposition 정리

**핵심 결과**: 두 손실 모두 token-level에서 적용해도 **strictly proper 유지**.

### LLaMA 결과

- LLaMA-7B WMT22: 17.6 → **20.9** BLEU (+3.3, Brier)
- CNN/DailyMail summarization: ROUGE-1 28.66 → **32.15** (+3.49, Brier)
- LLaMA-13B에서도 +2~3 BLEU 향상

### CALM과의 관계

→ **CALM의 BrierLM 평가 메트릭의 직접 토대**. 그리고 strictly proper scoring rule의 LM 적용이 가능함을 입증함으로써 CALM의 Energy Score 학습으로 가는 다리.

---

## 9. Patch-Level Training for LLMs (ICLR 2025)

- **arXiv**: 2407.12665
- **저자**: Chenze Shao, Fandong Meng, Jie Zhou
- **코드**: [github.com/shaochenze/PatchTrain](https://github.com/shaochenze/PatchTrain)

### 핵심 기여

K개의 연속 토큰의 embedding을 **평균**해서 단일 patch embedding 생성.

$$e_{\text{patch}} = \frac{1}{K} \sum_{k=1}^{K} e(x_{(i-1)K+k})$$

### 학습 손실

Patch가 가진 K개 토큰의 cross-entropy를 모두 단일 head로 예측:

$$\mathcal{L}_{\text{patch}} = -\sum_{k=1}^{K} \log p_\theta(x_{(i-1)K+k} \mid e_{\text{patch}, 1:i-1})$$

### Two-Stage 학습

- $\lambda$ 비율: patch-level (시퀀스 길이 1/K → 비용 1/K)
- $(1-\lambda)$ 비율: token-level fine-tuning

**총 비용**:

$$\text{Cost} = \frac{\lambda}{K} + (1 - \lambda) \cdot \text{baseline}$$

$K=4, \lambda = 2/3$에서 **0.5× 비용**.

### 결과 (370M ~ 2.7B Transformer, Pile 360B)

- Perplexity 동급 또는 개선
- Zero-shot avg accuracy **+0.5%** (MMLU, HellaSwag, PIQA, WinoGrande, ARC-E/C)
- MT-Bench 동등

### CALM과의 관계

- 동일한 $K=4$ chunk-size 발상의 **이산 도메인 버전**
- Patch 평균이라는 **단순 압축** → CALM의 **autoencoder 기반 압축** (>99.9% reconstruction)으로 진화
- "K개를 하나로 다루면 비용이 1/K"라는 motif 공유

---

## 10. EAR — Continuous Visual AR (ICML 2025)

- **arXiv**: 2505.07812
- **저자**: Chenze Shao, Fandong Meng, Jie Zhou
- **코드**: [github.com/shaochenze/EAR](https://github.com/shaochenze/EAR)

### 핵심 기여

비전 도메인에서 vector quantization 없이 continuous autoregressive 생성.

### Energy Score 손실 (CALM과 동일 형태)

$$\mathcal{L}(p, y) = \| x_1 - y \|^\alpha + \| x_2 - y \|^\alpha - \| x_1 - x_2 \|^\alpha$$

### 기존 방법 통합 분석 — 모두 scoring rule의 특수 케이스

| 방법 | Scoring rule |
|------|------------|
| **GIVT** (Generative Infinite-Vocab Transformer) | Logarithmic score 최대화 (GMM 기반 likelihood) |
| **MAR** (diffusion loss) | Hyvärinen score $-\mathbb{E}[\| \nabla \log p - \nabla \log q \|^2]$ 최대화 |
| **EAR** | Energy score (위 식) |

### ImageNet 256×256 결과 (with CFG)

| 모델 | Params | FID ↓ | IS ↑ |
|------|--------|-------|------|
| EAR-B | 205M | 2.83 | 253.3 |
| EAR-L | 474M | 2.37 | 273.8 |
| **EAR-H** | **937M** | **1.97** | **289.6** |

### CALM과의 관계

- **방법론적으로 CALM의 직접 전구체**
- 동일한 energy score loss, MLP head + uniform noise 구조
- 비전(EAR) → 언어(CALM)로 동일 프레임워크 이식
- CALM은 추가로 (a) BrierLM 평가, (b) rejection-based temperature sampling, (c) K-token autoencoder 압축 도입

---

## 11. SLED — Speech Energy Distance LM (NeurIPS 2025, 자매 논문)

- **arXiv**: 2505.13181
- **저자**: Zhengrui Ma, Yang Feng, **Chenze Shao**, Fandong Meng, Jie Zhou, Min Zhang
- **코드**: [github.com/ictnlp/SLED-TTS](https://github.com/ictnlp/SLED-TTS)

### 핵심 기여

음성 waveform → 연속 latent 시퀀스 → **energy distance** AR 학습.

### 효과

- Residual vector quantization 회피 → 이산화 오류 제거
- Hierarchical 구조 제거
- Zero-shot 및 스트리밍 TTS에서 강한 성능

### CALM과의 관계

- 동일한 energy distance 학습 목적
- Cross-modality 적용
- **CALM(언어) · EAR(이미지) · SLED(음성) = 동일 기술 패밀리의 3-modality 확장**

---

## 12. 시리즈 진화 종합

```
NAT (병렬 생성, 속도)
   ↓
Patch-Level Training (token 압축, 학습 효율)
   ↓
Scoring Rules for LM (학습 목적 함수 일반화)
   ↓
   ├── EAR (연속 vision AR)
   └── SLED (연속 speech AR)
        ↓
★ CALM (연속 language AR, 통합 완성형)
```

**4가지 축의 합류**:

| 축 | 기여 논문 | CALM에서 |
|----|---------|---------|
| **청크 압축** | Patch-Level Training | Autoencoder K=4 |
| **새 scoring rule** | Scoring Rules for LM | Energy Score, BrierLM |
| **연속 AR** | EAR, SLED | Continuous LM |
| **Likelihood-free 도구** | (모두) | Sampling, 평가, 학습 통합 |

---

## 13. 한 줄 요약

> **CALM은 "토큰의 정보 천장"을 깨고 연속 벡터로 가는 첫 본격 LM이며, 그 모든 구성요소(Energy Score 학습, BrierLM 평가, Rejection Sampling)는 같은 그룹의 4편 시리즈가 4년에 걸쳐 축적한 결과다.**

---

## 14. 관련 블로그 포스트

- [연속 의미 공간 언어 모델 서베이](continuous-vector-language-models.md) — CALM이 속한 큰 흐름
- [Diffusion LM 서베이](diffusion-language-models-survey.md) — Energy Score와 diffusion의 비교
- [DPO 심화 리뷰](dpo-review.md) — Likelihood-free 학습의 다른 사례
- [BLT-D 리뷰](blt-d-review.md), [MBLM 리뷰](mblm-review.md) — 같은 시대 바이트 LM

---

## 참고 자료

### CALM
- [CALM 논문 (arXiv:2510.27688)](https://arxiv.org/abs/2510.27688) | [HTML](https://arxiv.org/html/2510.27688v1)
- [저자 블로그 / 프로젝트](https://shaochenze.github.io/blog/2025/CALM/)
- [GitHub: shaochenze/calm](https://github.com/shaochenze/calm)

### 시리즈
- [Scoring Rules for LM (ICML 2024, arXiv:2405.18906)](https://arxiv.org/abs/2405.18906)
- [Patch-Level Training (ICLR 2025, arXiv:2407.12665)](https://arxiv.org/abs/2407.12665)
- [EAR (ICML 2025, arXiv:2505.07812)](https://arxiv.org/abs/2505.07812)
- [SLED (NeurIPS 2025, arXiv:2505.13181)](https://arxiv.org/abs/2505.13181)

### 저자
- [Chenze Shao homepage](https://shaochenze.github.io/)
- [DBLP](https://dblp.org/pid/227/3123.html)
