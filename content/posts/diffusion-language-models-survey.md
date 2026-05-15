---
title: "[서베이] Diffusion Language Model 완전 정복 — Diffusion-LM부터 LLaDA, Mercury, GIDD까지"
date: 2026-05-06
tags: ["서베이", "Diffusion", "LLM", "LLaDA", "Mercury", "SEDD", "MDLM"]
categories: ["ML/AI"]
summary: "Diffusion Language Model(DLM)의 모든 것. 이미지 디퓨전을 텍스트로 가져온 동기부터 D3PM, Diffusion-LM, SEDD, MDLM, LLaDA, Mercury, GIDD까지. '마스킹 디퓨전 = 가중 MLM' 정리, score entropy, AR vs Diffusion 비교, 2026년 전망까지 50+ 논문 종합."
math: true
toc: true
draft: false
---

## 왜 Diffusion Language Model인가?

표준 LLM(GPT, LLaMA)은 **자회귀(autoregressive)** — 토큰을 **하나씩 순차적으로** 생성한다:

$$P(x_1, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})$$

이미지 분야에서는 **확산 모델(diffusion model)**이 GAN과 자회귀를 압도하고 있다. **그렇다면 텍스트도?**

### Diffusion이 텍스트에 가져오는 4가지 이점

| 이점 | 설명 |
|------|------|
| **병렬 생성** | 모든 토큰을 동시에 생성/정제 → 5-128배 빠름 |
| **양방향 컨텍스트** | causal mask 없이 전체 시퀀스 동시 attention |
| **반복적 정제** | 출력을 다시 보고 수정 가능 (AR의 exposure bias 해결) |
| **Reversal Curse 완화** | LLaDA가 GPT-4o를 역방향 시 완성에서 능가 |

### 핵심 도전

> **텍스트는 본질적으로 이산(discrete)이다 — 토큰 사이에 "가우시안 노이즈"가 자연스럽지 않다.**

→ 두 가지 해결책이 등장:
1. **Continuous DLM**: 토큰을 임베딩하고 임베딩 공간에서 확산
2. **Discrete DLM**: 토큰 알파벳에 직접 stochastic corruption

이 두 갈래의 진화를 50+ 논문으로 추적한다.

---

## 1. 역사적 흐름

```
2020-2021
  └── DDPM (Ho et al.) — 이미지 확산의 폭발적 성공

2021
  ├── D3PM (NeurIPS) — 첫 원리적 이산 확산
  └── Multinomial Diffusion / Argmax Flows (NeurIPS)

2022
  ├── Diffusion-LM (NeurIPS) — 첫 임베딩 공간 텍스트 확산
  ├── CDCD (DeepMind)
  ├── DiffuSeq (ICLR'23) — Seq2Seq 확산
  └── SSD-LM (ACL'23) — simplex 기반

2023
  ├── GENIE, AR-Diffusion (NeurIPS)
  ├── Plaid — 첫 GPT-2 능가 likelihood
  ├── PLANNER (Apple, NeurIPS)
  └── SEDD (ICML 2024 Best Paper)

2024
  ├── MDLM (NeurIPS) — masked diffusion = MLM 정리
  ├── MD4 (DeepMind)
  ├── DiffuLLaMA — AR → 확산 변환
  ├── Discrete Flow Matching (Meta)
  └── DoT — 확산 + CoT

2025
  ├── LLaDA 8B (ICLR Oral) — LLaMA3와 경쟁
  ├── Block Diffusion (ICLR Oral)
  ├── GIDD — self-correction
  ├── Mercury (Inception Labs) — 상용화
  ├── Gemini Diffusion (Google I/O)
  ├── Seed Diffusion (ByteDance) — 2,146 tok/s
  ├── Dream 7B, Dream-Coder, DiffuCoder
  └── d1, DCoLT — RL + reasoning

2026
  ├── LLaDA 2.0 (100B MoE)
  └── FS-DFM (128x 가속)
```

---

## 2. 수학적 기초

### Forward Process (노이즈 추가)

**연속**:
$$q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

마진:
$$q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)$$

**이산 (D3PM)**:
$$q(x_t \mid x_{t-1}) = \text{Cat}(x_t; p = x_{t-1} Q_t)$$

마진:
$$q(x_t \mid x_0) = \text{Cat}(x_t; p = x_0 \bar{Q}_t), \quad \bar{Q}_t = Q_1 Q_2 \cdots Q_t$$

### Reverse Process (denoising)

신경망 $p_\theta(x_{t-1} \mid x_t)$이 노이즈를 점진적으로 제거:

$$p_\theta(x_0) = \int p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t) \, dx_{1:T}$$

### ELBO

$$\log p(x_0) \geq \mathbb{E}_q\left[\log p(x_0 \mid x_1)\right] - \sum_{t>1} \mathbb{E}_q\left[D_{KL}(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t))\right]$$

---

## 3. 연속 디퓨전 LM (임베딩 공간)

### 3.1 Diffusion-LM (Stanford, NeurIPS 2022) — 시조

- **arXiv**: 2205.14217
- **저자**: Xiang Lisa Li, John Thickstun, Ishaan Gulrajani, Percy Liang, Tatsunori Hashimoto

**3단계 구조**:

```
[Embedding] 토큰 → 가우시안 (단어 임베딩 중심)
[Diffusion] 임베딩 공간에서 노이즈 → denoising
[Rounding]  최종 임베딩 → softmax → 토큰
```

**학습 목표**:
$$\mathcal{L} = \mathbb{E}\left[\| x_0 - f_\theta(x_t, t) \|^2\right] + \mathcal{L}_{\text{rounding}}$$

매 스텝에서 **clean embedding $x_0$를 직접 예측**하도록 강제 (단순 노이즈 예측 아님).

**기여**: 6개 fine-grained 제어 생성 과제(POS, syntax tree, 길이, infilling 등)에서 **gradient-based control**로 SOTA.

**한계**: 200+ 스텝 필요, 작은 규모, AR보다 fluency 약함.

### 3.2 DiffuSeq (Shark-NLP, ICLR 2023)

- **arXiv**: 2210.08933
- **핵심**: **Partial noising** — source 시퀀스는 그대로, target만 노이징

$$q(z_t \mid z_0) = \mathcal{N}(z_t; \sqrt{\bar{\alpha}_t} z_0, (1-\bar{\alpha}_t) I) \quad \text{(target에만 적용)}$$

**과제**: Paraphrase (QQP 147K), Question Generation, Dialogue, Text Simplification.

**결과**: 6개 baseline (사전학습 LM 포함) 능가, **다양성 압도적**.

### 3.3 SSD-LM (CMU, ACL 2023)

- **arXiv**: 2210.17432
- **핵심 2가지**:
  - **Semi-autoregressive**: 블록 단위 생성 (전체 시퀀스가 아니라 블록씩)
  - **Simplex-based**: 학습된 latent가 아니라 **자연 어휘 simplex** 위에서 확산

**장점**: off-the-shelf classifier guidance를 별도 적응 없이 바로 사용 가능 → 모듈식 제어.

**결과**: 품질+다양성에서 GPT-2 능가, 디퓨전 baseline 압도.

### 3.4 CDCD (DeepMind)

- **arXiv**: 2211.15089
- **저자**: Sander Dieleman et al.
- 노이즈 임베딩에서 토큰을 **cross-entropy로 직접 예측**
- Time-warped 비균일 timestep 샘플링

### 3.5 Plaid (Stanford, NeurIPS 2023)

- **arXiv**: 2305.18619
- **저자**: Ishaan Gulrajani, Tatsunori Hashimoto
- **첫 번째 GPT-2 124M 능가 likelihood** 디퓨전 LM (Plaid 1B)
- 가중치+임베딩 양쪽에 well-posed 손실
- AR과 다른 compute-optimal 영역 발견

### 3.6 PLANNER (Apple, NeurIPS 2023)

- **arXiv**: 2306.02531
- **2-stage**: latent **planning** (의미 임베딩 확산) + AR **decoding**
- Coarse-to-fine 단락 생성

### 3.7 AR-Diffusion (Microsoft, NeurIPS 2023)

- **arXiv**: 2305.09515
- **Multi-level diffusion**: 토큰 수준 + 문장 수준 timestep
- **이른 위치의 토큰이 더 많은 denoising 단계** → AR 같은 좌→우 의존성을 non-AR로 모방
- 기존 텍스트 확산 대비 **100-600배 빠름**

---

## 4. 이산 디퓨전 LM

### 4.1 D3PM (Google, NeurIPS 2021) — 시조

- **arXiv**: 2107.03006
- **저자**: Jacob Austin, Daniel Johnson, Jonathan Ho, Daniel Tarlow, Rianne van den Berg
- **첫 원리적 이산 확산**

**4가지 transition matrix 변형**:

| 종류 | 정의 | Stationary | 특징 |
|------|------|-----------|------|
| **Uniform** | $[Q_t]_{ij} = (1-\beta_t) \delta_{ij} + \beta_t/K$ | 균일 1/K | 모든 토큰으로 전이 가능 |
| **Absorbing/Mask** | 각 토큰 → [MASK] (확률 $\beta_t$, mask는 흡수) | $\delta_{[MASK]}$ | **BERT MLM과 동일** |
| **Discretized Gaussian** | 연속 가우시안 모방 | - | 임베딩 가까운 토큰으로 |
| **Nearest-neighbor** | 임베딩 공간에서 가까운 이웃 | - | 의미적 정렬 |

**손실**: Hybrid = $\mathcal{L}_{VLB} + \lambda \cdot \mathcal{L}_{x_0}$

**결정적 발견**: **Absorbing-state(Mask) 변형이 텍스트에 압도적** → BERT/MLM과 디퓨전의 깊은 연결을 처음 시사.

### 4.2 Concrete Score Matching (Stanford, NeurIPS 2022)

- **arXiv**: 2211.00802
- **저자**: Chenlin Meng, Kristy Choi, Jiaming Song, Stefano Ermon

**Concrete score** 정의:

$$s_\theta(x)_y \approx \frac{p_{\text{data}}(y)}{p_{\text{data}}(x)}$$

연속의 Stein score를 이산으로 일반화 — Manhattan 거리에서의 **국소적 확률 변화율**. SEDD의 기반.

### 4.3 SEDD (Stanford, ICML 2024 **Best Paper**)

- **arXiv**: 2310.16834
- **저자**: Aaron Lou, Chenlin Meng, Stefano Ermon

#### Score Entropy Loss (핵심 기여)

기존 cross-entropy 대신 **Bregman divergence 기반** 손실:

$$\mathcal{L}_{SE} = \mathbb{E}_{x \sim p_{\text{data}}} \sum_{y \sim x} w_{xy} \left[ s_\theta(x)_y - \frac{p_{\text{data}}(y)}{p_{\text{data}}(x)} \log s_\theta(x)_y + h\left(\frac{p_{\text{data}}(y)}{p_{\text{data}}(x)}\right) \right]$$

- $w_{xy}$: rate matrix 항
- Bregman 기반 → **유한 손실 보장** (Fisher 기반은 발산 위험)
- ELBO 형태 → max-likelihood 학습 가능

#### Denoising Score Entropy (실용 버전)

데이터 비율 $p_{data}(y)/p_{data}(x)$를 직접 알 수 없으므로 **denoising 버전**으로 우회 — tractable + ELBO 동등.

#### 결과

- 기존 디퓨전 대비 **perplexity 25-75% 감소**
- **GPT-2 능가** (동일 compute)
- generative perplexity **6-8배** 향상
- **32배 적은 evaluation**으로 동등 품질

---

## 5. ★ 핵심 통찰: Masked Diffusion = 가중 MLM

### 정리

> **Absorbing-state masked diffusion의 ELBO는 BERT MLM 손실의 시간 가중 적분과 정확히 동등하다.**

연속 시간 마스킹 디퓨전:

$$\text{ELBO} = -\int_0^1 \frac{\alpha'(t)}{1 - \alpha(t)} \cdot \mathbb{E}_{x_0, x_t}\left[\sum_{i: x_t^i = [\text{MASK}]} \log p_\theta(x_0^i \mid x_t)\right] dt$$

여기서 $\alpha(t)$는 마스킹 스케줄.

**의미**:
- 적분 안의 항은 **마스킹 비율 $(1-\alpha(t))$에서의 MLM cross-entropy**
- 즉 ELBO = 다양한 마스킹 비율의 MLM 손실 가중합

### 함의

> **모든 BERT 스타일 인코더는 "재가중 한 번"이면 generator가 된다.**

이 정리는 MDLM (Sahoo et al.)과 MD4 (Shi et al.)가 동시에 증명했다.

### MDLM (Cornell, NeurIPS 2024)

- **arXiv**: 2406.07524
- **저자**: Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, Volodymyr Kuleshov
- **결과**: SEDD 대비 perplexity bound **17% 개선** (33B 토큰), AR 대비 **14% 이내**
- **SUBS** (substitution) 파라미터화, semi-AR 샘플링 지원
- **ByteDance Seed Diffusion + Nvidia Genmol**의 기반

### MD4 (DeepMind, NeurIPS 2024)

- **arXiv**: 2406.04329
- 연속 시간 ELBO = cross-entropy 손실의 단순 가중 적분
- **GenMD4**: state-dependent 마스킹 스케줄
- GPT-2 스케일에서 **5개 zero-shot LM 벤치마크 중 4개에서 SOTA**

### GIDD (ETH, ICML 2025)

- **arXiv**: 2503.04482
- **저자**: Dimitri von Rütte et al.
- 마스킹 디퓨전을 **모든 선형 보간**으로 일반화:

$$x_t = \alpha(t) x_0 + (1-\alpha(t)) \pi(t)$$

여기서 $\pi(t)$는 임의의 mixing 분포 (uniform, mask, hybrid).

#### 핵심 발견: Self-Correction

**Mask only**: 마스크는 흡수 상태 → 한번 마스크 되면 못 복귀. 모델이 **자기 실수 수정 불가**.

**Hybrid (mask + uniform noise)**: 모델이 **자기 실수를 식별하고 수정** 가능 → Gemma 2 9B에서 generative PPL **최대 55% 개선**.

---

## 6. LLaDA 패밀리 — 8B에서 100B까지

### 6.1 LLaDA 8B (Renmin Univ + Ant Group, ICLR 2025)

- **arXiv**: 2502.09992
- **저자**: Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, Chongxuan Li
- **Code**: [github.com/ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)

**핵심**:
- **8B 마스킹 디퓨전 LLM**, **from scratch 학습** (AR 변환 X)
- 학습: **2.3T 토큰, 0.13M H800 GPU 시간**
- **Forward**: 랜덤 마스킹
- **Reverse**: 모든 마스킹 토큰을 **동시에 예측**

**손실**:

$$\mathcal{L} = -\mathbb{E}\left[\frac{1}{|M|} \sum_{i \in M} \log p_\theta(x_0^i \mid x_t)\right]$$

(마스크된 토큰만 cross-entropy)

**결과**:

| 벤치마크 | LLaMA2 7B | LLaMA3 8B | **LLaDA 8B** |
|---------|----------|----------|-----------|
| MMLU | 45.3 | 66.6 | **65.9** |
| GSM8K | 14.6 | 56.4 | **70.7** |
| HumanEval | 14.0 | 33.5 | **33.5** |

- **LLaMA2 거의 모든 과제에서 능가**
- **LLaMA3와 경쟁력**
- **Reversal curse 극복**: GPT-4o를 역방향 시 완성에서 능가

### 6.2 LLaDA 1.5 (2025.05)

- **arXiv**: 2505.19223
- **VRPO** (Variance-Reduced Preference Optimization)
- ELBO 추정자의 분산 분석 → bias/variance 경계 도출
- 최적 Monte Carlo budget + antithetic sampling
- **GSM8K +4.7, HumanEval +3.0, IFEval +4.0**

### 6.3 LLaDA-V (2025.05)

- **arXiv**: 2505.16933
- **순수 디퓨전 기반 멀티모달** LLM
- Vision encoder + MLP + LLaDA
- LLaMA3-V와 경쟁

### 6.4 LLaDA 2.0 (2025.12)

- **arXiv**: 2512.15745
- **100B 파라미터로 스케일링** — **첫 100B 디퓨전 LM**
- 체계적인 **AR → dLLM 변환** (from scratch X)
- 3-phase block-level WSD 학습 (warm-up → stable → decay)
- **변형**: LLaDA2.0-mini (16B MoE), **LLaDA2.0-flash (100B MoE, 6.1B activated)**

---

## 7. 상용화 시대 — Mercury, Gemini Diffusion, Seed Diffusion

### 7.1 Mercury (Inception Labs, 2025.06)

- **arXiv**: 2506.17298
- **창업자**: Stefano Ermon (SEDD 저자), Aditya Grover, Volodymyr Kuleshov (MDLM 저자)
- **첫 상용 디퓨전 LLM**

| 모델 | 속도 (H100) | HumanEval | MBPP |
|------|-----------|----------|------|
| **Mercury Coder Mini** | **1,109 tok/s** | 88.0 | 77.1 |
| **Mercury Coder Small** | **737 tok/s** | 90.0 | ~75 |

- 속도 최적화 frontier 모델 대비 **10배 빠름**
- 32K native, 128K 확장 가능
- AWS Bedrock, Azure AI Foundry 제공

### 7.2 Gemini Diffusion (Google DeepMind, 2025.05)

- Google I/O 2025 발표
- **1,479 tok/s 평균** (0.84s startup)
- Gemini 2.0 Flash Lite와 거의 동등 (HumanEval 89.6 vs 90.2)

### 7.3 Seed Diffusion (ByteDance + Tsinghua AIR, 2025.08)

- **arXiv**: 2508.02193
- **2,146 tok/s on H20** — **가장 빠른 dLLM**
- 코드 특화 이산 확산
- MDLM 기반

### 속도 비교 한눈에

| 모델 | tok/s | 비교 |
|------|-------|-----|
| **Seed Diffusion** | **2,146** | dLLM 1위 |
| Gemini Diffusion | 1,479 | Google |
| Mercury Coder Mini | 1,109 | 첫 상용 |
| Mercury Coder Small | 737 | 더 큰 변형 |
| AR baseline | 50-200 | 표준 LLM |

---

## 8. Flow Matching for Text

### 8.1 Discrete Flow Matching (Meta FAIR, NeurIPS 2024)

- **arXiv**: 2407.15595
- **저자**: Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky T.Q. Chen, Gabriel Synnaeve, Yossi Adi, Yaron Lipman

source ↔ target 간 **모든 확률 경로** 일반화. **이산 Continuity Equation**:

$$\frac{\partial p_t}{\partial t} = -\nabla \cdot (p_t u_t)$$

AR과 이산 flow의 격차 메움.

### 8.2 FS-DFM (Apple + Ohio State, ICLR 2026)

- **arXiv**: 2509.20624
- **Few-Step Discrete Flow Matching**
- 샘플링 단계 수를 **명시적 파라미터**로 학습
- **8-step이 1024-step baseline과 perplexity 동등** → **128배 빠름**

### 8.3 Statistical Flow Matching (NeurIPS 2024)

- **arXiv**: 2405.16441
- 범주 분포의 **Riemannian 다양체**에서 flow
- Fisher information 메트릭의 geodesic
- Exact likelihood (variational bound 아님)

---

## 9. Reasoning + RL — 디퓨전이 추론할 수 있는가?

### 9.1 DoT (Diffusion of Thoughts, NeurIPS 2024)

- **arXiv**: 2402.07754
- 디퓨전에 CoT 통합
- **작은 DoT-SEDD가 GPT-2-medium + CoT를 ~10% 능가** (수학)

### 9.2 d1 (UCLA, 2025.04)

- **arXiv**: 2504.12216
- SFT + **diffu-GRPO** (마스킹 dLLM 첫 policy gradient RL)
- d1-LLaDA에서 **자기 검증/수정** "aha 모먼트" 관찰
- dLLM 중 GSM8K + MATH500 SOTA

### 9.3 DCoLT (Diffusion Chain of Lateral Thought, 2025.05)

- **arXiv**: 2505.10446
- 매 디퓨전 스텝 = latent thinking action
- 전체 trajectory에 outcome-based RL
- **DCoLT-LLaDA 성능 향상**:
  - GSM8K +9.8
  - MATH +5.7
  - MBPP +11.4
  - **HumanEval +19.5**
- Plackett-Luce 랭킹 기반 unmasking 정책

---

## 10. AR vs Diffusion 종합 비교

| 차원 | Diffusion | Autoregressive |
|------|----------|---------------|
| **속도** | 병렬, 5-10배 빠름 (Seed Diffusion 2146 tok/s) | 순차, KV-cache로도 한계 |
| **품질 (PPL)** | MDLM가 AR 대비 14% 이내, SEDD가 GPT-2 능가 | 오랫동안 SOTA |
| **학습 비용** | **2-5배 더 많은 데이터 필요** (Quokka scaling laws) | 토큰당 효율적 |
| **제어성** | 우수 (gradient guidance, classifier guidance, infilling) | 약함 (prompt/RL 필요) |
| **양방향성** | YES — full bidirectional | NO — causal only |
| **Reversal curse** | 완화 (LLaDA가 GPT-4o 능가) | 심각 |
| **Test-time compute scaling** | YES — step 수 증가로 품질↑ | 제한적 (CoT 길이만) |
| **Infilling/Editing** | Native | 특수 토큰 필요 |
| **Long-context** | 미성숙 | 성숙 (RoPE 등) |

---

## 11. 디퓨전 ↔ AR 변환 — 가장 실용적인 시나리오

### DiffuLLaMA (HKUNLP, ICLR 2025)

- **arXiv**: 2410.17891
- **AR 모델을 디퓨전으로 변환**: GPT2 127M → LLaMA 7B
- **<200B 토큰 continual pretraining**으로 변환 완료
- AR-디퓨전 학습 목표의 깊은 연결 입증

### Block Diffusion / BD3-LMs (Cornell, ICLR 2025 Oral)

- **arXiv**: 2503.09573
- **AR과 디퓨전 사이의 보간**
- 시퀀스를 블록으로 분해 → 블록 간 AR + 블록 내 디퓨전
- **블록 크기가 quality/efficiency 노브**
- KV caching + 병렬 샘플링 가능
- 임의 길이 생성 가능

### LLaDA 2.0의 100B 변환

LLaDA 2.0이 **AR → dLLM 변환 방식으로 100B 달성** — from scratch 학습보다 훨씬 비용 효율적.

---

## 12. 샘플링 기법 정리

| 기법 | 설명 | 사용처 |
|------|------|------|
| **Standard reverse** | 순차적 q-posterior 샘플링 | Baseline |
| **Ancestral sampling** | $x_{t-1} \sim p_\theta(\cdot \mid x_t)$ | DDPM 스타일 |
| **τ-leaping** | 여러 위치 동시 업데이트 | SEDD, 빠른 이산 |
| **Predictor-corrector** | 폐쇄형 predictor + Langevin/Gibbs | SEDD |
| **DDIM-like deterministic** | Non-Markovian implicit | DNDM |
| **Confidence parallel** | 확신도 높은 토큰만 commit | LLaDA, Mercury |
| **Semi-AR** | 블록 단위 | MDLM, BD3-LM |
| **Consistency / few-step** | 다단계 → 소단계 distillation | CDLM (3.6-14.5x), FS-DFM (128x) |

---

## 13. 한계와 도전

### 13.1 학습 비용 (Quokka Scaling Laws)

- **arXiv**: 2510.03280
- DLM은 같은 FLOP에서 **2-5배 더 많은 데이터** 필요
- Corruption이 더 많은 샘플 요구
- Uniform diffusion이 masked보다 더 좋게 스케일링

### 13.2 Long-context 미성숙

AR의 RoPE처럼 검증된 long-context 기법 부재. 대부분 32K 이하.

### 13.3 평가 어려움

표준 perplexity 계산 어려움 → ELBO bound, generative perplexity 등 새 지표 필요.

### 13.4 인프라 부재

기존 토큰 기반 인프라(검색, 캐싱, RAG)와 호환 어려움.

### 13.5 Mask only의 self-correction 부재

Mask는 흡수 상태 → GIDD의 hybrid noise가 해결.

---

## 14. 2026년 전망

### 단기 (1년 내)

1. **상용 dLLM 보편화**: Mercury, Gemini Diffusion, Seed Diffusion 외 추가
2. **추론 dLLM 표준화**: d1, DCoLT 계열 → reasoning에서도 AR과 경쟁
3. **AR → dLLM 변환** 표준 레시피 등장 (LLaDA 2.0 방식)

### 중기 (2-3년)

4. **하이브리드가 표준**: Block Diffusion 같은 AR+Diffusion 구조
5. **멀티모달 통합**: LLaDA-V, LLaDA2.0-Uni 같은 통합 디퓨전 모델
6. **Test-time compute** 표준화: step 수로 quality 조정

### 장기 (5년+)

7. **dLLM이 메인스트림**: 속도/제어성/양방향성의 결합으로
8. **새 패러다임**: Continuous + Discrete + AR 하이브리드

---

## 15. 한눈에 보는 핵심 모델 비교

| 모델 | 종류 | 규모 | 특징 |
|------|------|------|------|
| Diffusion-LM (2022) | 연속 | <1B | 최초, gradient control |
| D3PM (2021) | 이산 | <1B | 최초 원리적 이산 |
| SEDD (2024) | 이산 | <1B | Score entropy, GPT-2 능가 |
| MDLM (2024) | 마스킹 | <1B | MLM 정리 증명 |
| **LLaDA 8B** | 마스킹 | **8B** | LLaMA3와 경쟁 |
| **Mercury** | 마스킹 | 비공개 | 첫 상용, 1109 tok/s |
| **Seed Diffusion** | 마스킹 | 비공개 | 2146 tok/s 최고 속도 |
| GIDD | Hybrid | <1B | Self-correction |
| **LLaDA 2.0** | MoE 마스킹 | **100B** | 첫 100B dLLM |
| FS-DFM | Flow | <1B | 8-step 1024-step 동등 |

---

## 16. 핵심 한 줄 요약

> **"Mask 디퓨전 = 가중 BERT MLM"이라는 정리가 모든 것을 바꿨다. BERT가 사실은 generator였고, 디퓨전이 그것을 풀어준 것이다.**
>
> **Mercury가 1109 tok/s를 보여준 순간, 'AR이 정답이다'라는 가정도 끝났다.**

---

## 17. 관련 블로그 포스트

- [연속 의미 공간 언어 모델 서베이](continuous-vector-language-models.md) — 더 넓은 맥락
- [SONAR 리뷰](sonar-review.md) — 다국어 임베딩 공간
- [DPO 리뷰](dpo-review.md) — RL 단순화
- [Search-R1 리뷰](search-r1-review.md) — RL + 검색
- [강화학습 입문](reinforcement-learning-beginner-guide.md) — RL 기초

---

## 참고 자료 (50+)

### 시조 논문
- [D3PM (arXiv:2107.03006)](https://arxiv.org/abs/2107.03006) — 첫 이산 확산
- [Multinomial Diffusion (arXiv:2102.05379)](https://arxiv.org/abs/2102.05379)
- [Diffusion-LM (arXiv:2205.14217)](https://arxiv.org/abs/2205.14217) | [코드](https://github.com/XiangLi1999/Diffusion-LM)
- [Concrete Score Matching (arXiv:2211.00802)](https://arxiv.org/abs/2211.00802)

### 연속 디퓨전
- [DiffuSeq (arXiv:2210.08933)](https://arxiv.org/abs/2210.08933) | [코드](https://github.com/Shark-NLP/DiffuSeq)
- [SSD-LM (arXiv:2210.17432)](https://arxiv.org/abs/2210.17432) | [코드](https://github.com/xhan77/ssd-lm)
- [CDCD (arXiv:2211.15089)](https://arxiv.org/abs/2211.15089)
- [GENIE (arXiv:2212.11685)](https://arxiv.org/abs/2212.11685)
- [Plaid (arXiv:2305.18619)](https://arxiv.org/abs/2305.18619)
- [PLANNER (arXiv:2306.02531)](https://arxiv.org/abs/2306.02531) | [코드](https://github.com/apple/ml-planner)
- [AR-Diffusion (arXiv:2305.09515)](https://arxiv.org/abs/2305.09515)

### 이산/마스킹 디퓨전 (모던)
- [SEDD (arXiv:2310.16834)](https://arxiv.org/abs/2310.16834) | [코드](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)
- [MDLM (arXiv:2406.07524)](https://arxiv.org/abs/2406.07524) | [코드](https://github.com/kuleshov-group/mdlm)
- [MD4 (arXiv:2406.04329)](https://arxiv.org/abs/2406.04329) | [코드](https://github.com/google-deepmind/md4)
- [GIDD (arXiv:2503.04482)](https://arxiv.org/abs/2503.04482) | [코드](https://github.com/dvruette/gidd)

### LLaDA 패밀리
- [LLaDA (arXiv:2502.09992)](https://arxiv.org/abs/2502.09992) | [코드](https://github.com/ML-GSAI/LLaDA)
- [LLaDA 1.5 (arXiv:2505.19223)](https://arxiv.org/abs/2505.19223)
- [LLaDA-V (arXiv:2505.16933)](https://arxiv.org/abs/2505.16933)
- [LLaDA 2.0 (arXiv:2512.15745)](https://arxiv.org/abs/2512.15745)

### 변환 / 하이브리드
- [DiffuLLaMA (arXiv:2410.17891)](https://arxiv.org/abs/2410.17891) | [코드](https://github.com/HKUNLP/DiffuLLaMA)
- [Block Diffusion (arXiv:2503.09573)](https://arxiv.org/abs/2503.09573) | [코드](https://github.com/kuleshov-group/bd3lms)

### 코드 dLLM
- [Dream 7B (arXiv:2508.15487)](https://arxiv.org/abs/2508.15487) | [코드](https://github.com/DreamLM/Dream)
- [Dream-Coder (arXiv:2509.01142)](https://arxiv.org/abs/2509.01142)
- [DiffuCoder (arXiv:2506.20639)](https://arxiv.org/abs/2506.20639) | [코드](https://github.com/apple/ml-diffucoder)

### 상용
- [Mercury (arXiv:2506.17298)](https://arxiv.org/abs/2506.17298) | [Inception Labs](https://www.inceptionlabs.ai/)
- [Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/)
- [Seed Diffusion (arXiv:2508.02193)](https://arxiv.org/abs/2508.02193)

### Flow Matching
- [Discrete Flow Matching (arXiv:2407.15595)](https://arxiv.org/abs/2407.15595)
- [FS-DFM (arXiv:2509.20624)](https://arxiv.org/abs/2509.20624)
- [Statistical Flow Matching (arXiv:2405.16441)](https://arxiv.org/abs/2405.16441)

### Reasoning + RL
- [DoT (arXiv:2402.07754)](https://arxiv.org/abs/2402.07754) | [코드](https://github.com/HKUNLP/diffusion-of-thoughts)
- [d1 (arXiv:2504.12216)](https://arxiv.org/abs/2504.12216) | [코드](https://github.com/dllm-reasoning/d1)
- [DCoLT (arXiv:2505.10446)](https://arxiv.org/abs/2505.10446)

### 서베이
- [A Survey on Diffusion Language Models (arXiv:2508.10875)](https://arxiv.org/abs/2508.10875)
- [Quokka Scaling Laws (arXiv:2510.03280)](https://arxiv.org/abs/2510.03280)
