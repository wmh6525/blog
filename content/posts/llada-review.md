---
title: "[논문 리뷰] LLaDA — 8B 마스킹 디퓨전 LLM, 데이터·하이퍼파라미터·학습/추론 완전 해부"
date: 2026-05-15
tags: ["논문리뷰", "Diffusion", "LLM", "LLaDA", "마스킹디퓨전"]
categories: ["ML/AI"]
summary: "LLaDA 8B 논문 심화 리뷰. from scratch 학습으로 LLaMA3와 경쟁한 마스킹 디퓨전 LLM. 2.3T 토큰 학습 데이터, WSD 학습 스케줄, 마스킹 손실 수식, 역확산 샘플링 절차, LLaDA 1.5(VRPO)와 2.0(100B MoE)까지 학습·추론을 논문 기반으로 상세 해부."
math: true
toc: true
draft: false
---

## 논문 정보

- **제목**: Large Language Diffusion Models
- **저자**: Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, Chongxuan Li
- **소속**: Renmin University of China (Gaoling School of AI, "ML-GSAI") + Ant Group
- **arXiv**: 2502.09992 (2025.02, ICML 2025)
- **코드**: [github.com/ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)

> 이 글은 [Diffusion LM 서베이](diffusion-language-models-survey.md)의 심화 편이다. LLaDA를 **학습 데이터·하이퍼파라미터·학습/추론 절차** 중심으로 논문 그대로 해부한다.

---

## 1. 한 줄 요약

> **8B 마스킹 디퓨전 LLM을 from scratch로 학습하여, 자회귀 없이 LLaMA3 8B와 경쟁하고 reversal curse를 극복했다.**

---

## 2. 아키텍처 — LLaMA3와 무엇이 다른가

LLaDA의 마스크 예측기는 **bidirectional Transformer 인코더** (causal mask 없음). 표준 decoder-only LLM에서 causal mask만 제거한 것.

| 항목 | LLaDA 8B | LLaMA3 8B |
|------|---------|-----------|
| 레이어 | 32 | 32 |
| 히든 차원 | 4096 | 4096 |
| 어텐션 헤드 | 32 | 32 |
| KV 헤드 | **32 (vanilla MHA)** | 8 (GQA) |
| FFN 히든 | **12288** (축소) | 14336 |
| 어휘 크기 | 126,464 | 128,256 |
| 최대 시퀀스 | 4096 | 8192 |
| 위치 임베딩 | RoPE | RoPE |
| 정규화 | RMSNorm | RMSNorm |

**핵심 차이 3가지**:

1. **No causal mask** — 전체 양방향 어텐션 (정의적 변경)
2. **GQA 대신 vanilla MHA** — LLaDA는 **KV 캐싱과 호환 불가** (양방향이라 좌→우 캐시 재사용 불가) → GQA의 효율 이점이 없으므로 단순한 MHA 사용
3. **FFN 차원 축소** (14336 → 12288) — MHA 추가 파라미터를 상쇄해 총 ~8B 유지

마스크 토큰 ID = **126336**.

---

## 3. 학습 데이터

### 사전학습 코퍼스

| 항목 | 값 |
|------|-----|
| 총 토큰 | **2.3조 (2.3T)** |
| 소스 | 웹 코퍼스 (수동 + LLM 기반 품질 필터링) + 코드·수학·다국어 고품질 데이터 |
| 데이터 혼합 결정 | 축소판 AR 모델로 혼합 비율 가이드 |
| 시퀀스 길이 | 4096 고정 |
| 가변 길이 처리 | **1%의 데이터를 [1, 4096] 균등 랜덤 길이로 샘플링** → 추론 시 가변 길이 생성 가능 |

### SFT 데이터

- **450만 (4.5M) 프롬프트-응답 쌍**
- 도메인: 코드, 수학, instruction-following, 다중 턴 대화

### 토크나이저

자체 BPE 토크나이저 (어휘 126,464), 마스크 토큰 ID 126336.

---

## 4. 학습 절차

### 4.1 Forward (마스킹) 프로세스

각 토큰 $i$를 독립적으로:

$$q_{t|0}(x_t^i | x_0^i) = \begin{cases} 1-t & x_t^i = x_0^i \\ t & x_t^i = [\text{MASK}] \end{cases}$$

- **마스킹 비율 $t \sim \text{Uniform}(0, 1]$** — 시퀀스마다 샘플링
- $t=0$ → 완전 비마스킹, $t=1$ → 완전 마스킹
- 실제 코드에서는 마스킹 확률이 $(1-\epsilon) t + \epsilon$, $\epsilon = 10^{-3}$ (0으로 나눔 방지)

### 4.2 손실 함수 (사전학습 — ELBO)

$$\mathcal{L}(\theta) = -\mathbb{E}_{t, x_0, x_t}\left[\frac{1}{t} \sum_i \mathbb{1}[x_t^i = M] \cdot \log p_\theta(x_0^i | x_t)\right]$$

- 마스킹된 토큰에 대해서만 cross-entropy
- **마스킹 비율로 정규화** ($1/t$)
- 이것은 **음의 로그 가능도의 변분 상한** — $-\mathbb{E}[\log p_\theta(x_0)] \leq \mathcal{L}(\theta)$

### 4.3 SFT 손실

응답 토큰만 마스킹, 프롬프트 토큰은 노이즈 없음:

$$-\mathbb{E}_{t, p_0, r_0, r_t}\left[\frac{1}{t} \sum_i \mathbb{1}[r_t^i = M] \cdot \log p_\theta(r_0^i | p_0, r_t)\right]$$

### 4.4 사전학습 하이퍼파라미터

| 항목 | 값 |
|------|-----|
| **컴퓨트** | **0.13M H800 GPU 시간** |
| 옵티마이저 | AdamW, weight decay 0.1 |
| LR 스케줄 | **Warmup-Stable-Decay (WSD)** |
| - Warmup | 0 → 4e-4, 2000 iter |
| - Stable | 4e-4 유지 (첫 1.2T 토큰) |
| - Decay 1 | 1.2T 토큰에서 1e-4로 하락 (다음 0.8T) |
| - Decay 2 | 마지막 0.3T 토큰, 1e-4 → 1e-5 선형 |
| 글로벌 배치 크기 | 1280 |
| 시퀀스 길이 | 4096 고정 |

**흥미로운 사실**: 1.2T 토큰에서 학습 크래시 발생 → LR을 4e-4에서 1e-4로 낮춰 해결. 그래서 스케줄에 그 "꺾임"이 있다.

### 4.5 SFT 하이퍼파라미터

| 항목 | 값 |
|------|-----|
| 에폭 | 3 |
| 옵티마이저 | AdamW, weight decay 0.1 |
| 글로벌 배치 크기 | 256 |
| LR | 0 → 2.5e-5 warmup (50 iter), 마지막 10%는 2.5e-6으로 선형 감쇠 |

---

## 5. 추론 / 샘플링 절차

### 5.1 Reverse 프로세스

완전 마스킹된 답변($t=1$)에서 시작 → $N$ 스텝에 걸쳐 점진적 비마스킹 → $t=0$. 타임스텝 $t_k = 1 - k/N$.

### 5.2 각 스텝 ($t \to s = t - 1/N$)

```
1. Forward pass: 모든 마스킹 토큰 예측
   x̂_0 = argmax p_θ(·|x_t)   (temperature=0이면 greedy)
2. Remask: 방금 예측한 토큰의 일부(s/t 비율)를 다시 마스킹
   → 남은 마스크 수가 forward 스케줄과 일치하도록
```

### 5.3 Remasking 전략 3가지

| 전략 | 방식 |
|------|------|
| **Random remasking** | s/t 비율을 균등 랜덤하게 다시 마스킹 |
| **Low-confidence remasking** (기본값) | 확신도 높은 예측은 유지, 낮은 것만 다시 마스킹 |
| **Semi-autoregressive (블록)** | 시퀀스를 블록으로 나눠 블록 단위 좌→우 생성, 각 블록 내에서 디퓨전 |

### 5.4 `generate.py` 기본값

```python
steps=128            # 샘플링 스텝
gen_length=128       # 생성 길이
block_length=128     # 블록 크기
temperature=0.0
cfg_scale=0.0        # Classifier-Free Guidance
remasking='low_confidence'
```

**품질/속도 트레이드오프**: 스텝이 많을수록 품질↑. **최적 품질은 샘플링 스텝 ≈ 응답 길이**일 때 (스텝당 토큰 하나씩 "확정").

### 5.5 Classifier-Free Guidance (선택)

논문 본 실험에서는 미사용 (AR 베이스라인과 공정 비교 위해). 구현:

$$\text{logits} = \text{un\_logits} + (\text{cfg\_scale} + 1) \cdot (\text{logits} - \text{un\_logits})$$

---

## 6. 벤치마크

### 6.1 Base 모델 (LLaDA 8B Base vs LLaMA)

| 벤치마크 | LLaDA 8B | LLaMA3 8B | LLaMA2 7B |
|---------|---------|-----------|-----------|
| MMLU | 65.9 | 65.4 | 45.9 |
| **GSM8K** | **70.3** | 48.7 | 13.1 |
| **MATH** | **31.4** | 16.0 | 4.3 |
| HumanEval | 35.4 | 34.8 | 12.8 |
| HumanEval-FIM | 73.8 | 73.3 | 26.9 |
| **CMMLU** | **69.9** | 50.7 | 32.5 |
| **C-Eval** | **70.5** | 51.7 | 34.0 |
| BBH | 49.7 | 62.1 | 39.4 |

- LLaMA2 7B를 거의 모든 과제에서 능가
- LLaMA3 8B와 전반적으로 동등, **수학·중국어에서 압도**

### 6.2 Reversal Curse 실험

496개 중국 시 문장 쌍, 파인튜닝 없음.

| 모델 | 정방향 | 역방향 |
|------|-------|-------|
| GPT-4o | 82.7% | 34.3% |
| Qwen2.5-7B | 75.9% | 38.0% |
| **LLaDA-8B** | 51.8% | **45.6%** |

**핵심**: GPT-4o, Qwen2.5는 역방향에서 급락(reversal curse). LLaDA는 **거의 대칭적인 격차**(51.8 vs 45.6)를 보이고 역방향에서 GPT-4o를 능가. 원인 — 양방향 마스킹 학습에는 좌→우 방향 편향이 없다.

---

## 7. LLaDA 1.5 — VRPO (Variance-Reduced Preference Optimization)

- **arXiv**: 2505.19223

### 문제

디퓨전 LM에 DPO 스타일 선호 정렬을 하려면 다루기 힘든 로그 가능도를 **ELBO**로 대체해야 한다. ELBO는 Monte Carlo 추정 → **분산이 매우 크다** → 선호 그래디언트가 오염.

### 핵심 정리 (Theorem 1)

손실의 bias와 variance 모두 ELBO 기반 선호 점수 $\hat{s}_\theta$의 분산으로 경계:

$$\mathbb{E}[|\ell - \hat{\ell}|] \leq \sqrt{\mathbb{V}[\hat{s}_\theta]}, \qquad \mathbb{V}[\hat{\ell}] \leq 4 \cdot \mathbb{E}[\mathbb{V}[\hat{s}_\theta]]$$

### VRPO 3가지 기법 (모두 불편 추정 유지)

| 기법 | 방법 | 추가 FLOPs |
|------|------|----------|
| **샘플링 예산 증가** | $n = 8$ (기본) | ~8배 (전체의 <0.5%) |
| **최적 예산 배분** | 분산 최소화: $n_t = n$, $n_{yt} = 1$ (전 예산을 서로 다른 타임스텝에) | 0 |
| **Antithetic 샘플링** | $\pi_\theta$와 $\pi_{ref}$에 **같은 타임스텝·마스킹 시퀀스** 사용 → ELBO 추정의 차이에서 분산 상쇄 | 0 |

**Ablation**: antithetic 제거 시 점수 추정 분산이 ~1.0 → **2183.7**로 폭발.

### 결과 (LLaDA 1.5 vs LLaDA 8B Instruct)

| 벤치마크 | Instruct | 1.5 | Δ |
|---------|---------|-----|---|
| GSM8K | 78.6 | 83.3 | +4.7 |
| GPQA | 33.3 | 36.9 | +3.6 |
| HumanEval | 49.4 | 52.4 | +3.0 |
| IFEval | 62.2 | 66.2 | +4.0 |
| Arena-Hard | 10.0 | 14.3 | +4.3 |

---

## 8. LLaDA 2.0 — 100B로의 스케일링

- **arXiv**: 2512.15745

### 핵심 아이디어

from scratch 학습하지 않고 **기존 AR 모델을 dLLM으로 변환** (지식 상속). 베이스: Ling-mini-2.0, Ling-flash-2.0.

AR 모델을 **블록 크기 1의 Block Diffusion LM**으로 간주 → 변환이 매끄러운 점진적 적응.

### 모델 변형 (둘 다 MoE)

- **LLaDA2.0-mini**: 16B 총 파라미터
- **LLaDA2.0-flash**: 100B 총 파라미터 (6.1B activated) — **첫 100B 디퓨전 LM**

### Block-level WSD 3단계 학습

```
1. Warmup — 블록 크기 점진 증가: 1 → 4 → 32 → 64 → 4096
2. Stable — 블록 크기 4096 고정 (full-sequence MDLM), 대규모 학습
3. Decay — 블록 크기 다시 축소: 4096 → 32 (전역 → 국소 컨디셔닝)
```

### 사후 학습

- **SFT**: 보완적(antithetical) 마스킹 — 소스당 두 개의 보완 마스킹 인스턴스
- **CAP (Confidence-Aware Parallel) 학습**: 보조 손실로 ~2.1배 추론 가속
- **DPO**: 150만 선호 쌍, β=0.1

### 추론

temperature 0.0, 블록 크기 32, 신뢰도 임계값 0.95. LLaDA2.0-flash-CAP 처리량 ~535 tok/s (AR 베이스라인 237-256 tok/s).

---

## 9. LLaDA-MoE (중간 모델)

- **arXiv**: 2509.24389
- **첫 from-scratch MoE 디퓨전 LM**
- LLaDA-MoE-7B-A1B: 7B 총 파라미터, **~1.4B active**
- ~20T 토큰 학습
- LLaDA, LLaDA 1.5, Dream을 모두 능가

---

## 10. 한계

1. **생성 길이가 고정 하이퍼파라미터** — 적응적 길이 생성은 future work
2. **KV 캐시 사용 불가** — 양방향 어텐션의 본질적 제약
3. **AR보다 느림** — 최적 품질은 샘플링 스텝 ≈ 응답 길이 필요
4. 일부 과제(BBH, MBPP)에서 여전히 AR에 뒤짐
5. 논문 시점 RL 정렬 없음 → LLaDA 1.5(VRPO), 2.0(DPO)이 이후 보완

---

## 11. 핵심 요약

| 질문 | 답 |
|------|-----|
| **무엇** | 8B 마스킹 디퓨전 LLM, from scratch |
| **데이터** | 2.3T 사전학습 + 4.5M SFT 쌍 |
| **학습** | WSD 스케줄, 0.13M H800 시간, 마스킹 비율 $t \sim U(0,1]$ |
| **손실** | 마스킹 토큰만 CE, $1/t$ 정규화 (ELBO) |
| **추론** | $t=1$ 완전 마스킹 → N스텝 역확산, low-confidence remasking |
| **결과** | LLaMA3와 경쟁, reversal curse 극복 |

---

## 12. 관련 블로그 포스트

- [Diffusion LM 서베이](diffusion-language-models-survey.md) — 전체 맥락
- [SEDD 심화 리뷰](sedd-deep-review.md)
- [MDLM 심화 리뷰](mdlm-deep-review.md)
- [Mercury 심화 리뷰](mercury-deep-review.md)

---

## 참고 자료

- [LLaDA (arXiv:2502.09992)](https://arxiv.org/abs/2502.09992)
- [LLaDA 1.5 (arXiv:2505.19223)](https://arxiv.org/abs/2505.19223)
- [LLaDA-V (arXiv:2505.16933)](https://arxiv.org/abs/2505.16933)
- [LLaDA 2.0 (arXiv:2512.15745)](https://arxiv.org/abs/2512.15745)
- [코드: github.com/ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)
