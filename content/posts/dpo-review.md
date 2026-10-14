---
title: "[논문 리뷰] DPO — RLHF에서 강화학습을 제거하다, '당신의 LM은 몰래 보상모델이다'"
date: 2026-04-22
tags: ["논문리뷰", "DPO", "RLHF", "LLM", "정렬"]
categories: ["ML/AI"]
summary: "DPO(Direct Preference Optimization) 논문 상세 리뷰. RLHF의 3단계(SFT→RM→PPO)를 2단계로 단순화한 NeurIPS 2023 Outstanding Paper. KL 제약 보상 최대화의 닫힌 해를 활용해 강화학습 없이 선호도 학습을 단순 분류 문제로 치환한다."
math: true
toc: true
draft: false
---

<!-- LEARNING-START -->
## 처음 읽는 분을 위한 안내

**학습 위치:** [좋은 답을 배우고 기존 능력 지키기]({{< relref "learn" >}}#learning) · 2/9

같은 질문에 대한 두 답 중 사람이 어느 쪽을 더 선호하는지 알려 준다고 하자. 이 비교로 모델이 선호되는 답을 더 자주 만들도록 조정하는 방법이다. 처음에는 답변 쌍을 어떻게 쓰는지 이해하고, 수식 유도는 나중에 따라가자.

**먼저 알아둘 말**

- [선호 학습]({{< relref "ai-glossary" >}}#preference): 같은 요청의 여러 답 중 어떤 답을 더 선호하는지 비교한 데이터로 모델을 조정하는 학습이다.
- [정책·롤아웃]({{< relref "ai-glossary" >}}#policy): 정책은 주어진 상황에서 행동을 선택하는 방식이다. 롤아웃은 그 정책으로 행동하고 결과를 받은 과정을 기록한 것이다.
- [KL divergence]({{< relref "ai-glossary" >}}#kl): 두 확률 분포의 차이를 측정하는 양. 비교 방향을 바꾸면 값과 학습 효과도 달라질 수 있다.

**이번 글의 목표:** 두 답의 선호 비교로 모델을 조정하는 원리 이해하기.

**이 학습 경로의 앞 글:** [\[입문\] 강화학습 완전 정복 — 게임 배우는 아이부터 PPO·DPO까지 그림으로 이해하기]({{< relref "posts/reinforcement-learning-beginner-guide" >}}).

**본문에서 먼저 볼 순서:** **1. 문제: RLHF는 왜 복잡하고 불안정한가?** → **2. 핵심 통찰: "당신의 LM은 이미 보상모델이다"** → **6. 실험 결과**.

첫 독서에서는 문제와 방법의 흐름을 잡자. 수식·코드·세부 성능표는 두 번째에 읽어도 된다. 구현과 수학적 주장까지 확인하려면 그 부분을 다시 살펴봐야 한다.

<!-- PAPER-PREREQUISITES -->
### 선수 개념과 읽을 논문

**먼저 이해할 개념:** 확률과 로그, 손실 최소화, 학습률, 정책과 보상; KL을 쓰는 글은 비교 방향도 확인.

아래 순서는 이 글을 이해하기 위한 공부 순서다. ‘개념 준비’는 읽기 안내이며, 직접적인 연구 계보는 연결 칸에 따로 표시했다. 이미 해당 개념을 설명할 수 있다면 아는 부분은 넘어가도 된다.

| 순서·논문 | 연결·익힐 개념 | 먼저 볼 부분 |
|---|---|---|
| 1. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) · [블로그 해설]({{< relref "posts/reinforcement-learning-beginner-guide" >}}) | **개념 준비**: 정책·보상·정책 갱신 | 행동의 확률을 갱신하는 목적과 큰 변경을 제한하는 장치 |
| 2. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) · [블로그 해설]({{< relref "posts/reinforcement-learning-beginner-guide" >}}) | **개념 준비**: SFT·선호 데이터·보상 모델·RLHF | 지도 학습, 보상 학습, 정책 학습을 나눈 구성도 |

**현재 글과 연결하기:** 위 개념을 익힌 뒤, 이 글에서 ‘두 답의 선호 비교로 모델을 조정하는 원리 이해하기’에 해당하는 부분을 읽자. 기존 방법의 어느 단계가 바뀌었는지 그림이나 입력·출력으로 설명해 보자.

[선수 논문 전체 지도와 원문 읽는 법]({{< relref "paper-prerequisites" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- PAPER-PREREQUISITES-END -->

<!-- LEARNING-START-END -->

---

## 논문 정보

- **제목**: Direct Preference Optimization: Your Language Model is Secretly a Reward Model
- **저자**: Rafael Rafailov\*, Archit Sharma\*, Eric Mitchell\*, Stefano Ermon, Christopher D. Manning, Chelsea Finn
- **소속**: Stanford University, CZ Biohub
- **학회**: **NeurIPS 2023 Outstanding Paper Award**
- **arXiv**: 2305.18290
- **코드**: [github.com/eric-mitchell/direct-preference-optimization](https://github.com/eric-mitchell/direct-preference-optimization)

---

## 1. 문제: RLHF는 왜 복잡하고 불안정한가?

### 기존 RLHF 파이프라인 (3단계)

```
[1] SFT: 고품질 데이터로 지도학습 파인튜닝
          ↓
[2] Reward Model: 선호 쌍 (y_w ≻ y_l)으로 별도 보상 모델 학습
          ↓
[3] PPO: 학습된 r_φ로 KL 제약 하 보상 최대화 강화학습
```

### RLHF의 고질병

| 문제 | 설명 |
|------|------|
| **4개 모델 동시 로딩** | 정책, 참조, 보상, 가치 모델 → GPU 메모리 4배 |
| **매 스텝 샘플링** | 정책에서 생성 → 계산 비용 막대 |
| **RL 본질의 불안정성** | PPO 그래디언트 분산 큼, 하이퍼파라미터 민감 |
| **Reward Hacking** | 학습된 보상 모델이 정답 아님 → 과최적화 위험 |
| **구현 복잡도** | Actor-Critic, GAE, clipping, KL 계수 등 많은 튜닝 포인트 |

> **DPO의 질문**: 정말로 RL이 필요한가?

---

## 2. 핵심 통찰: "당신의 LM은 이미 보상모델이다"

### 한 줄 요약

> **KL 제약 보상 최대화 문제의 최적해가 닫힌 형태로 존재하므로, 보상함수를 정책으로 재매개변수화하면 RL 없이 단순 분류 손실만으로 동일한 최적정책을 얻을 수 있다.**

정책 자체가 **암묵적 보상 모델**(implicit reward model) 역할을 하므로, 별도 보상 모델이 필요 없다.

선호 데이터에는 같은 질문에 대한 선택된 답과 선택되지 않은 답이 들어간다. DPO는 이 쌍을 이용해 정책의 상대적인 확률을 조정한다. 수식에서 참조 정책이 함께 나오는 것은 기존 정책과 얼마나 달라지는지를 학습에 반영하기 위해서다. 아래 유도는 보상 최대화 문제를 이 답변 쌍의 확률 비교로 어떻게 옮기는지 따라가는 과정이다.

---

## 3. 수학적 유도 (상세)

### 3.1 RLHF 목적함수

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} [r_\phi(x, y)] - \beta \cdot D_{KL}[\pi_\theta(y|x) \| \pi_{\text{ref}}(y|x)]$$

- $\beta$: KL 제약 강도
- $\pi_{\text{ref}}$: 참조 정책 (보통 SFT 모델)

### 3.2 닫힌 형태 최적 정책

KL divergence를 풀어 쓰면 다음 식으로 정리된다.

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

여기서 분배함수 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp(\frac{1}{\beta} r(x, y))$.

**문제**: $Z(x)$는 모든 $y$에 대한 합산이라 **계산 불가능**.

### 3.3 보상을 정책으로 표현 (핵심 변수치환)

위 식 양변에 로그를 취하고 $r$에 대해 정리:

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

**핵심 관찰**: $Z(x)$는 $y$에 의존하지 않는다!

### 3.4 Bradley-Terry 모델

두 응답에 대한 선호 확률:

$$p(y_1 \succ y_2 | x) = \frac{\exp(r(x, y_1))}{\exp(r(x, y_1)) + \exp(r(x, y_2))} = \sigma(r(x, y_1) - r(x, y_2))$$

**두 보상의 차를 계산하면 $Z(x)$가 상쇄된다!**

$$r(x, y_1) - r(x, y_2) = \beta \log \frac{\pi^*(y_1|x)}{\pi_{\text{ref}}(y_1|x)} - \beta \log \frac{\pi^*(y_2|x)}{\pi_{\text{ref}}(y_2|x)}$$

### 3.5 DPO 손실 함수

BT 모델에 대입 + MLE:

$$\boxed{\ \mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]\ }$$

### 3.6 그래디언트의 의미

$$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \mathbb{E}\left[\underbrace{\sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w))}_{\text{보상 순서 오류 가중치}} \cdot \left[\nabla_\theta \log \pi(y_w|x) - \nabla_\theta \log \pi(y_l|x)\right]\right]$$

- $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$: **암묵적 보상**
- 가중치는 "암묵 보상이 선호를 얼마나 틀리게 순위 매겼는가"에 비례

**직관**: 선호 응답은 가능도 증가, 비선호 응답은 가능도 감소. **단, 오분류일수록 가중치가 커져** 단순 likelihood ratio 손실의 붕괴를 방지.

---

## 4. RLHF(PPO) vs DPO

| | RLHF (PPO) | DPO |
|--|-----------|-----|
| 단계 수 | **3단계** (SFT→RM→PPO) | **2단계** (SFT→DPO) |
| 필요 모델 | 정책, 참조, 보상, 가치 **4개** | 정책, 참조 **2개** |
| 학습 중 샘플링 | **매 스텝 필요** | 불필요 |
| 손실 함수 | PPO (Actor-Critic, clipping, GAE...) | **Log-sigmoid 분류 손실** |
| 구현 복잡도 | 매우 높음 | **PyTorch 10줄** |
| 안정성 | RL 본질적 불안정 | **안정** |
| 보상 모델 | 별도 네트워크 | **정책에 암묵적으로 내장** |

### DPO 구현 (논문 Appendix B)

```python
import torch.nn.functional as F

def dpo_loss(pi_logps, ref_logps, yw_idxs, yl_idxs, beta):
    pi_yw_logps, pi_yl_logps = pi_logps[yw_idxs], pi_logps[yl_idxs]
    ref_yw_logps, ref_yl_logps = ref_logps[yw_idxs], ref_logps[yl_idxs]
    
    pi_logratios = pi_yw_logps - pi_yl_logps
    ref_logratios = ref_yw_logps - ref_yl_logps
    
    losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
    rewards = beta * (pi_logps - ref_logps).detach()
    
    return losses, rewards
```

정말 단순하다.

---

## 5. 이론적 분석

### Theorem 1 (표현력 손실 없음)

> Plackett-Luce 모델과 일치하는 **모든 보상 함수 동치류**는 $r(x,y) = \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}$ 형태로 **유일하게** 재매개변수화 가능하다.

**함의**: DPO의 재매개변수화는 보상 모델이 가진 표현력을 **전혀 잃지 않는다**. 이론적으로 RLHF와 완전히 동등.

### PPO 불안정성의 이론적 원인

PPO의 실제 목적함수는 **soft value function**을 학습/추정해야 함:

$$\max_\pi \mathbb{E}\left[r_\phi(x, y) - \beta \log \sum_y \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r_\phi\right)\right]$$

이 정규화 항이 분산이 크고 추정이 어려움 → PPO의 불안정성 원인.

**DPO는 이 항이 자동으로 상쇄되므로** 안정적이다.

---

## 6. 실험 결과

### 6.1 Controlled Sentiment (IMDb)

- GPT-2-large 기반, 감성 분류기로 ground-truth 보상 제공
- **DPO가 PPO-GT(oracle 보상 사용)를 지배(strictly dominates)** — 모든 KL 값에서 더 높은 보상
- RL의 oracle보다 더 나은 결과

### 6.2 Summarization (Reddit TL;DR)

| 방법 | GPT-4 Win Rate |
|------|---------------|
| SFT | ~40% |
| Preferred-FT | ~42% |
| PPO (최적 temp) | 57% |
| **DPO (temp=0)** | **~61%** |

- **샘플링 온도 robust**: PPO는 고온에서 base 수준으로 붕괴하지만 DPO는 안정

### 6.3 Single-turn Dialogue (Anthropic HH)

- **DPO만 유일하게** chosen 베이스라인을 이김 (~60%+)
- Best-of-128 수준 성능을 훨씬 적은 계산으로 달성
- 수렴 속도도 빠름 (~1,000 steps 이내)

### 6.4 OOD 일반화 (CNN/DailyMail)

| 방법 | temp=0 | temp=0.25 |
|------|--------|-----------|
| PPO | 0.26 | 0.23 |
| **DPO** | **0.36** | **0.31** |

분포 밖에서도 DPO가 우수.

### 6.5 인간 평가

- GPT-4 평가와 인간 평가의 일치도 = 인간끼리의 일치도 → GPT-4 평가 신뢰성 확인
- DPO vs PPO에서 인간의 **58%가 DPO 선호**

---

## 7. 하이퍼파라미터

| 항목 | 값 |
|------|-----|
| $\beta$ | **0.1** (기본), 0.5 (TL;DR) |
| batch size | 64 |
| Optimizer | RMSprop |
| Learning rate | **1e-6** (linear warmup 150 steps) |
| Label smoothing | 0 ~ 0.5 (Conservative DPO) |

**$\beta$의 역할**:
- 너무 크면: 참조 모델에 갇혀 개선 없음
- 너무 작으면: Reward hacking, 분포 이탈
- 일반적 범위: **0.01 ~ 0.5**

---

## 8. 학습 데이터 포맷

각 샘플은 3-튜플:

```python
{
    "prompt": "하늘은",
    "chosen": " 푸르다.",    # 선호 응답 (y_w)
    "rejected": " 녹색이다."  # 비선호 응답 (y_l)
}
```

대화형 포맷도 지원:

```python
{
    "prompt": [{"role": "user", "content": "..."}],
    "chosen": [{"role": "assistant", "content": "..."}],
    "rejected": [{"role": "assistant", "content": "..."}]
}
```

---

## 9. 후속 연구 — DPO의 진화

| 방법 | 논문 | 핵심 차이 |
|------|------|---------|
| **IPO** (2023) | Azar et al. | Sigmoid 포화 문제 해결 — identity mapping으로 직접 margin 최적화 |
| **KTO** (2024) | Ethayarajh et al. | **Unpaired binary label**만 필요 (desirable/undesirable). Kahneman-Tversky 손실 회피 이론 |
| **ORPO** (2024) | Hong et al. | **참조 모델 불필요**. SFT + 선호 학습을 **1-stage**로 통합 |
| **SimPO** (NeurIPS 2024) | Meng et al. | **길이 정규화된 평균 log-prob**을 암묵 보상으로. **참조 모델 제거**. AlpacaEval 2에서 DPO +6.4점 |
| **CPO** (2024) | Xu et al. | 번역 특화, SFT 정규화 추가, 참조 모델 프리 |
| **cDPO / Robust DPO** | - | 라벨 노이즈 robust, label smoothing |
| **Iterative DPO** | - | Online DPO — 현재 정책이 생성한 응답에 선호 라벨 → 반복. Zephyr, Llama-3에 적용 |

### SimPO (가장 주목할 만한 후속작)

$$\mathcal{L}_{\text{SimPO}} = -\log \sigma\left(\beta \cdot \left(\frac{1}{|y_w|} \log \pi_\theta(y_w|x) - \frac{1}{|y_l|} \log \pi_\theta(y_l|x)\right) - \gamma\right)$$

- **참조 모델 $\pi_{\text{ref}}$ 완전 제거**
- 길이 정규화 ($|y|$로 나눔) → 길이 편향 완화
- Target reward margin $\gamma$ 추가

---

## 10. 실전 사용 (TRL)

```python
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

trainer = DPOTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=load_dataset("trl-lib/ultrafeedback_binarized", split="train"),
    args=DPOConfig(
        learning_rate=1e-6,
        beta=0.1,
        loss_type="sigmoid",  # "ipo", "hinge", "robust" 등 지원
        bf16=True,
        gradient_checkpointing=True,
    ),
)
trainer.train()
```

### 주요 로깅 메트릭

| 메트릭 | 의미 |
|-------|------|
| `rewards/chosen` | 선호 응답의 암묵 보상 |
| `rewards/rejected` | 비선호 응답의 암묵 보상 |
| `rewards/margins` | chosen - rejected (양수로 증가해야 정상) |
| `rewards/accuracies` | chosen > rejected 비율 |

---

## 11. 한계와 흔한 함정

### 11.1 과적합 (Overfitting)

Sigmoid 포화 → gradient 소실 → **암기에 갇힘**. IPO가 지적한 주요 비판.

### 11.2 길이 편향 (Length Bias)

인간 어노테이터가 긴 응답 선호 → DPO가 장황함 학습. **SimPO의 길이 정규화** 또는 길이 매칭 페어로 완화.

### 11.3 데이터 편향 내재화

선호 쌍의 어노테이터 편향/스타일 편향을 직접 학습 → **다양한 소스의 고품질 데이터** 필수.

### 11.4 $\beta$ 민감도

너무 크면 ref에 고정, 너무 작으면 reward hacking. **0.01~0.5 범위 탐색** 필요.

### 11.5 Implicit Reward Over-optimization

오래 학습하면 win rate 감소 현상 → RLHF의 reward hacking과 유사.

### 11.6 로그 확률 동시 감소

chosen/rejected 모두 **동시에 감소**하는 패턴 발견 — 비선호 억제가 선호 증가보다 강함. **SFT loss 추가**로 완화.

---

## 12. 실제 채택 사례

| 모델 | 적용 |
|------|------|
| **Zephyr-7B-β** | Mistral-7B + UltraFeedback DPO → 2023년 최고 7B 오픈 모델 |
| **Tulu 2/3** (Allen AI) | Llama 3.1 기반, Length-normalized DPO가 PPO/DPO/SimPO 중 최고 |
| **Llama 2/3 Instruct** | Rejection sampling + PPO/DPO 혼합 |
| **Mixtral Instruct, Qwen, DeepSeek** | 대부분 오픈 instruction 모델이 DPO/변형 채택 |

**UltraFeedback** 데이터셋이 오픈 커뮤니티 표준 선호 데이터 (60k → 300k+ 확장).

---

## 13. 핵심 요약

### DPO의 3가지 기여

1. **수학적 기여**: KL 제약 보상 최대화의 닫힌 해를 BT 모델에 대입해 분배함수 $Z(x)$를 상쇄
2. **이론적 기여**: Theorem 1로 RLHF와의 완전 등가성 증명
3. **실용적 기여**: RL 제거 → 단순 분류 손실 → 구현 복잡도 대폭 감소

### 한 줄로

> **"RLHF의 복잡한 3단계 파이프라인을, 수학적 등가성을 유지하며 2단계 분류 문제로 치환한 Nobel급 통찰."**

---

## 14. RAG/RL 학습 맥락에서 DPO의 위치

```
일반 LM 학습 (사전학습 + SFT)
        ↓
★ DPO/IPO/SimPO로 선호도 정렬
        ↓
도메인 RAG 최적화
  ├── 검색기 FT (contrastive)
  ├── RAFT (방해 문서 혼합)
  └── DPO (충실도 선호)
        ↓
최종 정렬된 RAG 시스템
```

**RAG에서 DPO 적용**:
- 선호: 문서를 정확히 인용한 답변
- 비선호: 할루시네이션된 답변
- 효과: **충실도(faithfulness) 향상**

---

## 15. 관련 블로그 포스트

- [도메인 최적화 LLM for RAG](domain-optimized-llm-for-rag.md) — DPO를 RAG에 적용
- [RAFT 상세 리뷰](raft-review.md) — 방해 문서 혼합 SFT
- [Search-R1 상세 리뷰](search-r1-review.md) — RL 기반 검색 에이전트
- [CCS 상세 리뷰](cycle-consistent-search-review.md) — 정답 없는 RL
- [RAG End-to-end 공동 학습 서베이](rag-end-to-end-training-survey.md)

---

## 참고 자료

- [DPO 논문 (arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [공식 구현 (eric-mitchell/direct-preference-optimization)](https://github.com/eric-mitchell/direct-preference-optimization)
- [HuggingFace TRL DPOTrainer 문서](https://huggingface.co/docs/trl/main/en/dpo_trainer)
- [DPO 종합 서베이 (arXiv:2410.15595)](https://arxiv.org/html/2410.15595v3)
- [IPO (arXiv:2310.12036)](https://arxiv.org/abs/2310.12036)
- [KTO (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [SimPO (NeurIPS 2024)](https://github.com/princeton-nlp/SimPO)
- [Tülu 3 (arXiv:2411.15124)](https://arxiv.org/abs/2411.15124)

<!-- LEARNING-END -->
---

## 읽은 뒤 확인하고 다음으로

**확인 질문:** 두 답의 선호 비교로 모델을 조정하는 원리 이해하기를 목표로, 본문에서 다룬 입력·중간 과정·결과를 자기 말로 설명해 보자. 실험이나 사례가 있다면 어떤 조건에서 확인했는지도 덧붙여 보자.

**다음 글:** [\[리뷰\] SDFT — 새로 배워도 잊지 않는 파인튜닝: Self-Distillation Enables Continual Learning]({{< relref "posts/sdft-self-distillation-review" >}}) — 예시를 본 모델이 보지 않은 모델을 가르치는 과정 이해하기.

[전체 학습 순서]({{< relref "learn" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- LEARNING-END-END -->
