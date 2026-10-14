---
title: "[개념 정리] Bayesian Surprise — 놀라움의 수학적 정의와 뇌의 주의 메커니즘"
date: 2026-03-26
tags: ["논문리뷰", "베이지안", "정보이론", "신경과학"]
categories: ["ML/AI"]
summary: "Bayesian Surprise를 수식부터 직관까지 상세히 설명한다. Shannon Surprise와의 차이, KL divergence로의 정의, 뇌의 주의(attention) 메커니즘과의 연결, 그리고 Predictive Coding과의 관계를 다룬다."
math: true
toc: true
draft: false
---

<!-- LEARNING-START -->
## 처음 읽는 분을 위한 안내

**학습 위치:** [문장을 읽으며 기억하는 모델]({{< relref "learn" >}}#memory) · 1/13

보기 드문 일을 겪는 것과 그 일 때문에 생각을 크게 바꾸는 것은 다르다. 이 글은 두 종류의 놀라움을 확률로 구분한다. 비나 동전 예시를 먼저 읽고, 믿음이 관찰 전후에 어떻게 바뀌는지 이해한 뒤 수식을 보자.

**먼저 알아둘 말**

- [확률 분포]({{< relref "ai-glossary" >}}#distribution): 가능한 결과들에 확률을 나눠 준 표현. 모델은 한 답만 정하는 대신 다음 출력의 여러 가능성을 계산할 수 있다.
- [사전·사후 분포]({{< relref "ai-glossary" >}}#bayesian): 새 관찰 전의 믿음을 확률로 나타낸 것이 사전 분포, 관찰을 반영해 갱신한 것이 사후 분포다.
- [KL divergence]({{< relref "ai-glossary" >}}#kl): 두 확률 분포의 차이를 측정하는 양. 비교 방향을 바꾸면 값과 학습 효과도 달라질 수 있다.

**이번 글의 목표:** 드문 사건과 믿음의 변화를 구분하기.

**이 학습 경로의 앞 글:** [AI 기초 설명]({{< relref "ai-basics" >}}).

토큰과 벡터가 낯설다면 [텍스트 데이터 다루기]({{< relref "posts/build-llm-ch2-working-with-text-data" >}})의 설명부터 읽자.

**본문에서 먼저 볼 순서:** **1. 직관: "놀라움"이란 무엇인가?**.

첫 독서에서는 문제와 방법의 흐름을 잡자. 수식·코드·세부 성능표는 두 번째에 읽어도 된다. 구현과 수학적 주장까지 확인하려면 그 부분을 다시 살펴봐야 한다.

<!-- PAPER-PREREQUISITES -->
### 선수 개념과 읽을 논문

**먼저 이해할 개념:** 사건의 확률과 로그, 조건부 확률·베이즈 정리, 사전/사후 분포, KL divergence.

조건부 확률·베이즈 정리·KL divergence를 먼저 익히자. Shannon의 정보량과 믿음 변화의 차이를 잡은 뒤, 본문에서 다루는 Bayesian Surprise를 읽으면 된다.

아래 순서는 이 글을 이해하기 위한 공부 순서다. ‘개념 준비’는 읽기 안내이며, 직접적인 연구 계보는 연결 칸에 따로 표시했다. 이미 해당 개념을 설명할 수 있다면 아는 부분은 넘어가도 된다.

| 순서·논문 | 연결·익힐 개념 | 먼저 볼 부분 |
|---|---|---|
| 1. [A Mathematical Theory of Communication](https://onlinelibrary.wiley.com/doi/10.1002/j.1538-7305.1948.tb00917.x) | **개념 준비**: 정보량·엔트로피·확률과 로그 | 개별 사건의 정보량과 분포의 불확실성을 구분하는 설명 |

**현재 글과 연결하기:** 위 개념을 익힌 뒤, 이 글에서 ‘드문 사건과 믿음의 변화를 구분하기’에 해당하는 부분을 읽자. 기존 방법의 어느 단계가 바뀌었는지 그림이나 입력·출력으로 설명해 보자.

[선수 논문 전체 지도와 원문 읽는 법]({{< relref "paper-prerequisites" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- PAPER-PREREQUISITES-END -->

<!-- LEARNING-START-END -->

---

## 1. 직관: "놀라움"이란 무엇인가?

일상에서 말하는 놀라움을 수학으로 표현하려면 두 가지를 먼저 구분해야 한다.

- **예상 밖의 데이터**: "비가 올 확률이 10%인데 비가 왔다" → 데이터 자체가 드물다
- **믿음이 바뀌는 경험**: "비가 올 확률이 10%라고 믿었는데, 새 증거를 보고 70%로 바꿨다" → **내 모델이 바뀌었다**

예상 밖의 데이터가 주는 놀라움은 Shannon Surprise, 믿음이 바뀌는 정도는 **Bayesian Surprise**라고 부른다.

가령 드문 결과가 나왔어도 원래 모델이 그 가능성을 충분히 설명하고 있었다면, 믿음은 크게 바뀌지 않을 수 있다. 반대로 새 관찰 때문에 어느 모델을 믿어야 할지가 달라지면 Bayesian Surprise가 커진다. 아래 수식을 읽을 때도 관찰한 사건의 확률과 관찰 전후 모델 분포의 변화를 따로 생각하면 두 정의를 혼동하지 않게 된다.

---

## 2. Shannon Surprise (전통적 놀라움)

### 정의

데이터 $d$가 관찰되었을 때의 놀라움:

$$S_{Shannon}(d) = -\log p(d)$$

확률이 낮을수록 놀랍다. 이것은 **데이터의 드물음**을 측정한다.

### 한계

Shannon Surprise는 **모델의 변화를 반영하지 못한다**.

예시: 동전을 100번 던져서 모두 앞면이 나왔다.
- 101번째에 앞면: $S_{Shannon} = -\log(0.5) = 1$ bit (공정한 동전 가정)
- 하지만 이미 100번 앞면을 봤으므로, **나는 이 동전이 편향되었다고 믿음을 바꿨다**
- 101번째 앞면은 나에게 전혀 놀랍지 않다 — 이미 예상했으니까

Shannon Surprise는 이 차이를 포착하지 못한다.

---

## 3. Bayesian Surprise — 믿음의 변화량

### 핵심 논문

- **제목**: Bayesian Surprise Attracts Human Attention
- **저자**: Laurent Itti, Pierre Baldi (USC)
- **학회**: NeurIPS 2005 (당시 NIPS)

### 정의

Bayesian Surprise는 **데이터를 관찰하기 전후로 믿음(모델)이 얼마나 바뀌었는지**를 측정한다.

관찰자가 모델 $\mathcal{M}$에 대해 사전 분포 $p(\mathcal{M})$을 가지고 있다고 하자. 데이터 $D$를 관찰한 후 사후 분포 $p(\mathcal{M} \mid D)$로 업데이트된다.

Bayesian Surprise는 이 두 분포 사이의 **KL divergence**:

$$S_{Bayes}(D) = D_{KL}\left(p(\mathcal{M} \mid D) \;\lVert\; p(\mathcal{M})\right)$$

$$= \int p(\mathcal{M} \mid D) \log \frac{p(\mathcal{M} \mid D)}{p(\mathcal{M})} \, d\mathcal{M}$$

### 직관

- $p(\mathcal{M})$: 데이터를 보기 **전**의 믿음
- $p(\mathcal{M} \mid D)$: 데이터를 본 **후**의 믿음
- $D_{KL}$: 두 분포 사이의 "거리" (엄밀히는 비대칭 발산)

**믿음이 많이 바뀌면 놀라운 것이고 적게 바뀌면 놀랍지 않은 것이다.**

---

## 4. 수식 전개: 베이즈 정리와의 연결

### 사후 분포 (Bayes' Rule)

$$p(\mathcal{M} \mid D) = \frac{p(D \mid \mathcal{M}) \cdot p(\mathcal{M})}{p(D)}$$

### KL Divergence 전개

$$S_{Bayes}(D) = \int p(\mathcal{M} \mid D) \log \frac{p(\mathcal{M} \mid D)}{p(\mathcal{M})} \, d\mathcal{M}$$

베이즈 정리를 대입하면:

$$= \int p(\mathcal{M} \mid D) \log \frac{p(D \mid \mathcal{M})}{p(D)} \, d\mathcal{M}$$

$$= \int p(\mathcal{M} \mid D) \log p(D \mid \mathcal{M}) \, d\mathcal{M} - \log p(D)$$

$$= \mathbb{E}_{p(\mathcal{M}|D)}[\log p(D \mid \mathcal{M})] - \log p(D)$$

**해석**: Bayesian Surprise = 사후 분포 하에서의 평균 로그 가능도 - 데이터의 로그 증거

---

## 5. 구체적 예시: 동전 던지기

### 설정

- 모델 파라미터: 동전의 앞면 확률 $\theta \in [0, 1]$
- 사전 분포: $p(\theta) = \text{Beta}(\alpha, \beta)$

### Beta 분포 복습

$$p(\theta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha, \beta)}$$

- $\alpha$: 앞면 관찰 횟수 + 1
- $\beta$: 뒷면 관찰 횟수 + 1
- 균일 사전 분포: $\text{Beta}(1, 1)$

### 데이터 관찰

앞면($H$)이 나왔다면:

$$p(\theta \mid H) = \text{Beta}(\alpha + 1, \beta)$$

### Bayesian Surprise 계산

두 Beta 분포 사이의 KL divergence:

$$S_{Bayes}(H) = D_{KL}(\text{Beta}(\alpha+1, \beta) \;\lVert\; \text{Beta}(\alpha, \beta))$$

$$= \log \frac{B(\alpha, \beta)}{B(\alpha+1, \beta)} + (\alpha+1-\alpha)\left[\psi(\alpha+1) - \psi(\alpha+\beta+1)\right]$$

$$= \log \frac{\alpha + \beta}{\alpha} + \psi(\alpha+1) - \psi(\alpha+\beta+1)$$

여기서 $\psi$는 digamma 함수, $B$는 beta 함수.

### 숫자 예시

| 상황 | 사전 $(\alpha, \beta)$ | 데이터 | Bayesian Surprise | 해석 |
|------|----------------------|--------|-------------------|------|
| 아무것도 모름 | (1, 1) | 앞면 | **0.19 nats** | 약간 놀라움 — 균일 분포에서 조금 이동 |
| 공정하다고 확신 | (50, 50) | 앞면 | **0.005 nats** | 거의 안 놀라움 — 이미 확신이 강함 |
| 편향 의심 중 | (2, 10) | 앞면 | **0.12 nats** | 꽤 놀라움 — 뒷면 위주였는데 앞면! |
| 편향 의심 중 | (2, 10) | 뒷면 | **0.01 nats** | 안 놀라움 — 예상대로 |

핵심: **동일한 데이터(앞면)라도, 관찰자의 사전 믿음에 따라 놀라움이 달라진다.**

---

## 6. Shannon Surprise vs Bayesian Surprise

| 항목 | Shannon Surprise | Bayesian Surprise |
|------|-----------------|-------------------|
| 정의 | $-\log p(d)$ | $D_{KL}(p(\mathcal{M} \mid D) \lVert p(\mathcal{M}))$ |
| 측정 대상 | **데이터의 드물음** | **믿음의 변화량** |
| 모델 의존 | 고정된 하나의 모델 | 모델에 대한 **분포** |
| 관찰자 의존 | 아니오 | **예** (사전 분포에 의존) |
| 반복 관찰 | 항상 동일 | **감소** (이미 학습했으므로) |
| 수학적 성질 | 항상 $\geq 0$ | 항상 $\geq 0$ (KL 성질) |

### 핵심 차이를 보여주는 예시

**시나리오**: 주사위를 1000번 던져서 모두 6이 나왔다. 1001번째도 6이 나왔다.

- **Shannon Surprise**: $-\log(1/6) = 2.58$ bits — 여전히 "놀라운" 이벤트
- **Bayesian Surprise**: $\approx 0$ — 전혀 놀랍지 않음. 이미 이 주사위가 6만 나온다고 학습했으므로 믿음이 바뀌지 않는다

---

## 7. 뇌의 주의(Attention) 메커니즘과의 연결

### Itti & Baldi (2005)의 핵심 발견

> **인간의 시선(eye fixation)은 Bayesian Surprise가 높은 위치로 향한다.**

실험:
1. 피험자에게 자연 이미지/비디오를 보여주며 시선 추적
2. 이미지의 각 위치에서 Bayesian Surprise 계산
3. 시선 고정 위치와 Bayesian Surprise 맵의 상관 분석

결과: Bayesian Surprise가 기존의 saliency 모델(밝기, 색상, 방향의 대비)보다 **인간의 시선을 더 잘 예측**했다.

### 뇌는 Shannon Surprise가 아닌 Bayesian Surprise에 반응한다

- 처음 보는 패턴 → 높은 Bayesian Surprise → **주의 집중**
- 반복된 패턴 → 낮은 Bayesian Surprise → **주의 감소** (habituation)
- 이것은 Shannon Surprise로는 설명 불가 (같은 패턴의 Shannon Surprise는 항상 동일)

---

## 8. Predictive Coding과의 연결

### Free Energy Principle (Friston)

Karl Friston의 자유 에너지 원리에서, 뇌는 **variational free energy**를 최소화한다:

$$F = D_{KL}(q(\theta) \lVert p(\theta \mid D)) = -\text{ELBO}$$

이를 분해하면:

$$F = \underbrace{D_{KL}(q(\theta) \lVert p(\theta))}_{\text{Complexity (≈ Bayesian Surprise)}} - \underbrace{\mathbb{E}_q[\log p(D \mid \theta)]}_{\text{Accuracy}}$$

**Complexity 항이 곧 Bayesian Surprise**이다. 뇌가 자유 에너지를 최소화한다는 것은, Bayesian Surprise를 적절히 관리한다는 것이다.

### Predictive Coding에서의 역할

Predictive Coding Network에서:

$$\text{Prediction Error} = x_l - \hat{x}_l$$

이 prediction error가 크다는 것은 **현재 모델의 예측이 실제와 다르다**는 것이고 모델을 업데이트해야 한다는 것이다. 이것은 Bayesian Surprise의 신경 구현으로 해석할 수 있다:

- **높은 prediction error** → 높은 Bayesian Surprise → 모델 업데이트 (학습)
- **낮은 prediction error** → 낮은 Bayesian Surprise → 모델 유지

---

## 9. 응용 분야

### 9.1 이상 탐지 (Anomaly Detection)

Bayesian Surprise가 높은 데이터 포인트 = 기존 모델에서 벗어나는 이상치.

$$\text{Anomaly Score}(d) = D_{KL}(p(\theta \mid d) \lVert p(\theta))$$

Shannon Surprise 기반($-\log p(d)$)보다 **적응적** — 정상 패턴이 변화해도 자동으로 추적한다.

### 9.2 호기심 기반 강화학습 (Curiosity-Driven RL)

에이전트의 내적 보상(intrinsic reward)으로 Bayesian Surprise를 사용:

$$r_{intrinsic} = D_{KL}(p(\theta \mid s, a, s') \lVert p(\theta))$$

새로운 전이 $(s, a, s')$가 모델을 많이 바꾸면 높은 보상 → **탐험 촉진**.

### 9.3 능동 학습 (Active Learning)

가장 "놀라운" 데이터를 선택적으로 라벨링:

$$d^{\ast} = \arg\max_d \mathbb{E}_{p(y|d)}[S_{Bayes}(d, y)]$$

예상 Bayesian Surprise가 가장 높은 데이터를 먼저 라벨링하면, 적은 라벨로 빠르게 학습할 수 있다.

### 9.4 어텐션 메커니즘 설계

Transformer의 softmax attention 대신 **Bayesian Surprise 기반 어텐션 가중치**를 사용하는 연구:

$$\text{Attention}(q, k_i) \propto S_{Bayes}(k_i \mid q, \text{context})$$

문맥에서 모델을 가장 많이 바꾸는 키에 높은 가중치를 준다.

---

## 10. 수학적 성질 요약

### 비음수성

$$S_{Bayes}(D) = D_{KL}(p(\mathcal{M} \mid D) \lVert p(\mathcal{M})) \geq 0$$

KL divergence의 성질에 의해 항상 0 이상. 등호는 $p(\mathcal{M} \mid D) = p(\mathcal{M})$일 때 — 데이터가 믿음을 전혀 바꾸지 않을 때.

### 가법성 (독립 데이터)

데이터 $D_1, D_2$가 독립이면:

$$S_{Bayes}(D_1, D_2) \approx S_{Bayes}(D_1) + S_{Bayes}(D_2 \mid D_1)$$

### 순서 의존성

일반적으로 $S_{Bayes}(D_1 \text{ then } D_2) \neq S_{Bayes}(D_2 \text{ then } D_1)$. 같은 데이터라도 **관찰 순서에 따라 놀라움이 다르다** — 직관과 일치한다.

### 학습에 따른 감소

동일한 유형의 데이터를 반복 관찰하면 Bayesian Surprise는 **단조 감소**한다:

$$S_{Bayes}(D_n) \leq S_{Bayes}(D_1) \quad \text{(같은 유형의 데이터)}$$

사후 분포가 점점 집중되면서(분산 감소), 새 데이터가 분포를 덜 바꾸기 때문이다.

<!-- LEARNING-END -->
---

## 읽은 뒤 확인하고 다음으로

**확인 질문:** 드문 사건과 믿음의 변화를 구분하기를 목표로, 본문에서 다룬 입력·중간 과정·결과를 자기 말로 설명해 보자. 실험이나 사례가 있다면 어떤 조건에서 확인했는지도 덧붙여 보자.

**다음 글:** [\[개념 정리\] State Space Model(SSM) — 수식으로 이해하는 Mamba의 수학적 기반]({{< relref "posts/ssm-explained" >}}) — 새 입력과 과거 요약으로 상태가 갱신되는 흐름 이해하기.

[전체 학습 순서]({{< relref "learn" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- LEARNING-END-END -->
