---
title: "[논문 리뷰] Forward-Forward Algorithm: 역전파 없이 두 번의 Forward Pass로 학습하기"
date: 2026-03-19
tags: ["논문리뷰", "비역전파", "Forward-Forward", "Hinton"]
categories: ["ML/AI"]
summary: "Geoffrey Hinton이 제안한 Forward-Forward Algorithm 리뷰. Backward pass를 완전히 제거하고, positive/negative 두 번의 forward pass만으로 레이어별 독립 학습을 수행한다."
math: true
toc: true
draft: false
---

<!-- LEARNING-START -->
## 처음 읽는 분을 위한 안내

**학습 위치:** [역전파를 바꾸면 무엇이 달라질까?]({{< relref "learn" >}}#no-bp) · 2/7

좋은 입력과 나쁜 입력을 보여 주고 각 계산 층이 둘을 구분하도록 가르치는 방법이다. 전체 출력에서 거슬러 올라오는 학습 신호 대신 층별 점수를 사용한다. 어떤 입력을 좋고 나쁘다고 정했는지도 함께 읽어야 한다.

**먼저 알아둘 말**

- [그래디언트·역전파]({{< relref "ai-glossary" >}}#gradient): 그래디언트는 숫자를 어느 방향으로 바꾸면 손실이 달라지는지 나타낸다. 역전파는 연결된 계산을 거슬러 그 값을 구하는 방법이다.
- [손실 함수]({{< relref "ai-glossary" >}}#loss): 예측이 목표에서 얼마나 벗어났는지 계산하는 기준. 학습에서는 보통 이 값을 줄이려 한다.
- [파라미터·가중치]({{< relref "ai-glossary" >}}#parameter): 모델 안에서 학습으로 조정하는 숫자. 모든 내부 숫자가 항상 학습 대상인 것은 아니다.

**이번 글의 목표:** 좋은 입력과 나쁜 입력을 층마다 구분하는 학습 이해하기.

**이 학습 경로의 앞 글:** [역전파 없이 학습할 수 있을까? — Forward-Forward부터 EBT까지]({{< relref "posts/no-bp-methods-survey" >}}).

**본문에서 먼저 볼 순서:** **1. 동기: 왜 Backward Pass를 없애려 하는가?** → **2. 핵심 아이디어: Goodness 기반 대조 학습** → **6. 실험 결과**.

첫 독서에서는 문제와 방법의 흐름을 잡자. 수식·코드·세부 성능표는 두 번째에 읽어도 된다. 구현과 수학적 주장까지 확인하려면 그 부분을 다시 살펴봐야 한다.

<!-- PAPER-PREREQUISITES -->
### 선수 개념과 읽을 논문

**먼저 이해할 개념:** 손실 함수, 미분·연쇄법칙, 가중치와 활성값의 구분; 에너지 기반 방법은 최적화와 수렴.

먼저 역전파의 연쇄법칙과 레이어별 손실을 익히자. EqProp은 에너지와 국소 학습 신호를 비교하는 선택 독서이며 Forward-Forward의 직접 기반으로 표시하지 않는다.

아래 순서는 이 글을 이해하기 위한 공부 순서다. ‘개념 준비’는 읽기 안내이며, 직접적인 연구 계보는 연결 칸에 따로 표시했다. 이미 해당 개념을 설명할 수 있다면 아는 부분은 넘어가도 된다.

| 순서·논문 | 연결·익힐 개념 | 먼저 볼 부분 |
|---|---|---|
| 1. [Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation](https://arxiv.org/abs/1602.05179) | **개념 준비**: 에너지·평형 상태·두 단계 학습 | 자유 상태와 목표로 살짝 유도한 상태의 비교 |

**현재 글과 연결하기:** 위 개념을 익힌 뒤, 이 글에서 ‘좋은 입력과 나쁜 입력을 층마다 구분하는 학습 이해하기’에 해당하는 부분을 읽자. 기존 방법의 어느 단계가 바뀌었는지 그림이나 입력·출력으로 설명해 보자.

[선수 논문 전체 지도와 원문 읽는 법]({{< relref "paper-prerequisites" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- PAPER-PREREQUISITES-END -->

<!-- LEARNING-START-END -->

---

## 논문 정보

- **제목**: The Forward-Forward Algorithm: Some Preliminary Investigations
- **저자**: Geoffrey Hinton
- **발표**: NeurIPS 2022 (2022.12)
- **키워드**: Local Learning, Contrastive Learning, Biologically Plausible

---

## 1. 동기: 왜 Backward Pass를 없애려 하는가?

역전파(BP)는 현대 딥러닝의 핵심이지만 근본적 문제가 있다:

1. **생물학적 비타당성**: 뇌에서 backward pass에 해당하는 메커니즘이 없다
2. **Weight transport 문제**: Forward와 backward에서 동일한 가중치를 사용해야 한다
3. **순차적 의존성**: Layer locking — 한 계층이 완료되어야 다음 계층 업데이트 가능
4. **전체 계산 그래프 필요**: Forward pass의 모든 활성값을 저장해야 한다

Hinton은 backward pass를 **아예 제거**해 이 문제들을 해결하려 한다.

---

## 2. 핵심 아이디어: Goodness 기반 대조 학습

### 기본 원리

각 레이어가 **독립적으로** 학습한다. Positive 데이터(실제 데이터)에서는 "goodness"를 높이고, negative 데이터(생성된 가짜 데이터)에서는 낮추는 것이 목표다.

### Goodness 정의

Hidden 활성값의 제곱합:

$$g = \sum_j h_j^2$$

여기서 $h_j$는 해당 레이어의 $j$번째 hidden unit 활성값이다.

### Layer-wise Loss

$$\mathcal{L} = \log\left(1 + e^{-(g_{pos} - \theta)}\right) + \log\left(1 + e^{g_{neg} - \theta}\right)$$

- $g_{pos}$: positive 데이터의 goodness
- $g_{neg}$: negative 데이터의 goodness
- $\theta$: 임계값 (positive/negative 분리 경계)

직관적으로:
- **Positive**: goodness $\gt  \theta$ → "이건 진짜다"
- **Negative**: goodness $\lt  \theta$ → "이건 가짜다"

goodness는 해당 레이어의 활성값으로 계산하는 점수다. Positive와 negative 입력을 구분하도록 각 레이어의 점수를 학습하므로, 학습 신호를 출력층에서 모든 레이어로 전달하는 과정을 바꾼다. 이때 negative 예제를 어떻게 만드는지도 학습 문제의 일부다. 아래 학습 절차는 두 forward pass와 레이어별 갱신이 어떤 순서로 연결되는지 보여준다.

---

## 3. 학습 과정

### Positive Pass

1. 실제 데이터 $x$를 입력
2. 각 레이어에서 활성값 계산
3. Goodness를 $\theta$ 이상으로 올리도록 학습

### Negative Pass

1. Negative 데이터 $\tilde{x}$ 생성 (예: 잘못된 레이블 합성)
2. 각 레이어에서 활성값 계산
3. Goodness를 $\theta$ 이하로 내리도록 학습

```
Positive: (image, correct_label) → 각 layer goodness ↑
Negative: (image, wrong_label)   → 각 layer goodness ↓
```

### Negative 데이터 생성 방법

Hinton이 제안한 방식 — **레이블을 이미지 첫 몇 픽셀에 임베딩**:

- Positive: 이미지 + 정답 레이블 → 첫 10 픽셀에 one-hot 인코딩
- Negative: 이미지 + 랜덤 오답 레이블 → 첫 10 픽셀에 잘못된 one-hot

추론 시: 모든 가능한 레이블에 대해 goodness 계산 → 가장 높은 것 선택

---

## 4. BP와의 근본적 차이

| 항목 | Backpropagation | Forward-Forward |
|------|----------------|-----------------|
| Pass 횟수 | 1 forward + 1 backward | 2 forward (pos + neg) |
| 학습 단위 | 전체 네트워크 (global) | **레이어별 독립** (local) |
| Error 전파 | 출력 → 입력 역방향 | **없음** |
| 활성값 저장 | 전체 필요 | 현재 레이어만 |
| 병렬화 | Layer locking | **레이어 독립 학습 가능** |
| 생물학적 타당성 | 낮음 | **높음** |

---

## 5. 알고리즘

```
for each mini-batch:
    # Positive phase
    x_pos = concat(image, correct_label_embedding)
    for layer l = 1 to L:
        h_l = activation(W_l · normalize(h_{l-1}))
        g_pos = sum(h_l^2)
        loss_pos = log(1 + exp(-(g_pos - θ)))
        W_l ← W_l - α · ∂loss_pos/∂W_l    # LOCAL gradient만 사용

    # Negative phase
    x_neg = concat(image, random_wrong_label_embedding)
    for layer l = 1 to L:
        h_l = activation(W_l · normalize(h_{l-1}))
        g_neg = sum(h_l^2)
        loss_neg = log(1 + exp(g_neg - θ))
        W_l ← W_l - α · ∂loss_neg/∂W_l    # LOCAL gradient만 사용
```

중요: 각 레이어의 gradient는 **해당 레이어 내에서만** 계산된다. 다른 레이어의 정보가 필요 없다.

---

## 6. 실험 결과

### MNIST 성능

| 방법 | Test Error |
|------|-----------|
| BP (softmax) | ~1.4% |
| Forward-Forward | ~1.4% |
| FF (unsupervised) | ~1.7% |

MNIST에서는 BP와 **동등한 성능**을 보인다.

### 한계

- **CIFAR-10 등 복잡한 데이터셋에서 BP 대비 성능 저하**
- Negative 샘플 생성 방식에 민감
- 레이블을 이미지에 임베딩하는 트릭이 부자연스러움

---

## 7. 변형과 후속 연구

### PEPITA (2022)

Forward-Forward의 한계를 개선한 변형. 두 번째 forward pass에서 perturbation을 추가하여 error 정보를 간접적으로 전달.

### CLAPP++ (EPFL/TU Wien, 2026)

가장 주목할 만한 후속 연구. Local-SSL(self-supervised learning)이 deep linear network에서 **BP-SSL과 동일한 weight update**를 수행할 수 있음을 이론적으로 증명했다.

### FF의 응용 가능성

- **RAG 임베딩의 triplet 구조**(query, relevant, irrelevant)에 자연스럽게 적용 가능
- 뉴로모픽 하드웨어에서 analog 연산과의 친화성
- 프라이버시: 활성값만 로컬에서 학습, 전체 gradient 전송 불필요

---

## 8. 자기회귀 언어 모델 적용의 한계

FF를 언어 모델에 적용하려면 근본적인 문제가 있다:

**Positive/Negative 정의가 모호하다**

- 이미지 분류: "이 이미지가 고양이인가?" → 명확한 positive/negative
- 자기회귀 LM: "다음 토큰이 무엇인가?" → 정답이 확률적이고, "좋은 문장"/"나쁜 문장"의 경계가 불분명

가능한 우회:
1. 정답 다음 토큰 = positive, 랜덤 토큰 = negative (NCE 방식)
2. 정답 문장 = positive, 셔플된 문장 = negative
3. Goodness를 확률 추정기로 변환

하지만 어느 방식도 표준 cross-entropy + softmax의 효율성에 미치지 못한다.

---

## 9. 의의와 평가

### 의의

1. **Backward pass 완전 제거**: 가장 급진적인 비역전파 접근 중 하나
2. **레이어 독립성**: 진정한 모듈러 학습 가능
3. **Hinton의 영향력**: 비역전파 연구에 대한 관심 환기

### 한계

1. **성능 격차**: 복잡한 데이터셋에서 BP 대비 열등
2. **Negative 샘플 의존**: 생성 방식이 성능에 큰 영향
3. **LM 적용 어려움**: 자기회귀 생성에서의 positive/negative 정의 모호
4. **확장성 미검증**: 대규모 모델에서의 실험 부족

### 개인적 생각

FF는 아이디어의 **단순함과 급진성**에서 가치가 있다. "Backward pass가 정말 필요한가?"라는 근본적 질문을 던진다. 실용적으로는 Predictive Coding(Prospective Configuration)이나 진화전략이 더 유망하지만, FF가 열어준 "local contrastive learning" 방향은 뉴로모픽 하드웨어 맥락에서 계속 발전할 가능성이 있다.

CLAPP++의 이론적 결과가 특히 흥미롭다 — local learning이 BP와 동일한 업데이트를 할 수 있다는 증명은, FF 계열의 한계가 **원리적**이 아니라 **구현적**임을 시사한다.

<!-- LEARNING-END -->
---

## 읽은 뒤 확인하고 다음으로

**확인 질문:** 좋은 입력과 나쁜 입력을 층마다 구분하는 학습 이해하기를 목표로, 본문에서 다룬 입력·중간 과정·결과를 자기 말로 설명해 보자. 실험이나 사례가 있다면 어떤 조건에서 확인했는지도 덧붙여 보자.

**다음 글:** [\[논문 리뷰\] Prospective Configuration: 역전파를 넘어서는 학습 — 먼저 상상하고, 그다음 굳힌다]({{< relref "posts/prospective-configuration-review" >}}) — 현재 상태를 먼저 조정하고 학습하는 순서 이해하기.

[전체 학습 순서]({{< relref "learn" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- LEARNING-END-END -->
