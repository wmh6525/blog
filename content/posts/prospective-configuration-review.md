---
title: "[논문 리뷰] Prospective Configuration: 역전파를 넘어서는 학습 — 먼저 상상하고, 그다음 굳힌다"
date: 2026-03-19
tags: ["논문리뷰", "Predictive Coding", "비역전파", "뉴로모픽"]
categories: ["ML/AI"]
summary: "Nature Neuroscience 2024 논문 리뷰. Prospective Configuration은 '활성값을 먼저 조정하고 가중치를 나중에 바꾸는' 학습 방식으로, 깊은 네트워크에서 역전파보다 안정적이고 continual learning에서 우위를 보인다."
math: true
toc: true
draft: false
---

## 논문 정보

- **제목**: Inferring Neural Activity Before Plasticity: A Foundation for Learning Beyond Backpropagation
- **저자**: Yuhang Song, Beren Millidge, Tommaso Salvatori, Thomas Lukasiewicz, Zhenghua Xu, Rafal Bogacz
- **게재**: Nature Neuroscience, Volume 27, February 2024
- **이론 논문**: "A Theoretical Framework for Inference and Learning in Predictive Coding Networks" (ICLR 2023)
- **코드**: [GitHub](https://github.com/YuhangSong/Prospective-Configuration)

---

## 1. 핵심 아이디어: 순서가 다르다

BP와 Prospective Configuration(PC)의 근본적 차이는 **학습의 순서**이다.

| 단계 | Backpropagation | Prospective Configuration |
|------|----------------|--------------------------|
| 1 | Forward pass (활성값 계산) | Forward pass + **target clamping** |
| 2 | Backward pass (gradient 계산) | **Inference relaxation** (모든 층 동시 조정) |
| 3 | Weight update (global gradient) | Weight update (**local Hebbian rule**) |

**BP**: 가중치를 먼저 바꾸고, 활성값이 따라간다.
**PC**: **활성값을 먼저 바꾸고**(prospective), 가중치가 따라간다.

비유하면, BP는 "일단 벽을 부수고 결과를 봄"이고, PC는 "먼저 완성된 모습을 상상하고, 그에 맞게 벽돌을 놓음"이다.

---

## 2. 수학적 구조

### 네트워크 설정

L-layer 네트워크에서 각 계층의 **예측값**:

$$\mu_l = w_{l-1} \cdot f(x_{l-1})$$

### 에너지 함수

전체 에너지는 각 계층의 **prediction error 제곱합**:

$$E(\mathbf{x}, \mathbf{w}) = \frac{1}{2} \sum_l \|\epsilon_l\|^2$$

여기서 prediction error: $\epsilon_l = x_l - \mu_l = x_l - w_{l-1} f(x_{l-1})$

### Phase 1: Inference (Neural Dynamics)

입출력을 고정(clamp)한 상태에서, **중간 계층 활성값을 에너지 경사 하강으로 반복 업데이트**한다:

$$\Delta x_l = -\gamma \frac{\partial E}{\partial x_l} = -\gamma \left( \epsilon_l - w_l^T (\epsilon_{l+1} \odot f'(w_l x_l)) \right)$$

이를 T step 반복하여 평형 상태 $x^* = \{x_0^*, x_1^*, \ldots, x_L^*\}$에 수렴한다.

핵심은 입력과 출력 **양쪽을 고정**하고, 중간 계층들이 **양방향 정보**를 받아 "학습 후의 활성 패턴"을 **미리** 찾아내는 것이다.

### Phase 2: Learning (Weight Plasticity)

평형 수렴 후, 가중치를 **한 번만** 업데이트:

$$\Delta w_l = \alpha \cdot \epsilon_{l+1}^* \odot f'(w_l x_l^*) \cdot (x_l^*)^T$$

이것은 **순수 local Hebbian rule**이다:
- pre-synaptic activity $x_l^*$와 post-synaptic prediction error $\epsilon_{l+1}^*$의 곱
- 각 시냅스가 자기 직전/직후 뉴런의 정보만으로 업데이트
- **global error signal이 불필요**

---

## 3. 왜 작동하는가? — Target Alignment

논문의 핵심 발견은 **Target Alignment** 개념이다.

$$\text{Target Alignment} = \cos(\theta)$$

여기서 $\theta$는 output이 target 방향으로 이동해야 하는 각도 vs 실제 weight update 후 이동하는 각도이다.

| | BP | Prospective Config |
|--|----|--------------------|
| 얕은 네트워크 | 높음 | 높음 |
| **깊은 네트워크 (15층)** | **급격히 하락** | **~1.0 유지** |

BP의 문제: 한 계층의 가중치를 바꾸면 다른 계층에 **간섭(interference)** 발생 → 의도한 방향과 다르게 이동한다.

PC의 해결: 모든 계층이 **동시에** 조정되므로, 가중치 변경이 **보상적(compensatory)** → 간섭이 최소화된다.

---

## 4. BP와의 이론적 관계

Whittington & Bogacz (2017)의 중요한 정리:

> **평형 상태에서 PC의 prediction error는 BP의 gradient와 동일하다**
>
> $$\epsilon_l^* = \delta_l^{BP} \quad \text{(at equilibrium, when } \Sigma = 1\text{)}$$

하지만 실제로는 유한 inference step에서 PC 업데이트가 BP와 **달라지며**, 이 차이가 오히려 장점이 된다:

1. 모든 계층의 정보를 **동시에** 반영 (BP는 순차적)
2. 계층 간 간섭을 보상
3. 암묵적 regularization 효과

---

## 5. 알고리즘 정리

```
Input: data x, label y, weights w, γ(inference rate), α(learning rate), T steps

1. Clamp: x_0 = x, x_L = y
2. Initialize hidden: x_l = forward pass 결과 (μ_l)

3. INFERENCE (t = 0 to T):
   for each hidden layer l = 1, ..., L-1:
     ε_l = x_l - w_{l-1} f(x_{l-1})
     x_l ← x_l - γ·(ε_l - w_lᵀ(ε_{l+1} ⊙ f'(w_l x_l)))
   if energy E 수렴 → break

4. LEARNING (수렴 후 1회):
   for each layer l = 0, ..., L-1:
     Δw_l = α · ε*_{l+1} ⊙ f'(w_l x*_l) · (x*_l)ᵀ
     w_l ← w_l + Δw_l
```

**Incremental 변형 (iPC)**: inference loop 내에서 매 step weight update도 동시 수행하며, 수렴이 보장된다.

---

## 6. 실험 결과

FashionMNIST 기준, fully connected network:

| 시나리오 | BP | Prospective Config | 승자 |
|---------|-----|-------------------|------|
| 표준 학습 | ~0.12 error | ~0.12 error | 동등 |
| **Deep (15층)** | 느린 수렴 | **빠른 수렴** | PC |
| **Online (batch=1)** | 성능 급락 | **유지** | PC |
| **Continual learning** | catastrophic forgetting | **완화** | PC |
| **Concept drift** | 적응 느림 | **빠른 적응** | PC |
| **적은 데이터** | 과적합 | **더 나은 일반화** | PC |
| **강화학습** (Acrobot 등) | 보통 | **더 높은 보상** | PC |

핵심: 표준 설정에서는 동등하지만, **생물학적으로 현실적인 조건**(online, continual, few-shot)에서 PC가 우위를 보인다.

---

## 7. 관련 모델과의 관계

| 모델 | Prospective Configuration과의 관계 |
|------|----------------------------------|
| **PCN** | PC의 주요 구현체. 2-phase 학습이 곧 prospective configuration |
| **Hopfield Networks** | 동일 원리 — relaxation 후 학습 |
| **Boltzmann Machines** | 평형 도달 후 weight update |
| **Generalized EM** | Inference = E-step, Weight update = M-step |
| **Target Propagation** | PC 평형 활성값이 TP의 "target"과 유사한 역할 |

---

## 8. Feedback Alignment과의 비교

| 항목 | Feedback Alignment | Prospective Configuration |
|------|-------------------|--------------------------|
| BP 제거 범위 | backward path만 ($W^T$ → random $B$) | forward + backward 모두 local |
| Error 전파 | global error를 random matrix로 전달 | local prediction error (계층별) |
| Weight update | 여전히 global error 필요 | **순수 local Hebbian** |
| 생물학적 타당성 | 중간 | **높음** |
| 성능 | BP와 거의 동등 | 표준: 동등, online/continual: **BP 초과** |

---

## 9. 의의와 한계

### 의의

1. **패러다임 전환**: "가중치 먼저" → "활성값 먼저"라는 근본적으로 다른 학습 순서
2. **깊은 네트워크 안정성**: Target Alignment가 깊이에 관계없이 유지
3. **생물학적 타당성**: 순수 local Hebbian rule로 학습
4. **실용적 우위**: Continual learning, online learning, few-shot에서 BP 초과

### 한계

- 표준 i.i.d. 학습에서는 BP와 **동등** (압도적 우위 아님)
- Inference 반복이 필요하여 **학습 속도 느림**
- 대규모 모델(Transformer, LLM)에서의 검증 부재
- 수렴 보장이 이론적으로 특정 조건에 한정

---

## 10. 개인적 생각

Prospective Configuration의 "먼저 상상하고, 그다음 굳힌다"는 아이디어는 직관적으로 매력적이다. Meta-PCN과 함께 읽으면, Predictive Coding 계열이 깊은 네트워크 학습에서 역전파의 근본적 한계를 어떻게 우회하는지 그림이 그려진다.

특히 **continual learning에서의 우위**는 실용적으로 중요하다. 실제 세계에서 데이터는 i.i.d.가 아니며, 새로운 정보를 기존 지식과 통합하는 능력은 AGI로 가는 핵심 요소다. PC가 이 문제에서 구조적 이점을 가진다는 것은, 단순한 "BP 대안" 이상의 의미를 가진다.
