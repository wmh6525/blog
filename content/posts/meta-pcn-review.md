---
title: "[논문 리뷰] Meta-PCN: Meta Prediction Error를 활용한 안정적이고 확장 가능한 Deep Predictive Coding Networks"
date: 2026-03-19
tags: ["논문리뷰", "Predictive Coding", "비역전파", "뉴로모픽"]
categories: ["ML/AI"]
summary: "ICLR 2026에서 발표된 Meta-PCN 논문 리뷰. 깊은 Predictive Coding Network의 두 가지 근본적 병리(PE 불균형, EVPE)를 동적 평균장 이론으로 분석하고, meta-prediction error 기반 프레임워크로 해결한다."
math: true
toc: true
draft: false
---

<!-- LEARNING-START -->
## 처음 읽는 분을 위한 안내

**학습 위치:** [역전파를 바꾸면 무엇이 달라질까?]({{< relref "learn" >}}#no-bp) · 4/7

층마다 자기 예측의 오차로 학습해도 오차의 크기와 내용이 서로 다르면 문제가 생길 수 있다. 이 연구는 깊은 모델에서 그 차이를 분석하고 조정한다. 어떤 층에 어떤 신호가 전달되는지부터 따라가 보자.

**먼저 알아둘 말**

- [Predictive Coding·예측 오차]({{< relref "ai-glossary" >}}#pc): 각 층의 예측과 현재 상태의 차이를 이용해 상태와 가중치를 조정하는 계산 방식. 상태 조정과 가중치 학습의 순서를 구분해 본다.
- [손실 함수]({{< relref "ai-glossary" >}}#loss): 예측이 목표에서 얼마나 벗어났는지 계산하는 기준. 학습에서는 보통 이 값을 줄이려 한다.
- [상태·SSM]({{< relref "ai-glossary" >}}#state): 상태는 이전까지 읽은 정보를 다음 단계에 전달하는 내부 값이다. SSM은 이 상태가 입력에 따라 바뀌고 출력을 만드는 관계를 모델링한다.

**이번 글의 목표:** 층마다 예측 오차가 다를 때 생기는 문제 이해하기.

**이 학습 경로의 앞 글:** [\[논문 리뷰\] Prospective Configuration: 역전파를 넘어서는 학습 — 먼저 상상하고, 그다음 굳힌다]({{< relref "posts/prospective-configuration-review" >}}).

**본문에서 먼저 볼 순서:** **1. 배경: Predictive Coding Network란?** → **2. 핵심 기여: 두 가지 근본적 병리의 발견** → **4. 실험 결과**.

첫 독서에서는 문제와 방법의 흐름을 잡자. 수식·코드·세부 성능표는 두 번째에 읽어도 된다. 구현과 수학적 주장까지 확인하려면 그 부분을 다시 살펴봐야 한다.

<!-- PAPER-PREREQUISITES -->
### 선수 개념과 읽을 논문

**먼저 이해할 개념:** 손실 함수, 미분·연쇄법칙, 가중치와 활성값의 구분; 에너지 기반 방법은 최적화와 수렴.

아래 순서는 이 글을 이해하기 위한 공부 순서다. ‘개념 준비’는 읽기 안내이며, 직접적인 연구 계보는 연결 칸에 따로 표시했다. 이미 해당 개념을 설명할 수 있다면 아는 부분은 넘어가도 된다.

| 순서·논문 | 연결·익힐 개념 | 먼저 볼 부분 |
|---|---|---|
| 1. [Inferring neural activity before plasticity as a foundation for learning beyond backpropagation](https://doi.org/10.1038/s41593-023-01514-1) · [블로그 해설]({{< relref "posts/prospective-configuration-review" >}}) | **개념 준비**: 상태 최적화·가중치 학습·예측 오차 | 활성 상태를 먼저 조정하고 가중치를 나중에 바꾸는 순서 |
| 2. [Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation](https://arxiv.org/abs/1602.05179) | **개념 준비**: 에너지·평형 상태·두 단계 학습 | 자유 상태와 목표로 살짝 유도한 상태의 비교 |

**현재 글과 연결하기:** 위 개념을 익힌 뒤, 이 글에서 ‘층마다 예측 오차가 다를 때 생기는 문제 이해하기’에 해당하는 부분을 읽자. 기존 방법의 어느 단계가 바뀌었는지 그림이나 입력·출력으로 설명해 보자.

[선수 논문 전체 지도와 원문 읽는 법]({{< relref "paper-prerequisites" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- PAPER-PREREQUISITES-END -->

<!-- LEARNING-START-END -->

---

## 논문 정보

- **제목**: Stable and Scalable Deep Predictive Coding Networks with Meta Prediction Errors
- **저자**: Myoung Hoon Ha, Hyunjun Kim, Yoondo Sung, Youngha Jo, Min S. Kang, Sang Wan Lee
- **소속**: KAIST, 서울대학교, LG CNS
- **학회**: ICLR 2026
- **키워드**: Predictive Coding, Local Learning, Neuromorphic Computing, Mean-Field Theory

---

## 1. 배경: Predictive Coding Network란?

Predictive Coding(PC)은 뇌가 외부 환경에 대한 **예측을 지속적으로 생성**하고 **예측 오차(Prediction Error, PE)를 최소화**하여 내부 표현을 정제한다는 이론이다. PCN은 이를 신경망 아키텍처로 구현한 것으로, 역전파 대신 **순수 local learning rule**을 사용한다.

### PCN의 핵심 구조

각 계층 $l$에는 latent state $\mathbf{z}_l$이 있다. 순방향 예측 함수는 이 상태로 다음 계층을 예측한다.

$$\hat{\mathbf{z}}_{l+1} = f_l(\mathbf{z}_l) = \phi(W_l \cdot \mathbf{z}_l + b_l)$$

PE는 $\delta_l = \mathbf{z}_l - \hat{\mathbf{z}}_l$이며, 전체 목적 함수(자유 에너지)는:

$$\mathcal{F} = \frac{1}{2} \sum_{l=2}^{L} \lVert \delta_l \rVert_2^2$$

PCN은 **추론 단계**(latent state 업데이트)와 **학습 단계**(가중치 업데이트)를 번갈아 수행한다.

### PCN의 장점과 한계

**장점:**
- 생물학적으로 타당한 local learning rule
- 대규모 병렬화 가능 → 뉴로모픽 컴퓨팅에 유망

**치명적 한계:**
- 네트워크 깊이가 증가하면 학습이 **점진적으로 불안정**해진다
- 이 불안정성의 근본 메커니즘이 제대로 이해되지 않았다

---

## 2. 핵심 기여: 두 가지 근본적 병리의 발견

이 논문의 가장 큰 기여는 **동적 평균장 이론(dynamical mean-field theory)**을 사용해 깊은 PCN의 불안정성을 야기하는 **두 가지 근본 병리**를 수학적으로 규명한 것이다.

### 병리 1: PE 불균형 (Imbalanced Prediction Errors)

PE가 네트워크 경계(입력/출력 계층)에 집중되고 **중간 계층에서 소실**되는 현상이다.

- 정보 전파 속도가 $O(\nu^k)$ ($\nu = \eta\sigma_w$)로 지수적 감쇠
- 결과적으로 **U자형 오차 프로파일** 형성
- 중간 계층에서 $\delta_{l+1} \approx 0$이면 $\nabla_{W_l}\mathcal{F} \approx 0$ → **Gradient Starvation**

여기서 **PE 딜레마**가 발생한다: PE를 최소화하는 것이 목표인데, PE가 0에 가까워지면 학습 신호 자체가 사라진다.

### 병리 2: EVPE (Exploding and Vanishing Prediction Errors)

추론 과정에서 latent state와 PE가 **지수적으로 성장하거나 감쇠**하는 현상이다.

$$\lVert \delta_l^{t+1} \rVert \approx \tau_t(\sigma_w) \lVert \delta_l^t \rVert$$

- $\tau_t \gt  1$: 기하급수적 성장 (폭발)
- $\tau_t \lt  1$: 지수적 감쇠 (소실)
- 안정 영역($\tau_t \approx 1$)은 네트워크가 깊어질수록 **기하급수적으로 축소**

> 기존 역전파의 gradient vanishing/exploding과 다른 점: EVPE는 **파라미터 업데이트 이전**, 추론 단계 자체에서 발생한다.

각 층이 예측 오차를 사용하더라도 오차의 규모와 정보량이 층마다 같지는 않을 수 있다. 깊은 PCN에서는 이 차이가 상태 조정과 가중치 학습에 함께 영향을 준다. 아래 PE 불균형과 EVPE는 그 실패를 서로 다른 관점에서 살펴본다. Meta-PCN의 개입이 오차의 크기를 조절하는지, 학습에 쓰이는 오차 정보를 바꾸는지 나눠 읽으면 각 장치의 역할이 드러난다.

---

## 3. 해결책: Meta-PCN 프레임워크

### 3.1 Meta Prediction Error 기반 손실 함수

핵심 아이디어: **PE의 PE**를 최소화한다.

순방향 예측을 초기 값으로 고정하고($\hat{\mathbf{z}}_l^{(t)} = c_l$), 오차 $\tilde{\delta}_l := \mathbf{z}_l - c_l$에 대해 새로운 목적 함수를 정의한다:

$$J(\tilde{\delta}) = \frac{1}{2} \sum_{l=2}^{L-1} \lVert \tilde{\delta}_l - g_l(\tilde{\delta}^{\ast}_{l+1}, h_{l+1}^{(0)}) \rVert_2^2$$

이것은 비선형 평형 시스템을 **선형화**하는 효과가 있다. 선형화된 정상 맵의 잔차를 0으로 만들어, PC 추론 동역학을 근본적으로 재구성한다.

**이 접근이 해결하는 문제:**
1. PE를 직접 최소화하여 gradient starvation을 야기하는 대신, **PE가 계층 간 균형 잡힌 전파**를 하도록 유도
2. 순방향 예측을 고정하여 **학습-테스트 불일치 완화**

### 3.2 가중치 정규화

랜덤 행렬 이론을 활용한 분산 기반 정규화:

$$W \leftarrow \frac{W}{(\sqrt{m} + \sqrt{n})\sigma_w}$$

이는 스펙트럼 노름 $\lVert W \rVert_2 \approx 1$을 보장하여:
- EVPE의 곱셈 스케일링 인자 $\tau_t(\sigma_w)$를 제어
- PE 불균형의 기하급수적 감쇠 패턴 방지
- 추가 하이퍼파라미터 불필요

---

## 4. 실험 결과

### 병리 해결 검증

| 항목 | 기존 PCN | Meta-PCN |
|------|---------|----------|
| PE 분포 | U자형 (경계 집중) | **균형 잡힌 분포** |
| 추론 안정성 | 지수적 성장/감쇠 | **안정적 궤적** |
| 수렴 속도 | 느린 수렴 | **빠르고 확정적** |

### 분류 성능 (CIFAR-10)

| 아키텍처 | BP | PCN | Meta-PCN |
|---------|-----|-----|----------|
| VGG-5 | ~87% | ~10% | **~89%** |
| VGG-13 | ~88% | ~10% | **~90%** |
| ResNet-18 | ~90% | ~10% | **~90%** |

핵심 결과:
- 기존 PCN은 깊이 증가 시 **10~20% 정확도**로 붕괴
- Meta-PCN은 모든 깊이에서 **80~90% 유지**
- 대부분의 구성에서 **역전파를 능가** (CIFAR-10에서 0.61%~1.73% 개선)
- **Local learning rule을 보존**하면서 달성

### Ablation Study

- Meta-PE 손실 제거 → 정확도 89.5% → **10.0%** (핵심 요소)
- 가중치 정규화 제거 → 1.3%p 감소 (보조 요소)

---

## 5. 의의와 한계

### 의의

1. **이론적 기여**: 동적 평균장 이론으로 PCN 불안정성의 근본 원인을 최초로 수학적 규명
2. **실용적 기여**: 역전파를 능가하는 local learning 프레임워크 제시
3. **뉴로모픽 관점**: 생물학적 타당성을 유지하면서 실용적 성능 달성

### 한계와 향후 과제

- 실험이 **이미지 분류(CIFAR, TinyImageNet)**에 한정 — 시퀀스/언어 모델로의 확장 필요
- 이론 분석의 가정(i.i.d. 가우시안 가중치, 선형화, 균일 계층 폭)과 실제 아키텍처 간 간극
- 현대 대규모 모델(Transformer 등)에서의 검증 부재

---

## 6. 개인적 생각

이 논문은 "역전파 없이 학습"이라는 도전적 주제에서 **이론과 실험이 잘 맞물린 연구**다. 특히:

- **PE의 PE를 최소화한다**는 meta-level 접근이 깔끔하다
- 동적 평균장 이론을 PCN에 적용한 분석이 체계적이다
- 역전파를 능가하는 결과는, local learning이 뉴로모픽 하드웨어에서 실용화될 가능성을 보여준다

다만, 이미지 분류를 넘어 **언어 모델이나 생성 모델**에서의 검증이 향후 핵심 과제가 될 것이다.

<!-- LEARNING-END -->
---

## 읽은 뒤 확인하고 다음으로

**확인 질문:** 층마다 예측 오차가 다를 때 생기는 문제 이해하기를 목표로, 본문에서 다룬 입력·중간 과정·결과를 자기 말로 설명해 보자. 실험이나 사례가 있다면 어떤 조건에서 확인했는지도 덧붙여 보자.

**다음 글:** [\[서베이\] 역전파 없이 언어 모델을 학습하는 연구 동향과 사례]({{< relref "posts/no-bp-lm-survey" >}}) — 역전파 대신 무엇을 계산하고 저장하는지 비교하기.

[전체 학습 순서]({{< relref "learn" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- LEARNING-END-END -->
