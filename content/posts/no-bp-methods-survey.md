---
title: "역전파 없이 학습할 수 있을까? — Forward-Forward부터 EBT까지"
date: 2026-03-16
tags: ["논문리뷰", "Forward-Forward", "No-BP", "Predictive-Coding", "EBT"]
categories: ["ML/AI"]
summary: "Hinton의 Forward-Forward, MeZO, Predictive Coding, Evolution Strategies, Energy-Based Transformers까지 — BP 대안 학습 방법론들을 정리하고, 각각의 LLM 적용 가능성을 검토한다."
math: true
toc: true
---

## 왜 역전파의 대안을 찾는가

역전파(backpropagation)는 딥러닝의 핵심이지만 몇 가지 한계가 있다:
- **생물학적 비현실성**: 뇌의 뉴런은 전역 에러 신호를 역방향으로 전달하지 않는다
- **메모리 비용**: 전체 활성값을 저장해야 해서 대규모 모델에서 메모리 병목
- **순차 의존성**: Forward → Backward 순서가 고정되어 파이프라인 병렬화에 제약

이런 한계를 극복하려는 다양한 접근을 조사했다.

---

## 1. Forward-Forward Algorithm (Hinton, 2022)

### 핵심 아이디어

BP의 forward + backward 두 패스를 **positive forward + negative forward** 두 패스로 대체한다.

- **Positive pass**: 실제 데이터를 입력. 각 레이어가 "goodness"를 최대화
- **Negative pass**: 오염된 데이터를 입력. 각 레이어가 "goodness"를 최소화

각 레이어의 goodness:

$$g = \sum_j h_j^2 \quad \text{(hidden 활성값의 제곱합)}$$

레이어별 독립 loss:

$$\mathcal{L}_{layer} = \log(1 + e^{-(g_{pos} - \theta)}) + \log(1 + e^{g_{neg} - \theta})$$

### Positive/Negative 샘플 생성

MNIST 예시에서 이미지의 처음 10픽셀에 레이블의 one-hot 인코딩을 삽입한다:
- Positive: 이미지 + **정답** 레이블 → goodness 높이기
- Negative: 이미지 + **오답** 레이블 → goodness 낮추기

### LM 적용 한계

자기회귀 생성에서 "The cat sat on the ___"의 다음 토큰은 "mat", "floor", "ground" 등 여러 정답이 확률적으로 존재한다. 이미지의 정답/오답처럼 깔끔한 이진 구분이 어렵다.

다만 **판별 태스크**에는 적용 가능:
- 감성분석, NLI: 자연스러운 positive/negative
- RAG 임베딩: (query, relevant_doc, irrelevant_doc) triplet 구조

### 관련 최신 논문

- **CFF** (Contrastive FF): ViT에 적용, supervised contrastive learning으로 5-20x 빠른 수렴
- **SCFF** (Self-Contrastive FF): Nature Communications 2025, CIFAR-10/STL-10 확장
- **CLAPP++** (EPFL, 2026.01): local-SSL이 deep linear network에서 BP-SSL과 동일한 weight update 가능 증명

---

## 2. MeZO — Zeroth-Order Optimizer

Princeton NLP, NeurIPS 2023. Forward pass만으로 gradient를 추정한다.

$$\hat{g} = \frac{f(w + \epsilon z) - f(w - \epsilon z)}{2\epsilon} \cdot z$$

여기서 $z$는 랜덤 perturbation 벡터.

- 12x 메모리 절감 (활성값 저장 불필요)
- 30B 모델을 단일 A100에서 파인튜닝 가능
- 단, **사전학습에는 부적합** — gradient 추정 분산이 너무 큼

---

## 3. Predictive Coding

뇌의 예측 메커니즘을 모방한 학습법. 각 레이어가 하위 레이어의 활성값을 예측하고, 예측 오차만 상위로 전달한다.

- Local energy minimization으로 학습
- **Prospective Configuration**: PC의 자연스러운 모드. 뉴럴 활동이 먼저 변하고, 가중치가 나중에 정착
- 구현상 `backward()`를 사용하더라도 실제로는 local gradient만 계산
- 한계: 소규모 CNN을 넘어서는 스케일링 미검증

---

## 4. Evolution Strategies

완전한 BP-free. 모집단의 perturbation으로 gradient를 추정하는 방식이다.

$$\nabla_w J \approx \frac{1}{n\sigma} \sum_{i=1}^{n} f(w + \sigma \epsilon_i) \cdot \epsilon_i$$

MeZO는 사실상 population size=1인 ES이다. 최근에는 RLHF/PPO의 대안으로 LLM 파인튜닝에 성공한 사례가 있다. 고도로 병렬화 가능하다는 장점이 크다.

---

## 5. Energy-Based Transformers (EBT)

(arXiv:2507.02092, 2025.07)

(context, prediction) 쌍에 에너지를 할당하고, 반복적 에너지 최소화 = "thinking"이라는 해석이 흥미롭다.

- Transformer++ 대비 **35% 더 나은 스케일링**
- System 2 thinking에서 **29% 향상**
- BP를 사용하지만 목적이 다름: 추론 시 반복적 에너지 최소화

---

## 정리

| 방법론 | BP 사용 | 핵심 장점 | LLM 사전학습 적합성 |
|--------|---------|----------|-------------------|
| Forward-Forward | 아니오 | 생물학적 현실성 | 낮음 (판별에만) |
| MeZO | 아니오 | 메모리 효율 | 파인튜닝만 |
| Predictive Coding | 제한적 | 뇌 유사 학습 | 미검증 |
| Evolution Strategies | 아니오 | 고병렬 | RLHF 대안 |
| EBT | 예 | 반복 추론 | 유망 |
| **KAN** | **예** | **해석 가능** | **아키텍처 혁신 (No-BP 아님)** |

현재 시점에서 BP를 완전히 대체하면서 LLM 사전학습에 적합한 방법론은 아직 없다. 다만 각 방법론의 아이디어(local loss, energy minimization, selective gating)를 SancMamba 설계에 참고했다.

---

*참고 논문:*
- Hinton (2022), "The Forward-Forward Algorithm"
- Malladi et al. (2023), "Fine-Tuning Language Models with Just Forward Passes" (MeZO)
- Rao & Ballard (1999), "Predictive coding in the visual cortex"
- arXiv:2601.21683, "Can Local Learning Match Self-Supervised Backprop?" (CLAPP++)
- arXiv:2507.02092, "Energy-Based Transformers"
