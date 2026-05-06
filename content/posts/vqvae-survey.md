---
title: "[서베이] 오토인코더에서 VQ-VAE까지 — 이산 표현 학습의 계보"
date: 2026-03-24
tags: ["논문리뷰", "VQ-VAE", "오토인코더", "생성모델", "VAE"]
categories: ["ML/AI"]
summary: "오토인코더(AE)부터 VQ-VAE, 그리고 최신 FSQ까지의 계보를 정리한다. 각 모델의 핵심 수식, 등장 동기, 특징과 한계, 그리고 다음 모델로의 발전 이유를 서베이 형식으로 추적한다."
math: true
toc: true
draft: false
---

## 계보 개관

```
AE (1986) → DAE (2008) → VAE (2013) → VQ-VAE (2017)
                                          ↓
                              VQ-VAE-2 (2019) → dVAE/DALL-E (2021)
                                          ↓
                              RQ-VAE (2022) → FSQ (2023)
```

---

## 1. Autoencoder (AE) — 차원 축소의 신경망 버전

### 등장 배경

PCA 같은 선형 차원 축소를 **비선형으로 확장**하려는 시도. 입력을 저차원 병목(bottleneck)으로 압축하고 다시 복원한다.

### 구조

$$z = f_\theta(x) \quad \text{(인코더)}$$

$$\hat{x} = g_\phi(z) \quad \text{(디코더)}$$

### 손실 함수

$$\mathcal{L}_{AE} = \lVert x - \hat{x} \rVert^2$$

### 특징

- 결정론적(deterministic) — 입력 $x$에 대해 $z$가 하나로 결정
- 잠재 공간에 구조가 없음 → **생성 모델로 사용 불가**
- $z$ 공간에서 보간(interpolation)하면 의미 없는 결과

### 한계 → VAE로의 동기

잠재 공간이 불연속적이고 구멍이 많아서, 임의의 $z$를 디코딩하면 쓸모없는 출력이 나온다. "잠재 공간에 구조를 부여"할 필요가 있다.

---

## 2. Denoising Autoencoder (DAE, 2008)

### 등장 동기

AE는 항등 함수를 학습할 위험이 있다. **입력에 노이즈를 추가**하여 더 로버스트한 표현을 학습시킨다.

### 구조

$$\tilde{x} = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

$$\mathcal{L}_{DAE} = \lVert x - g_\phi(f_\theta(\tilde{x})) \rVert^2$$

노이즈가 낀 입력에서 원본을 복원하려면, 데이터의 **본질적 구조**를 학습해야 한다.

### 의의

- Score matching과의 이론적 연결 (Vincent 2011)
- 이후 Diffusion Model의 기반이 됨 (DDPM이 사실상 다단계 DAE)

---

## 3. Variational Autoencoder (VAE, 2013)

### 등장 동기

AE의 잠재 공간에 **확률적 구조**를 부여하여 **생성 모델**로 사용하고 싶다.

### 핵심 아이디어

인코더가 점 하나가 아닌 **분포**를 출력한다:

$$q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$$

### 목적 함수: ELBO

$$\mathcal{L}_{VAE} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{재구성 항}} - \underbrace{D_{KL}(q_\phi(z|x) \lVert p(z))}_{\text{정규화 항}}$$

- **재구성 항**: 디코더가 $z$에서 $x$를 잘 복원하는지 (= MSE 또는 BCE)
- **KL 항**: 인코더 분포 $q(z \mid x)$가 사전 분포 $p(z) = \mathcal{N}(0, I)$에 가까운지

### Reparameterization Trick

$z \sim \mathcal{N}(\mu, \sigma^2)$에서 직접 샘플링하면 역전파 불가. 트릭:

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

$\epsilon$은 고정 노이즈이므로 $\mu$, $\sigma$에 대해 역전파 가능.

### 특징

- 잠재 공간이 **연속적이고 매끄러움** → 보간 가능
- 임의의 $z \sim \mathcal{N}(0, I)$를 디코딩하면 합리적 출력
- **생성 가능**: 사전 분포에서 샘플링 → 디코딩

### 한계 → VQ-VAE로의 동기

1. **Posterior Collapse**: KL 항이 너무 강하면 디코더가 $z$를 무시하고 독립적으로 생성 → $q(z \mid x) \approx p(z)$가 되어 인코더가 무의미해짐
2. **흐릿한 출력**: 연속 잠재 공간 + MSE 손실 → 여러 가능한 출력의 평균을 생성 → 블러
3. **모드 커버리지 vs 품질 트레이드오프**: 다양성을 높이면 품질 저하

---

## 4. VQ-VAE (2017) — 핵심 모델

### 논문 정보

- **제목**: Neural Discrete Representation Learning
- **저자**: Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu (DeepMind)
- **학회**: NeurIPS 2017

### 등장 동기

VAE의 연속 잠재 공간 대신 **이산 잠재 공간**을 사용하면:
- Posterior collapse 방지 (이산 코드는 무시하기 어려움)
- 강력한 자기회귀 사전분포 학습 가능
- 언어 같은 이산 구조와 자연스러운 결합

### 핵심 구조

1. **인코더**: 입력 $x$를 연속 벡터 $z_e(x)$로 인코딩
2. **양자화**: 코드북 $\lbrace e_1, e_2, \ldots, e_K \rbrace$에서 가장 가까운 벡터 선택
3. **디코더**: 선택된 코드북 벡터 $e_k$를 디코딩

### 양자화 (Quantization)

$$z_q(x) = e_k, \quad \text{where } k = \arg\min_j \lVert z_e(x) - e_j \rVert_2$$

인코더 출력 $z_e(x)$와 가장 가까운 코드북 벡터 $e_k$를 찾아 대체한다.

### 손실 함수

$$\mathcal{L}_{VQ} = \underbrace{\lVert x - \hat{x} \rVert^2}_{\text{재구성}} + \underbrace{\lVert \text{sg}[z_e(x)] - e_k \rVert^2}_{\text{코드북 학습}} + \beta \underbrace{\lVert z_e(x) - \text{sg}[e_k] \rVert^2}_{\text{커밋먼트}}$$

- **재구성 손실**: 디코더가 원본을 잘 복원하는지
- **코드북 손실**: 코드북 벡터 $e_k$를 인코더 출력 쪽으로 이동 (VQ 학습)
- **커밋먼트 손실**: 인코더 출력을 코드북 벡터 쪽으로 이동 (인코더가 코드북에 "커밋"하도록)

$\text{sg}[\cdot]$는 **stop-gradient** 연산자 — 해당 항의 그래디언트를 차단한다.

### Straight-Through Estimator (STE)

$\arg\min$은 미분 불가능하다. 순방향에서는 양자화된 $z_q$를 사용하고, 역방향에서는 **그래디언트를 인코더로 그대로 복사**한다:

$$\text{Forward}: z_q = e_k$$

$$\text{Backward}: \frac{\partial \mathcal{L}}{\partial z_e} \approx \frac{\partial \mathcal{L}}{\partial z_q}$$

### EMA 코드북 업데이트 (대안)

코드북 손실 대신 **Exponential Moving Average**로 코드북을 업데이트하는 방법도 널리 사용된다:

$$N_k^{(t)} = \gamma N_k^{(t-1)} + (1-\gamma) n_k^{(t)}$$

$$m_k^{(t)} = \gamma m_k^{(t-1)} + (1-\gamma) \sum_{i: q(z_i)=k} z_{e,i}$$

$$e_k^{(t)} = \frac{m_k^{(t)}}{N_k^{(t)}}$$

$n_k^{(t)}$는 현재 배치에서 코드 $k$에 할당된 벡터 수. 실제로 이 방식이 더 안정적이다.

### 사전 분포 학습

VQ-VAE 학습 후, 이산 코드 시퀀스에 대해 **자기회귀 모델**(PixelCNN, Transformer 등)을 학습한다:

$$p(z_1, z_2, \ldots, z_T) = \prod_t p(z_t \mid z_{<t})$$

이를 통해 새로운 코드 시퀀스를 생성하고 디코딩하면 새로운 데이터를 생성할 수 있다.

### VAE vs VQ-VAE 비교

| 항목 | VAE | VQ-VAE |
|------|-----|--------|
| 잠재 공간 | 연속 ($\mathbb{R}^d$) | **이산** ($\lbrace 1, \ldots, K \rbrace$) |
| 인코더 출력 | $\mu, \sigma$ (분포) | $z_e$ (점 벡터) |
| 정규화 | KL divergence | 코드북 + 커밋먼트 손실 |
| Posterior collapse | 발생 가능 | **방지됨** |
| 생성 방식 | $z \sim \mathcal{N}(0,I)$ → 디코더 | 사전 분포 → 코드 시퀀스 → 디코더 |
| 출력 품질 | 흐릿함 | **선명함** |

---

## 5. VQ-VAE-2 (2019) — 계층적 이산 표현

### 논문 정보

- **제목**: Generating Diverse High-Fidelity Images with VQ-VAE-2
- **저자**: Ali Razavi, Aaron van den Oord, Oriol Vinyals (DeepMind)
- **학회**: NeurIPS 2019

### 동기

VQ-VAE는 단일 해상도의 코드맵을 사용한다. **다중 해상도 계층 구조**로 표현력을 높인다.

### 구조

$$z_{top} \in \lbrace 1, \ldots, K \rbrace^{H_1 \times W_1} \quad \text{(전역 구조: 레이아웃, 형태)}$$

$$z_{bottom} \in \lbrace 1, \ldots, K \rbrace^{H_2 \times W_2} \quad \text{(로컬 디테일: 질감, 색상)}$$

- Top 코드맵: 작은 해상도, 전체적인 구조 담당
- Bottom 코드맵: 큰 해상도, 세부 디테일 담당 (top에 조건부)

### 사전 분포

계층적 자기회귀 모델:

$$p(z_{top}) = \prod_i p(z_{top,i} \mid z_{top,<i})$$

$$p(z_{bottom} \mid z_{top}) = \prod_j p(z_{bottom,j} \mid z_{bottom,<j}, z_{top})$$

### 결과

256×256 얼굴 이미지에서 FID 10 이하 — GAN에 근접하는 품질을 자기회귀 모델로 달성.

---

## 6. dVAE / DALL-E (2021) — 텍스트-이미지 생성

### 논문 정보

- **제목**: Zero-Shot Text-to-Image Generation
- **저자**: Aditya Ramesh et al. (OpenAI)

### dVAE란?

VQ-VAE의 변형으로, **Gumbel-Softmax**를 사용하여 양자화를 미분 가능하게 만든 것:

$$z = \text{Gumbel-Softmax}(\text{logits}, \tau)$$

Straight-Through Estimator 대신 **온도 $\tau$를 점진적으로 낮추는** annealing으로 이산 샘플에 수렴.

### DALL-E 파이프라인

1. **Stage 1**: dVAE로 이미지 → 32×32 이산 코드 (8192 코드북)
2. **Stage 2**: 텍스트 토큰 + 이미지 코드를 이어붙여 자기회귀 Transformer로 학습

$$p(\text{image codes} \mid \text{text tokens}) = \prod_i p(z_i \mid z_{<i}, \text{text})$$

### 의의

이미지를 **이산 토큰으로 변환**하여 언어 모델과 동일한 프레임워크로 처리할 수 있게 만들었다.

---

## 7. RQ-VAE (2022) — 잔차 양자화

### 논문 정보

- **제목**: Autoregressive Image Generation using Residual Quantization
- **저자**: Doyup Lee et al. (Kakao Brain)
- **학회**: CVPR 2022

### 동기

VQ-VAE의 단일 양자화는 코드북 크기 $K$에 의해 표현력이 제한된다. $K$를 키우면 코드북 활용률이 떨어진다 (codebook collapse).

### 핵심: 잔차 양자화 (Residual Quantization)

$$r_0 = z_e(x)$$

$$z_1 = \text{Quantize}(r_0), \quad r_1 = r_0 - z_1$$

$$z_2 = \text{Quantize}(r_1), \quad r_2 = r_1 - z_2$$

$$\vdots$$

$$z_D = \text{Quantize}(r_{D-1})$$

각 단계에서 **이전 양자화의 잔차(residual)**를 다시 양자화한다. $D$번 반복하면 총 $K^D$개의 표현이 가능하다.

### 장점

| 항목 | VQ-VAE | RQ-VAE |
|------|--------|--------|
| 유효 코드북 크기 | $K$ | $K^D$ |
| 코드 시퀀스 길이 | $H \times W$ | $H \times W \times D$ |
| 코드북 활용률 | 낮을 수 있음 | **높음** |
| 표현 정밀도 | 고정 | **단계적 정밀화** |

---

## 8. FSQ (2023) — Finite Scalar Quantization

### 논문 정보

- **제목**: Finite Scalar Quantization: VQ-VAE Made Simple
- **저자**: Fabian Mentzer, David Minnen, Eirikur Agustsson, Michael Tschannen (Google)
- **학회**: ICLR 2024

### 동기

VQ-VAE의 문제들:
- **Codebook collapse**: 대부분의 코드가 사용되지 않음
- **코드북 초기화** 민감성
- **EMA vs gradient** 업데이트 선택
- **커밋먼트 손실** 가중치 튜닝

### 핵심 아이디어: 벡터 양자화를 없앤다

각 차원을 독립적으로 **유한 수준으로 반올림**:

$$\hat{z}_i = \text{round}\left(\frac{L_i - 1}{2} \cdot \tanh(z_i)\right) \cdot \frac{2}{L_i - 1}$$

예를 들어 $d=6$ 차원, 각 차원이 $L = 5$ 수준이면:

$$\text{총 코드 수} = 5^6 = 15625$$

### VQ-VAE vs FSQ 비교

| 항목 | VQ-VAE | FSQ |
|------|--------|-----|
| 양자화 방식 | 최근접 이웃 탐색 | **독립 스칼라 반올림** |
| 코드북 | 학습 필요 ($K \times d$) | **불필요** |
| 보조 손실 | 코드북 + 커밋먼트 | **없음** |
| Codebook collapse | 발생 가능 | **원리적 불가** |
| 하이퍼파라미터 | $K$, $\beta$, EMA decay | $L$ (수준 수)만 |
| 그래디언트 | STE | STE (더 단순) |

### 결과

MaskGIT, UViM 등에서 VQ-VAE를 FSQ로 대체해도 **동등한 성능**, 코드북 활용률은 **100%**.

---

## 9. 전체 계보 비교

| 모델 | 연도 | 잠재 공간 | 핵심 기법 | 생성 방식 | 주요 한계 |
|------|------|----------|----------|----------|----------|
| **AE** | 1986 | 연속, 비구조적 | 병목 압축 | 불가 | 생성 불가 |
| **DAE** | 2008 | 연속 | 노이즈 + 복원 | 불가 | 생성 불가 |
| **VAE** | 2013 | 연속, 정규화 | ELBO + KL | $z \sim \mathcal{N}(0,I)$ | 흐릿함, posterior collapse |
| **VQ-VAE** | 2017 | **이산** | 벡터 양자화 + STE | 자기회귀 사전분포 | codebook collapse |
| **VQ-VAE-2** | 2019 | 이산, 계층적 | 다해상도 코드맵 | 계층적 자기회귀 | 학습 복잡도 |
| **dVAE** | 2021 | 이산 | Gumbel-Softmax | Transformer | 큰 코드북 필요 |
| **RQ-VAE** | 2022 | 이산, 잔차 | 잔차 양자화 $\times D$ | 자기회귀 | 코드 시퀀스 길어짐 |
| **FSQ** | 2023 | 이산 | **스칼라 반올림** | 다양 | 표현력 제한 가능 |

---

## 10. VQ-VAE의 현재 위치와 영향

### 영향을 받은 주요 모델

| 모델 | VQ-VAE 활용 방식 |
|------|----------------|
| **DALL-E** (OpenAI, 2021) | dVAE로 이미지 → 이산 토큰 → Transformer 생성 |
| **DALL-E 2** (2022) | CLIP + Diffusion이지만 dVAE 계보 |
| **Parti** (Google, 2022) | ViT-VQGAN으로 이미지 토큰화 |
| **MusicGen** (Meta, 2023) | RQ-VAE로 오디오 → 이산 코드 → Transformer |
| **SoundStorm** (Google, 2023) | SoundStream(RVQ) + 비자기회귀 생성 |
| **VideoPoet** (Google, 2023) | MAGVIT-v2(FSQ) 기반 비디오 토큰화 |

### 핵심 통찰

VQ-VAE 계열의 가장 큰 기여는 **"연속 데이터를 이산 토큰으로 변환"**하는 프레임워크를 확립한 것이다. 이를 통해:

1. **이미지/오디오/비디오를 언어처럼 취급** 가능
2. 강력한 **자기회귀 모델**(Transformer)을 비언어 도메인에 적용
3. **멀티모달 통합**: 텍스트 토큰 + 이미지 토큰 + 오디오 토큰을 하나의 시퀀스로

이것은 "모든 것을 토큰으로" (tokenize everything)라는 현대 AI의 핵심 패러다임과 직결된다.
