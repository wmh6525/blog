---
title: "[논문 리뷰] Hourglass Diffusion Transformers — 메가픽셀에서 DiT의 99% 연산을 절약하다"
date: 2026-03-27
tags: ["논문리뷰", "Diffusion", "Transformer", "Hourglass", "이미지생성"]
categories: ["ML/AI"]
summary: "HDiT 논문 상세 리뷰. Hourglass 패턴을 Diffusion Transformer에 적용하여 픽셀 공간에서 직접 고해상도 이미지를 생성한다. 1024x1024에서 DiT 대비 99% 이상의 FLOP 절감을 달성하면서, FFHQ-1024에서 diffusion SoTA FID 5.23을 기록한다."
math: true
toc: true
draft: false
---

## 논문 정보

- **제목**: Scalable High-Resolution Pixel-Space Image Synthesis with Hourglass Diffusion Transformers
- **저자**: Katherine Crowson, Stefan Andreas Baumann, Alex Birch, Tanishq Mathew Abraham, Daniel Z. Kaplan, Enrico Shippole
- **소속**: Stability AI, LMU Munich, Birchlabs
- **학회**: ICML 2024
- **코드**: [crowsonkb/k-diffusion](https://github.com/crowsonkb/k-diffusion)

---

## 1. 핵심 문제: DiT는 고해상도에서 비현실적으로 비싸다

DiT(Diffusion Transformer)는 모든 패치에 global self-attention을 적용한다. 패치 수 $n$에 대해:

$$\text{DiT}: O(n^2 \cdot d)$$

해상도가 2배 → 패치 4배 → **연산 16배**. 256x256에서 657 GFLOPs, 512x512에서 **6,341 GFLOPs**.

### HDiT의 해법: Hourglass 패턴

Global attention을 **고정 크기의 최저 해상도(16x16)**에서만 수행하고, 나머지는 **O(n) neighborhood attention**을 사용한다.

$$\text{HDiT}: O(n \cdot d)$$

| 해상도 | DiT (GFLOPs) | HDiT (GFLOPs) | 절감 |
|--------|-------------|-------------|------|
| 128x128 | 106 | 31 | ~70% |
| 256x256 | 657 | 65 | **~90%** |
| 512x512 | 6,341 | 198 | **~97%** |
| 1024x1024 | ~50,000+ | ~수백 | **>99%** |

---

## 2. 아키텍처: 다중 해상도 계층

```
입력 이미지 (256x256)
    ↓ 패치화 (p=4) → 64x64 토큰
[Level 1] Neighborhood Attention × 2  (64x64, 384ch)
    ↓ Pixel-UnShuffle → 32x32 토큰
[Level 2] Neighborhood Attention × 2  (32x32, 768ch)
    ↓ Pixel-UnShuffle → 16x16 토큰
[Level 3] Global Attention × 16       (16x16, 1536ch)  ← 핵심 연산
    ↑ Pixel-Shuffle → 32x32
[Level 2'] Neighborhood Attention × 2  + Skip 연결
    ↑ Pixel-Shuffle → 64x64
[Level 1'] Neighborhood Attention × 2  + Skip 연결
    ↓
출력 (256x256)
```

**해상도가 높아지면**: 외부 레벨만 추가하면 된다. 내부(16x16 global attention)는 고정 비용.

### 토큰 병합/분할

- **다운샘플**: Pixel-UnShuffle로 $(B, H, W, C) \to (B, H/2, W/2, 4C)$ → Linear로 채널 조정
- **업샘플**: Linear로 $4C$ 채널로 확장 → Pixel-Shuffle로 $(B, 2H, 2W, C)$

### Attention 전략

| 레벨 | 해상도 | Attention 종류 | 복잡도 |
|------|-------|--------------|-------|
| 외부 (고해상도) | 64x64, 32x32 | **Neighborhood** (kernel=7) | $O(n)$ |
| 내부 (저해상도) | 16x16 = 256 토큰 | **Global** | $O(256^2)$ = 상수 |

Neighborhood attention이 Swin (Shifted Window)보다 유의미하게 좋다 (FID 51.07 vs 55.93).

---

## 3. Transformer 블록 설계

LLaMA에서 영감받은 설계:

### AdaRMSNorm 컨디셔닝

DiT의 AdaLN(scale + shift + gate)을 **AdaRMSNorm(scale만)**으로 단순화:

$$\text{AdaRMSNorm}(x, \gamma) = \gamma \cdot \frac{x}{\text{RMS}(x)}$$

매핑 네트워크가 timestep/class에서 $\gamma$만 예측. 더 단순하면서 동등 성능.

### GEGLU FFN

출력 게이트 대신 **GEGLU 활성화** (데이터 기반 게이팅):

$$\text{GEGLU}(x) = \text{GELU}(W_1 x) \odot (W_2 x)$$

은닉 차원은 $3d$ (DiT의 $4d$ 대신). 출력 게이트 불필요.

### Scaled Cosine Similarity Attention

표준 dot-product 대신 Swin V2의 **cosine similarity attention**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \cdot K^\top}{\lVert Q \rVert \lVert K \rVert} \cdot \tau\right) V$$

$\tau$는 헤드별 학습 가능한 온도.

### 2D Axial RoPE

RoPE를 공간 축별로 분리 적용:

$$\text{RoPE}\_{2D}(q) = \text{RoPE}\_x(q_{:d/4}) \oplus \text{RoPE}\_y(q_{d/4:d/2}) \oplus q_{d/2:}$$

Q/K 차원의 1/4씩을 x축, y축 RoPE에 할당하고 나머지 1/2은 미수정.

---

## 4. 스킵 연결: 학습 가능한 선형 보간

U-Net의 concatenation이나 단순 addition 대신 **learnable linear interpolation (lerp)**:

$$x\_{merged} = f \cdot x\_{skip} + (1-f) \cdot x\_{upsampled}$$

$f$는 학습 가능한 스칼라.

| 스킵 방식 | FID |
|----------|-----|
| Concatenation | 33.75 |
| Addition | 28.37 |
| **Learnable lerp** | **27.74** |

---

## 5. 실험 결과

### FFHQ-1024x1024

| 모델 | 유형 | Params | FID |
|------|------|--------|-----|
| NCSN++ | Diffusion | 106M | 53.52 |
| **HDiT** | **Diffusion** | **85M** | **5.23** |
| HiT-B | GAN | 117M | 6.37 |
| StyleGAN2 | GAN | 30M | 2.70 |

HDiT는 **diffusion 모델 중 FFHQ-1024 SoTA** (FID 5.23). GAN에는 아직 미치지 못하지만, DINOv2 기반 지표(FD\_D2, KD\_D2)에서는 **GAN 포함 전체 SoTA**.

### ImageNet-256x256

| 모델 | 공간 | Params | FID (no CFG) | FID (CFG) |
|------|------|--------|-------------|----------|
| DiT-XL/2 | Latent | 675M+VAE | 9.62 | 2.27 |
| ADM | Pixel | 554M | 10.94 | 4.59 |
| **HDiT** | **Pixel** | **557M** | **6.92** | **3.21** |
| simple diffusion | Pixel | 2B | 2.77 | - |

HDiT는 DiT(latent 공간)보다 좋고, ADM(U-Net)보다 좋다. simple diffusion(2B)에는 뒤지지만, 모델이 4배 작다.

### Ablation 과정 (ImageNet-128)

| 단계 | 변경 | FID | GFLOPs |
|------|------|-----|--------|
| 기준 | DiT-B/4 | 42.03 | 106 |
| + Hourglass | 계층 구조 도입 | 50.76 | 32 |
| + Neighborhood Attn | Swin → NA | 51.07 | 29 |
| + GEGLU | FFN 개선 | 44.36 | 31 |
| + Axial RoPE | 위치 인코딩 | 41.41 | 31 |
| + Soft-Min-SNR | 손실 가중 | **27.74** | **31** |

최종: DiT 대비 **더 좋은 FID + 30%의 계산량**.

---

## 6. 왜 픽셀 공간인가?

대부분의 최신 모델(LDM, DiT)은 VAE 잠재 공간에서 작동한다. HDiT는 **픽셀 공간에서 직접** 작동한다.

**장점**:
- VAE 학습/압축 아티팩트 없음
- 세부 디테일 보존 (텍스트, 미세 구조)
- 아키텍처가 더 단순 (VAE 디코더 불필요)

**가능한 이유**: Hourglass 구조로 고해상도의 연산 비용을 선형으로 억제했기 때문.

---

## 7. Hourglass Transformer(언어)와의 비교

| 항목 | Hourglass (언어, NAACL 2022) | HDiT (이미지, ICML 2024) |
|------|---------------------------|-------------------------|
| 도메인 | 1D 시퀀스 (텍스트) | 2D 공간 (이미지) |
| 다운/업샘플 | Average/Attention pooling | **Pixel-UnShuffle/Shuffle** |
| Attention | 전체 global | **Neighborhood (외부) + Global (내부)** |
| 위치 인코딩 | 1D (학습) | **2D Axial RoPE** |
| 스킵 연결 | 단순 addition | **Learnable lerp** |
| 조건부 | 없음 | **AdaRMSNorm (timestep/class)** |
| 핵심 공유 | **다운샘플 → 처리 → 업샘플의 U-Net 패턴** |

---

## 8. 개인적 생각

HDiT는 "단순한 아이디어의 힘"을 잘 보여준다:

1. **Hourglass 패턴의 범용성**: 언어(Nawrot 2022) → 이미지 생성(HDiT 2024) → 같은 다운-처리-업 패턴이 도메인을 넘어 작동한다.

2. **O(n) 스케일링이 핵심**: 해상도가 높아질수록 DiT와의 격차가 기하급수적으로 벌어진다. 1024x1024에서 99% 절감은 실용적으로 엄청난 차이.

3. **픽셀 공간의 부활**: VAE 없이도 충분한 성능이 나온다는 것은, Hourglass 구조가 이미지의 다중 해상도 특성을 잘 활용한다는 증거.

4. **Neighborhood Attention > Swin**: 로컬 어텐션 구현에서 overlapping(NA)이 non-overlapping(Swin)보다 일관되게 좋다.

이 논문은 "고해상도 = 비쌈"이라는 공식을 깨뜨렸다. Hourglass 패턴이 언어(BLT, H-Net)와 이미지(HDiT) 모두에서 성공하고 있다는 것은, **다중 해상도 계층 구조**가 시퀀스 모델링의 근본적으로 좋은 귀납 편향임을 시사한다.
