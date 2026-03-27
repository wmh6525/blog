---
title: "[논문 리뷰] Hourglass Transformer — 계층적 구조로 시퀀스 모델의 효율성을 높이다"
date: 2026-03-27
tags: ["논문리뷰", "Hourglass", "Transformer", "계층적"]
categories: ["ML/AI"]
summary: "Hourglass Transformer 논문 상세 리뷰. U-Net에서 영감받은 다운샘플링-처리-업샘플링 구조로, 동일 계산량에서 표준 Transformer를 능가한다. BLT, H-Net, Dynamic Token Pooling으로 이어지는 계층적 시퀀스 모델의 원조."
math: true
toc: true
draft: false
---

## 논문 정보

- **제목**: Hierarchical Transformers Are More Efficient Language Models
- **저자**: Piotr Nawrot, Szymon Tworkowski, Michal Tyrolski (Warsaw), Lukasz Kaiser (OpenAI), Yuhuai Wu, Christian Szegedy, Henryk Michalewski (Google Research)
- **학회**: Findings of NAACL 2022
- **코드**: [google/trax (hourglass.py)](https://github.com/google/trax/blob/master/trax/models/research/hourglass.py), [lucidrains/hourglass-transformer-pytorch](https://github.com/lucidrains/hourglass-transformer-pytorch)

---

## 1. 핵심 아이디어: 모래시계 형태

U-Net에서 영감받아, Transformer에 **명시적 계층 구조**를 도입한다:

```
[Pre-vanilla 레이어] — 전체 해상도
        ↓ 다운샘플링 (k배 압축)
[Shortened 레이어] — 1/k 해상도 (저비용, 넓은 문맥)
        ↓ 업샘플링 (k배 복원)
[Post-vanilla 레이어] — 전체 해상도
```

표기법: `(N1@f1, N2@f2, N3@f3)` — 예: `(4@1, 8@3, 4@1)` = 4레이어(전체) → 8레이어(1/3 해상도) → 4레이어(전체)

### 왜 효율적인가?

Shortened 레이어의 시퀀스가 $L/k$이므로:
- **Self-attention**: $O(L^2) \to O((L/k)^2) = O(L^2/k^2)$ — $k^2$배 빠름
- **FFN**: $O(L) \to O(L/k)$ — $k$배 빠름

절약된 계산을 **더 많은 레이어나 더 큰 모델**에 재투자한다.

---

## 2. 다운샘플링 방법

4가지 방법을 비교:

| 방법 | enwik8 BPC | 설명 |
|------|----------|------|
| Average Pooling | 1.129 | stride $k$의 1D 평균 풀링 |
| Linear Pooling | 1.159 | $(L, d) \to (L/k, k \cdot d) \to (L/k, d)$ 선형 투영 |
| **Attention (AvgPool base)** | **1.124** | $x' = S(x) + \text{Attention}(Q=S(x), K=V=x)$ |
| Attention (LinearPool base) | 1.142 | 위와 동일하지만 base가 LinearPool |

Attention 기반 풀링이 가장 좋다 — 기본 풀링으로 초기 압축 후 attention으로 정제.

---

## 3. 업샘플링 방법

| 방법 | enwik8 BPC | 설명 |
|------|----------|------|
| Repeat | 1.148 | 각 벡터를 $k$번 복제 |
| Linear | 1.163 | $(L/k, d) \to (L/k, k \cdot d) \to (L, d)$ |
| Skip only ($U=x$) | 1.145 | 압축 표현 무시, 스킵 연결만 사용 |
| **$U(x,x')=x+\text{Linear}(x')$** | **1.132** | 스킵 + 압축 표현의 선형 업샘플링 합산 |

스킵 연결 + 선형 업샘플링 결합이 최적.

---

## 4. 스킵 연결의 중요성

다운샘플링 전의 전체 해상도 표현 $x$를 저장하고, 업샘플링 시 합산:

$$x \leftarrow x + \text{Upsampling}(x, x', k)$$

이것이 **세부 정보를 병목을 통해 보존**하는 핵심 메커니즘이다.

---

## 5. 자기회귀 시프트

인과적 모델에서 정보 누출 방지를 위해, 다운샘플링 직전에 시퀀스를 **$k-1$ 토큰 우측 시프트**:

$k$개 토큰을 하나로 풀링할 때, 마지막 $k-1$개 토큰의 정보가 풀링된 토큰에 포함되면 미래 정보를 보는 것이 된다. 시프트로 이를 방지.

---

## 6. 핵심 발견: Vanilla 레이어 배치

| Pre, Post 레이어 | enwik8 BPC | CIFAR-10 BPD |
|----------------|----------|-------------|
| (0, 0) — 없음 | 1.460 | 3.429 |
| (0, 2) | 1.176 | 3.108 |
| (2, 0) | 1.189 | 3.035 |
| (1, 1) | 1.171 | 3.012 |
| **(2, 2)** | **1.128** | **2.966** |

**핵심 통찰**:
1. Vanilla 레이어가 **없으면 치명적** (1.460 vs 1.128)
2. **대칭 배치**가 최적 — 다운샘플링 전후 모두 필요
3. 더 많을수록 좋지만, 수확 체감

---

## 7. Shortened 레이어 수의 효과

| Shortened 레이어 수 | enwik8 BPC | CIFAR-10 BPD |
|-------------------|----------|-------------|
| 1 | 1.164 | 3.28 |
| 4 | 1.134 | 3.16 |
| 8 | 1.111 | 3.07 |
| 16 | 1.096 | 3.03 |

Shortened 레이어는 **저비용**이므로 많이 추가해도 전체 계산량 증가가 적다. 품질은 일관되게 향상.

---

## 8. 재귀적 중첩 (Nested Hourglass)

아키텍처는 **재귀적으로 중첩** 가능하다:

```
(2@1, 1@2, 4@4, 1@2, 2@1)
```

= 2레이어(전체) → 1레이어(1/2) → **4레이어(1/4)** → 1레이어(1/2) → 2레이어(전체)

팰린드롬(대칭) 구조가 필수. 더 깊은 압축으로 더 넓은 문맥을 더 적은 비용으로 처리.

---

## 9. Shorten Factor Dropout

학습 시 shortening factor를 $\lbrace 2, 3 \rbrace$ 중 랜덤 샘플링하는 정규화 기법:

| 설정 | enwik8 BPC |
|------|----------|
| 고정 $k=2$ | 1.143 |
| 고정 $k=3$ | 1.134 |
| **$k \in \lbrace 2,3 \rbrace$ 드롭아웃** | **1.129** |

0.005-0.014 BPC 개선. 모델이 다양한 압축률에 적응하도록 만든다.

---

## 10. 실험 결과

### enwik8 (146M params)

| 모델 | Params | BPC |
|------|--------|-----|
| Transformer-XL | 277M | 0.99 |
| **Hourglass** | **146M** | **0.98** |
| Adaptive-Span | 209M | 0.98 |

**절반의 파라미터로 Transformer-XL과 동등.**

### ImageNet32

| 모델 | BPD |
|------|-----|
| PixelCNN | 3.83 |
| Image Transformer | 3.77 |
| **Hourglass** | **3.74** |

자기회귀 Transformer 중 **SoTA**.

### 효율성 비교 (동일 계산량)

모든 계산 수준에서 Hourglass가 vanilla Transformer를 능가. 예:

| 모델 | BPC | 메모리 | 속도 |
|------|-----|-------|------|
| 8@1 (vanilla 8층) | 1.151 | 5.75GB | 0.73 |
| 2@1, 8@3, 2@1 (Hourglass) | **1.111** | 5.50GB | 0.88 |

더 적은 메모리, 더 빠른 속도, 더 좋은 성능.

---

## 11. 후속 연구로의 영향

| 후속 연구 | Hourglass에서 받은 영향 |
|----------|----------------------|
| **Dynamic Token Pooling** (Nawrot, ACL 2023) | 고정 $k$ → **동적 경계** 학습 (같은 1저자) |
| **BLT** (Meta, 2024) | Local Encoder → Latent Transformer → Local Decoder 구조 |
| **H-Net** (CMU, 2025) | U-Net 구조 + E2E 동적 청킹 |
| **Hourglass Diffusion Transformers** | Diffusion 모델에 적용, 메가픽셀에서 99% FLOP 절감 |

Hourglass Transformer는 **"다운샘플링-처리-업샘플링"이라는 아키텍처 패턴**을 시퀀스 모델에 확립한 선구적 연구다. BLT와 H-Net은 이 패턴에 동적 경계를 추가한 것이다.

---

## 12. 핵심 설계 원칙 요약

1. **대칭 배치**: Vanilla 레이어를 병목 앞뒤에 균등 배치
2. **Attention 리샘플링**: 단순 풀링보다 attention 기반 다운/업샘플링
3. **스킵 연결**: 세부 정보를 병목을 우회하여 보존
4. **Shortened 레이어 확장**: 저비용이므로 적극 추가
5. **Shortening factor dropout**: 다양한 압축률에 대한 정규화
6. **자기회귀 시프트**: 인과성 유지를 위한 $k-1$ 시프트
