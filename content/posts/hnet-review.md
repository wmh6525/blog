---
title: "[논문 리뷰] H-Net: 완전 E2E 동적 청킹으로 토크나이저를 제거하다"
date: 2026-03-27
tags: ["논문리뷰", "H-Net", "바이트", "토크나이저프리", "Mamba"]
categories: ["ML/AI"]
summary: "H-Net 논문 상세 리뷰. 외부 모델 없이 바이트 시퀀스의 경계를 완전 E2E로 학습하는 계층적 아키텍처. Mamba-2 인코더/디코더 + Transformer 백본으로 BPE Transformer를 능가하며, 학습된 경계가 자연스럽게 ~4.5 bytes/chunk로 수렴한다."
math: true
toc: true
draft: false
---

<!-- LEARNING-START -->
## 처음 읽는 분을 위한 안내

**학습 위치:** [작은 단위를 묶어 표현하기]({{< relref "learn" >}}#representation) · 4/10

글을 어디서 끊어 묶을지 사람이 미리 정하지 않고 모델이 학습하게 할 수 있다. H-Net은 이 경계 선택을 전체 모델의 학습에 포함한다. 바이트가 묶음으로 바뀌고 다시 출력으로 복원되는 경로를 먼저 살펴보자.

**먼저 알아둘 말**

- [바이트]({{< relref "ai-glossary" >}}#byte): 컴퓨터가 데이터를 저장하는 단위. 글자를 바이트로 읽는 모델은 고정된 단어 조각 대신 더 작은 데이터 단위에서 출발한다.
- [패치·청크]({{< relref "ai-glossary" >}}#patch): 작은 단위 여러 개를 한 묶음으로 만든 것. 이미지 패치, 바이트 패치, 문서 청크는 서로 다른 대상을 묶는다.
- [학습과 추론]({{< relref "ai-glossary" >}}#training): 학습은 데이터로 모델 안의 숫자를 바꾸는 일이고, 추론은 모델에 입력을 넣어 결과를 얻는 일이다.

**이번 글의 목표:** 묶음의 경계 자체를 학습하는 방법 이해하기.

**이 학습 경로의 앞 글:** [\[서베이\] BLT와 바이트 수준 언어 모델의 계보 — 토크나이저 없는 LLM을 향해]({{< relref "posts/blt-byte-level-lm-survey" >}}).

토큰과 벡터가 낯설다면 [텍스트 데이터 다루기]({{< relref "posts/build-llm-ch2-working-with-text-data" >}})의 설명부터 읽자.

**본문에서 먼저 볼 순서:** **1. 핵심 기여: 외부 모델 없는 완전 E2E 동적 청킹** → **2. 아키텍처: U-Net 스타일 3단계** → **3. 동적 청킹 (DC): 핵심 혁신**.

첫 독서에서는 문제와 방법의 흐름을 잡자. 수식·코드·세부 성능표는 두 번째에 읽어도 된다. 구현과 수학적 주장까지 확인하려면 그 부분을 다시 살펴봐야 한다.

<!-- PAPER-PREREQUISITES -->
### 선수 개념과 읽을 논문

**먼저 이해할 개념:** 벡터 표현, 압축과 복원, 시퀀스 길이; 잠재 분포를 다루는 글은 확률.

아래 순서는 이 글을 이해하기 위한 공부 순서다. ‘개념 준비’는 읽기 안내이며, 직접적인 연구 계보는 연결 칸에 따로 표시했다. 이미 해당 개념을 설명할 수 있다면 아는 부분은 넘어가도 된다.

| 순서·논문 | 연결·익힐 개념 | 먼저 볼 부분 |
|---|---|---|
| 1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | **사용한 구성 요소 이해**: Attention·위치 정보·인코더/디코더 | 현재 토큰이 다른 위치의 정보를 읽는 구조 그림과 Attention 설명 |
| 2. [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) · [블로그 해설]({{< relref "posts/mamba2-review" >}}) | **사용한 구성 요소 이해**: 구조화된 Attention·상태 공간 이중성·청크 계산 | 재귀와 행렬 표현의 대응, 청크 안과 청크 사이 계산 |
| 3. [Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2412.09871) · [블로그 해설]({{< relref "posts/blt-byte-level-lm-survey" >}}) | **개념 준비**: 엔트로피 패칭·바이트 인코더/디코더 | 경계를 정하고 묶음을 백본에 넘기는 흐름 |

**현재 글과 연결하기:** 위 개념을 익힌 뒤, 이 글에서 ‘묶음의 경계 자체를 학습하는 방법 이해하기’에 해당하는 부분을 읽자. 기존 방법의 어느 단계가 바뀌었는지 그림이나 입력·출력으로 설명해 보자.

[선수 논문 전체 지도와 원문 읽는 법]({{< relref "paper-prerequisites" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- PAPER-PREREQUISITES-END -->

<!-- LEARNING-START-END -->

---

## 논문 정보

- **제목**: Dynamic Chunking for End-to-End Hierarchical Sequence Modeling
- **저자**: Sukjun Hwang, Brandon Wang, Albert Gu
- **소속**: Carnegie Mellon University, Cartesia AI
- **발표**: arXiv: 2507.07955 (2025.07)
- **코드**: [github.com/goombalab/hnet](https://github.com/goombalab/hnet)

---

## 1. 핵심 기여: 외부 모델 없는 완전 E2E 동적 청킹

BLT는 **외부 엔트로피 모델**로 패치 경계를 결정한다. H-Net은 이 의존성을 제거하고 경계를 **모델 자체가 E2E로 학습**한다.

| 접근 | MEGABYTE | BLT | **H-Net** |
|------|----------|-----|-----------|
| 패칭 | 고정 크기 | 외부 엔트로피 모델 | **E2E 학습** |
| 계층 | 단일 | 단일 | **다중 (재귀 가능)** |
| 외부 모델 | 불필요 | **필요** | 불필요 |
| 자기회귀 | 가능 | 가능 | 가능 |

청크 경계는 백본에 넘길 시퀀스의 길이와 정보 묶음을 동시에 바꾼다. H-Net에서는 이 경계를 외부 모델이 미리 정하는 대신, 전체 언어 모델의 학습 과정 안에서 조정한다. 그래서 경계 예측과 청크 표현을 따로 떼어 평가하기 어렵다. 아래 구조를 볼 때는 바이트가 경계를 거쳐 압축되고, 다시 바이트 출력으로 복원되는 경로를 함께 따라가면 된다.

---

## 2. 아키텍처: U-Net 스타일 3단계

```
바이트 시퀀스 (L=8192)
    ↓
[Encoder E] — Mamba-2 × 4 레이어
    ↓ 동적 청킹 (경계 학습)
[Main Network M] — Transformer × 24-27 레이어 (핵심, 파라미터 대부분)
    ↓ 디청킹 (업샘플링)
[Decoder D] — Mamba-2 × 4 레이어 + 인코더 잔차 연결
    ↓
바이트 출력
```

### 왜 Mamba-2를 인코더/디코더로?

이 ablation에서는 **순수 Mamba가 순수 Transformer보다 훨씬 좋은 결과**를 냈다.

| E-D 구성 | 성능 |
|---------|------|
| **M4-M4** (순수 Mamba) | **최고** |
| M2T1-T1M2 | 2등 |
| T1M2-M2T1 | 3등 |
| T2-T2 (순수 Transformer) | **최악** (더 많은 FLOPs에도 불구) |

SSM이 **고정 크기 상태로 정보를 압축**하는 본질적 특성이 청킹과 자연스럽게 정렬된다.

---

## 3. 동적 청킹 (DC): 핵심 혁신

DC는 **라우팅 모듈**, **스무딩 모듈**, **비율 손실** 세 구성요소로 이루어진다.

### 3.1 라우팅 모듈: 코사인 유사도 기반 경계 예측

인접한 인코더 출력 간의 **코사인 비유사도**로 경계 확률을 계산한다:

$$q_t = W_q \cdot \hat{x}_t, \quad k_t = W_k \cdot \hat{x}_t$$

$$p_t = \frac{1}{2}\left(1 - \frac{q_t^\top k_{t-1}}{\lVert q_t \rVert \lVert k_{t-1} \rVert}\right) \in [0, 1]$$

$$b_t = \mathbb{1}\lbrace p_t \geq 0.5 \rbrace$$

직관: 연속된 벡터가 의미적 경계를 가로지르면 유사도가 떨어지고 $p_t$가 높아진다.

> **주의**: H-Net은 Gumbel-Sigmoid를 사용하지 **않는다**. 코사인 유사도 라우팅 + Straight-Through Estimator(STE)를 사용한다.

### 3.2 다운샘플링

$b_t = 1$인 위치의 벡터만 **직접 선택** (direct selection). Mean pooling, max pooling, cross-attention보다 단순한 직접 선택이 가장 좋았다.

### 3.3 스무딩 모듈: 미분 가능한 업샘플링

이산 경계는 그래디언트 흐름을 차단한다. **지수 이동 평균(EMA)**으로 이산 연산을 연속화한다:

$$\bar{z}_t = P_t \cdot \hat{z}_t + (1 - P_t) \cdot \bar{z}_{t-1}$$

세 가지 역할:
1. **미분 가능한 경계 학습**: 이산 업샘플링을 연속 연산으로 변환
2. **적응적 오류 보정**: 확신 높은 청크($P_t \approx 1$)는 이산 유지, 낮은 확신($P_t \approx 0.5$)은 이전 청크로 보완
3. **학습 안정성**: 초기의 차선 청킹 패턴에 과적합 방지

### 3.4 비율 손실 (Ratio Loss)

MoE의 load balancing에서 영감. 극단적 압축/비압축 방지:

$$\mathcal{L}_{ratio} = \frac{N}{N-1}\left((N-1) \cdot F \cdot G + (1-F)(1-G)\right)$$

- $F = \frac{1}{L}\sum_t b_t$ (실제 선택 비율, 비미분)
- $G = \frac{1}{L}\sum_t p_t$ (평균 경계 확률, 미분 가능)
- $N$: 목표 압축 비율

$F = G = 1/N$일 때 최소. 전체 손실:

$$\mathcal{L} = \mathcal{L}_{AR} + \alpha \sum_{s=0}^{S-1} \mathcal{L}_{ratio}^s, \quad \alpha = 0.03$$

---

## 4. 자연 수렴: ~4.5 bytes/chunk

| 모델 | 목표 비율 | 실제 BPIC | BPE 참고 |
|------|----------|----------|---------|
| H-Net (1-stage, Large) | 1/6 | **4.8** | 4.6 |
| H-Net (1-stage, XL) | 1/6 | **4.7** | 4.6 |
| H-Net (2-stage) | 1/3 × 1/3 | **6.9-7.0** | - |

BPE 토크나이저의 압축률(~4.6 bytes/token)과 거의 동일하게 수렴한다!

시각화를 보면, 1-stage 모델은 경계를 **공백 위치에 주로 배치** — SpaceByte의 규칙 기반 접근을 순수 학습으로 재발견한 것이다.

---

## 5. 실험 결과

### 5.1 언어 모델링 (Validation BPB)

**XL 규모 (GPT-3 1.3B 매칭):**

| 모델 | BPB | 하류 평균 정확도 |
|------|-----|----------------|
| Transformer (BPE) | 0.730 | 55.5% |
| SpaceByte++ | 0.733 | - |
| H-Net (1-stage) | 0.728 | 56.7% |
| **H-Net (2-stage)** | **0.715** | **58.2%** |

H-Net (2-stage)가 BPE Transformer를 **+2.6%p** 능가.

**핵심 발견**: Large H-Net (2-stage)이 **XL BPE Transformer와 동등한 하류 성능** (55.5%) — 절반 크기로 동등 성능.

### 5.2 강건성 (HellaSwag 텍스트 변형)

| 모델 | 강건성 점수 |
|------|-----------|
| Transformer (BPE) | 20.2 |
| **H-Net (2-stage)** | **39.0** |

BPE 대비 **약 2배**의 강건성.

### 5.3 중국어 & 코드

| 모델 | 중국어 BPB | XWinograd-zh | 코드 BPB |
|------|----------|-------------|---------|
| Transformer (Llama-3) | 0.740 | 59.9% | 0.338 |
| **H-Net (2-stage)** | **0.703** | **66.3%** | **0.316** |

비라틴 문자에서 특히 큰 이점.

### 5.4 DNA (HG38)

H-Net이 isotropic 모델과 **같은 perplexity를 3.6배 적은 데이터**로 달성 — 극적인 데이터 효율성.

---

## 6. Ablation Study 요약

| 실험 | 핵심 발견 |
|------|----------|
| DC 구성요소 제거 | **스무딩 모듈 제거 시 가장 큰 성능 저하** (압축 비율 불안정) |
| E/D 아키텍처 | 순수 Mamba > 하이브리드 > 순수 Transformer |
| 다운샘플링 방식 | 직접 선택 > mean/max pooling > cross-attention |
| 하이브리드 백본 | Mamba-2:Transformer = 3:1 혼합이 순수 Transformer보다 약간 우수 |
| vs MoE | H-Net이 LlamaByte-MoE보다 우수 → 이점이 단순 희소성이 아닌 **의미적 희소성** |
| 증류 | 사전학습된 Llama로 백본 초기화 → <200B 바이트로 좋은 성능 |

---

## 7. BLT vs H-Net 비교

| 항목 | BLT | H-Net |
|------|-----|-------|
| 경계 결정 | 외부 엔트로피 모델 | **E2E 코사인 유사도** |
| 계층 | 단일 | **다중 (재귀 가능)** |
| 인코더/디코더 | 경량 Transformer | **Mamba-2** |
| 복잡도 | 높음 ("설정이 너무 복잡") | 상대적으로 단순 |
| 검증 규모 | **8B** | 1.3B |
| N-gram 해시 | 있음 | 없음 |

BLT는 더 큰 규모에서 검증되었지만 H-Net은 더 순수한 E2E 접근이다. 두 접근의 장점을 결합하는 것이 자연스러운 다음 단계.

---

## 8. SancMamba와의 연결

H-Net은 SancMamba의 목표와 가장 직접적으로 관련된 연구다:

| H-Net | SancMamba |
|-------|----------|
| 코사인 유사도 경계 | SSM $\Delta$ + boundary head |
| 직접 선택 다운샘플 | ChunkToConceptEncoder |
| EMA 스무딩 | (Gumbel-sigmoid STE) |
| 비율 손실 | SANC $C_{struct}$ (구조적 비용) |
| Mamba-2 인코더 | Mamba 인코더 |
| 다음 바이트 예측 | **다음 개념 토큰 예측** |

핵심 차이: H-Net은 바이트 수준 언어 모델링이 목표이고 SancMamba는 **개념 수준 추론**이 목표.

<!-- LEARNING-END -->
---

## 읽은 뒤 확인하고 다음으로

**확인 질문:** 묶음의 경계 자체를 학습하는 방법 이해하기를 목표로, 본문에서 다룬 입력·중간 과정·결과를 자기 말로 설명해 보자. 실험이나 사례가 있다면 어떤 조건에서 확인했는지도 덧붙여 보자.

**다음 글:** [\[논문 리뷰\] MBLM — 5M 바이트 컨텍스트를 단일 GPU로 처리하는 계층적 바이트 LLM]({{< relref "posts/mblm-review" >}}) — 작은 묶음을 큰 묶음으로 쌓는 계층 구조 이해하기.

[전체 학습 순서]({{< relref "learn" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- LEARNING-END-END -->
