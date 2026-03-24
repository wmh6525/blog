---
title: "[서베이] Mamba/SSM 계열 모델 완전 정리 — HiPPO에서 Nemotron-H까지"
date: 2026-03-24
tags: ["논문리뷰", "Mamba", "SSM", "State Space Model"]
categories: ["ML/AI"]
summary: "Mamba/SSM 계열의 모든 주요 모델을 정리한다. 이론적 기반(HiPPO, S4)부터 Mamba-1/2/3, 병렬 아키텍처(RWKV, RetNet, Griffin, xLSTM), 하이브리드(Jamba, Hymba, Nemotron-H), 비전(VMamba, MambaVision)까지 28개 모델의 계보와 핵심 기여를 추적한다."
math: true
toc: true
draft: false
---

## 1. SSM 계보 개관

```
HiPPO (2020) → S4 (2021) → S4D/S5 (2022) → H3 (2023) → Mamba (2023)
                                                            ↓
                                              Mamba-2/SSD (2024) → Mamba-3 (2026)
                                                            ↓
                                              하이브리드: Jamba, Hymba, Nemotron-H
                                                            ↓
                                              비전: Vim, VMamba, MambaVision
```

---

## 2. 기초 이론 (Foundations)

### HiPPO (NeurIPS 2020)

- **저자**: Albert Gu et al. (Stanford)
- **기여**: "기억 = 직교 다항식 기저로의 함수 근사"라는 수학적 정의. 최적 계수 추적이 선형 ODE $\frac{dc}{dt} = Ac + Bf(t)$로 표현됨을 증명
- **의의**: S4와 Mamba의 $A$ 행렬의 이론적 근거

### S4 (ICLR 2022, Outstanding Paper)

- **저자**: Albert Gu et al. (Stanford)
- **기여**: HiPPO 행렬을 NPLR(Normal Plus Low-Rank) 분해하여 $O(L \log L)$ 글로벌 컨볼루션 커널로 변환. SSM을 딥러닝에서 실용적으로 만든 최초의 연구
- **결과**: Sequential CIFAR-10 91%, Long Range Arena 전 과제 SoTA
- **Mamba와의 관계**: 직접적 선조. Mamba는 S4의 고정 파라미터를 입력 의존적으로 확장

### DSS / S4D (NeurIPS 2022)

- **저자**: Ankit Gupta, Albert Gu et al.
- **기여**: S4의 $A$ 행렬을 **대각 행렬**로 단순화해도 성능이 유지됨을 증명. ~2줄 코드로 커널 계산 가능
- **Mamba와의 관계**: 대각 SSM이 충분하다는 근거 확립 → Mamba의 대각 파라미터화 기반

### S5 (ICLR 2023, Oral)

- **저자**: Jimmy Smith et al. (Stanford)
- **기여**: 다수의 SISO SSM 대신 단일 **MIMO SSM**을 사용하고, 컨볼루션 대신 **parallel scan**으로 계산
- **결과**: Path-X 98.5% (LRA 최고 난이도 과제)
- **Mamba와의 관계**: Parallel scan 알고리즘이 Mamba의 하드웨어 친화적 selective scan에 직접 영감

---

## 3. Mamba 직전 선행 연구

### H3 — Hungry Hungry Hippos (ICLR 2023, Spotlight)

- **저자**: Daniel Fu, Tri Dao et al. (Stanford)
- **기여**: SSM이 언어 모델링에서 부족한 두 능력(이전 토큰 리콜, 토큰 간 비교)을 식별하고 해결. FlashConv(fused FFT) 도입
- **결과**: H3-Attention 하이브리드 2.7B가 Transformer보다 낮은 perplexity
- **Mamba와의 관계**: 직전 선행작. Mamba의 selective mechanism이 H3의 한계를 직접 해결

### Hyena Hierarchy (ICML 2023)

- **저자**: Michael Poli, Tri Dao et al. (Stanford)
- **기여**: 명시적 SSM 대신 **암묵적 장거리 컨볼루션 + 데이터 제어 게이팅**으로 attention 대체
- **Mamba와의 관계**: 같은 연구 그룹. 다른 메커니즘(암묵적 컨볼루션)이지만 같은 효율성 목표

### GSS — Gated State Spaces (ICLR 2023)

- **저자**: Harsh Mehta et al. (Google)
- **기여**: 두 SSM 스트림의 곱으로 **학습된 게이팅** 도입, 언어 모델링 특화
- **Mamba와의 관계**: SSM에 게이팅을 결합한 초기 시도 → Mamba가 입력 의존 선택성으로 발전

---

## 4. Mamba 계열 (Core Family)

### Mamba-1 (ICLR 2024)

- **저자**: Albert Gu, Tri Dao (CMU, Princeton)
- **기여**: **Selective SSM** — $B$, $C$, $\Delta$를 입력 의존적으로 만듦. 하드웨어 친화적 CUDA parallel scan 커널
- **결과**: Mamba-1.4B ≈ Transformer++ 2.8B (절반 파라미터로 동등 성능)
- **핵심**: $\Delta$가 "무엇을 기억하고 잊을지" 결정 → 내용 기반 선택 메커니즘

### Mamba-2 / SSD (ICML 2024)

- **저자**: Tri Dao, Albert Gu
- **기여**: **SSM = Structured Attention** 증명 (State Space Duality). Semi-separable matrix 이론으로 SSM, linear attention, RetNet을 통합
- **결과**: Mamba-1 대비 2-8배 빠른 학습, 더 큰 state 차원($N=128$) 가능
- **핵심**: Chunk-wise 알고리즘 — 청크 내부는 행렬곱(Attention), 청크 간은 순환(SSM)

### Mamba-3 (ICLR 2026)

- **저자**: Aakash Lahoti, Kevin Li, Tri Dao, Albert Gu et al.
- **기여**: 세 가지 개선 — (1) **Exponential-Trapezoidal 이산화** (2차 정확도), (2) **복소수 SSM** (= data-dependent RoPE), (3) **MIMO** (디코딩 FLOPs 4배, latency 동일)
- **결과**: GDN +0.6%p (SISO), +1.8%p (MIMO). Mamba-2 절반 상태 크기로 동등 perplexity
- **핵심**: Conv1d 제거, state-tracking 과제 해결, 추론 Pareto 프론티어 진전

---

## 5. 병렬 아키텍처 (SSM-Adjacent)

### RWKV (EMNLP 2023 Findings)

- **저자**: Bo Peng et al. (RWKV Foundation / EleutherAI)
- **기여**: Transformer 병렬 학습 + RNN $O(1)$ 추론을 결합. 고정 시간 감쇠 + 학습된 채널 믹싱
- **결과**: 14B 규모에서 Transformer와 동등. 1K-128K 시퀀스에서 일정한 throughput (~1200 tok/s)
- **vs Mamba**: RWKV는 고정 감쇠, Mamba는 입력 의존 선택. RWKV가 더 성숙한 생태계

### RetNet (arXiv 2023.07)

- **저자**: Yutao Sun et al. (Microsoft Research / Tsinghua)
- **기여**: Retention 메커니즘 — 병렬, 순환, 청크 3가지 계산 모드 지원
- **결과**: 7B에서 Transformer 대비 디코딩 8.4배 빠름, 메모리 70% 절감
- **vs Mamba**: RetNet의 retention = 지수 감쇠가 있는 linear attention ≈ 대각 SSM. Mamba-2 SSD가 이 관계를 형식화

### LRU — Linear Recurrent Unit (ICML 2023, Oral)

- **저자**: Antonio Orvieto et al. (Google DeepMind)
- **기여**: 주의 깊게 설계된 deep RNN(선형화, 대각화, 적절한 초기화)이 deep SSM과 동등함을 증명
- **vs Mamba**: RNN과 SSM의 이론적 다리. Griffin의 핵심 빌딩 블록

### Griffin (arXiv 2024.02, → RecurrentGemma)

- **저자**: Soham De, Albert Gu et al. (Google DeepMind)
- **기여**: Gated Linear Recurrence(RG-LRU) + Local Attention 하이브리드. Hawk(순수 재귀)와 Griffin(하이브리드) 제안
- **결과**: Hawk > Mamba (downstream), Griffin ≈ Llama-2 (6배 적은 학습 토큰). 14B까지 확장
- **vs Mamba**: DeepMind의 경쟁 접근. 게이트 선형 재귀 vs 선택적 SSM. Albert Gu가 양쪽 공저자

### xLSTM (NeurIPS 2024)

- **저자**: Maximilian Beck, Sepp Hochreiter et al. (JKU Linz)
- **기여**: LSTM을 지수적 게이팅과 행렬 메모리로 확장. sLSTM(스칼라) + mLSTM(행렬, 병렬화 가능)
- **결과**: Mamba, Transformer와 경쟁적 언어 모델링 성능
- **vs Mamba**: "부활한 RNN" 접근. mLSTM의 행렬 메모리가 linear attention/SSM과 형식적 연결

### MEGA (ICLR 2023)

- **저자**: Xuezhe Ma et al. (USC / Meta)
- **기여**: 지수적 이동 평균(EMA)을 게이트 어텐션에 결합하여 위치 인식 로컬 의존성 유도
- **결과**: LRA, 번역, 언어 모델링에서 Transformer와 SSM 모두 개선

---

## 6. Linear Attention / Delta Rule 계열

### Linear Attention (ICML 2020)

- **저자**: Angelos Katharopoulos et al. (EPFL)
- **기여**: Softmax attention을 커널 특성 맵으로 분해하면 $O(N)$이 되고, **자기회귀 attention = 행렬 상태의 linear RNN**임을 증명
- **vs Mamba**: Mamba-2 SSD 프레임워크의 이론적 기반. "Attention = SSM" 이중성의 원조

### GLA — Gated Linear Attention (ICML 2024)

- **저자**: Songlin Yang et al. (MIT)
- **기여**: 데이터 의존적 게이트가 있는 linear attention + FlashLinearAttention 알고리즘
- **결과**: FlashAttention-2보다 빠른 학습 throughput. Mamba/RetNet과 경쟁적
- **vs Mamba**: SSD 프레임워크에서 GLA = 게이트 있는 SSM의 이중 표현

### DeltaNet (NeurIPS 2024)

- **저자**: Songlin Yang et al. (MIT)
- **기여**: Delta rule(타겟 메모리 업데이트)로 linear transformer 학습. Householder 행렬 곱의 메모리 효율적 표현
- **결과**: 1.3B에서 Mamba, GLA보다 낮은 perplexity

### Gated DeltaNet — GDN (ICLR 2025)

- **저자**: Songlin Yang, Jan Kautz, Ali Hatamizadeh (NVIDIA / MIT)
- **기여**: 게이팅(빠른 메모리 삭제, $\alpha \to 0$) + delta rule(타겟 콘텐츠 업데이트, $\alpha \to 1$)을 통합
- **결과**: Mamba-2, DeltaNet 전반적 능가
- **vs Mamba**: Mamba-3 아키텍처의 핵심 구성 요소. GDN + Mamba-2 + SWA = Mamba-3

### HGRN / HGRN2 (NeurIPS 2023, Spotlight)

- **저자**: Zhen Qin, Songlin Yang et al.
- **기여**: 계층적 게이트 선형 RNN — 상위 레이어일수록 장기 의존성, 하위 레이어는 로컬
- **vs Mamba**: Gated DeltaNet의 선행작. 같은 저자(Songlin Yang)가 GDN으로 발전

### Based (ICML 2024, Spotlight)

- **저자**: Simran Arora et al. (Stanford / Hazy Research)
- **기여**: 짧은 sliding window attention(64) + dense global linear attention 결합. 모델 상태 크기와 리콜 능력의 트레이드오프 식별
- **결과**: Mamba와 동등한 perplexity, 리콜 과제에서 +6.22%p. FlashAttention-2 대비 24배 throughput

---

## 7. 하이브리드 모델 (SSM + Transformer)

### Jamba / Jamba-1.5 (ICLR 2025)

- **저자**: AI21 Labs
- **기여**: Transformer + Mamba **레이어 교차** + MoE. 256K 컨텍스트
- **결과**: 단일 80GB GPU에 적재. Jamba-1.5 (94B active / 398B total)는 NVIDIA RULER long-context SoTA
- **특징**: 최초의 대규모 SSM-Transformer 하이브리드

### Zamba / Zamba2 (arXiv 2024.05)

- **저자**: Zyphra
- **기여**: Mamba 백본 + **공유 Attention 모듈** (가중치를 전 레이어에서 공유). 파라미터 비용 최소화
- **결과**: Zamba2-7B가 Llama-3, Gemma, Mistral-7B 능가. 2배 빠른 time-to-first-token

### Hymba (arXiv 2024.11)

- **저자**: NVIDIA
- **기여**: **헤드 내 SSM-Attention 융합** + 학습 가능한 Meta Tokens
- **결과**: Hymba-1.5B > Llama-3.2-3B (2배 큰 모델을 능가). KV cache ~20%만 사용
- **특징**: 레이어 교차가 아닌 **헤드 수준** 융합

### Mamba-2-Hybrid (NVIDIA, arXiv 2024.06)

- **저자**: Roger Waleffe, Tri Dao, Albert Gu et al. (NVIDIA)
- **기여**: 8B 규모에서 순수 Mamba-2 vs 하이브리드 vs Transformer 체계적 비교
- **결과**: 8B Hybrid가 12개 벤치마크 전부에서 Transformer 초과 (+2.65%p)

### Nemotron-H (arXiv 2025.04)

- **저자**: NVIDIA (~200명 공저자)
- **기여**: 하이브리드 Mamba-2 + Transformer를 **56B, 20T 토큰**까지 확장. FP8 학습
- **결과**: Qwen-2.5, Llama-3.1과 동등 이상 정확도 + **3배 추론 속도**
- **특징**: 가장 대규모의 SSM 하이브리드 모델. 단일 RTX 5090에서 ~1M 토큰 컨텍스트

---

## 8. 비전 SSM 모델 (Vision)

### Vision Mamba / Vim (ICML 2024)

- **저자**: Lianghui Zhu et al. (HUST)
- **기여**: **양방향 Mamba 블록**으로 이미지 패치 처리. 최초의 범용 비전 SSM 백본
- **결과**: DeiT 전 스케일 능가. 고해상도(1248²)에서 2.8배 빠름, 메모리 86.8% 절감

### VMamba (NeurIPS 2024, Spotlight)

- **저자**: Yue Liu et al. (CAS / Huawei)
- **기여**: **Cross-Scan** (4방향 독립 SSM)으로 2D 공간에서 글로벌 수용 필드 확보
- **결과**: VMamba-T 82.2% (Swin-T +0.9%), VMamba-S 83.5%

### MambaVision (CVPR 2025)

- **저자**: Ali Hatamizadeh, Jan Kautz (NVIDIA)
- **기여**: 마지막 레이어에만 self-attention 추가한 Mamba 비전 하이브리드
- **결과**: ImageNet-1K Top-1 정확도 + throughput 모두 SoTA

---

## 9. 순수 SSM 대규모 모델

### Falcon Mamba (arXiv 2024.10)

- **저자**: TII (Abu Dhabi)
- **기여**: **Attention 없는 순수 Mamba 7B**, 5.8T 토큰 학습
- **결과**: Mistral 7B, Llama-3.1 8B 능가. Open LLM Leaderboard 최고 순수 Mamba
- **의의**: 충분한 데이터가 있으면 순수 SSM도 Transformer와 경쟁 가능함을 검증

---

## 10. 전체 모델 비교 표

### 효율적 시퀀스 모델의 스펙트럼

| 모델 | 연도 | 유형 | 학습 복잡도 | 추론/토큰 | 핵심 메커니즘 |
|------|------|------|-----------|----------|-------------|
| S4 | 2021 | SSM | $O(L \log L)$ | $O(1)$ | 고정 컨볼루션 커널 |
| H3 | 2023 | SSM+Gate | $O(L \log L)$ | $O(1)$ | SSM + multiplicative gate |
| **Mamba-1** | 2023 | Selective SSM | $O(L)$ | $O(1)$ | 입력 의존적 $B$, $C$, $\Delta$ |
| RWKV | 2023 | Linear RNN | $O(L)$ | $O(1)$ | 고정 시간 감쇠 + 채널 믹싱 |
| RetNet | 2023 | Retention | $O(L)$ | $O(1)$ | 지수 감쇠 linear attention |
| Griffin | 2024 | Gated LR + Attn | $O(L)$ | $O(1)$/$O(L)$ | RG-LRU + local attention |
| **Mamba-2** | 2024 | SSD | $O(L)$ | $O(1)$ | SSM = Structured Attention |
| GDN | 2025 | Gated Delta | $O(L)$ | $O(1)$ | 게이팅 + delta rule |
| **Mamba-3** | 2026 | SSD++ | $O(L)$ | $O(1)$ | Exp-Trap + 복소수 + MIMO |
| Transformer | - | Attention | $O(L^2)$ | $O(L)$ | Softmax QKV |

### 하이브리드 모델 비교

| 모델 | 규모 | SSM 종류 | 융합 방식 | 결과 |
|------|------|---------|----------|------|
| Jamba | 52B(MoE) | Mamba-1 | 레이어 교차 | 256K 컨텍스트, 단일 80GB |
| Zamba2 | 7B | Mamba-2 | 공유 Attention | Llama-3 능가, 2배 빠른 TTFT |
| Hymba | 1.5B | Mamba-2 | **헤드 내 융합** | 3B 모델 능가, 캐시 20% |
| Mamba-2-Hybrid | 8B | Mamba-2 | 혼합 레이어 | Transformer +2.65%p |
| **Nemotron-H** | **56B** | **Mamba-2** | **혼합 레이어** | **Llama-3.1 동등 + 3배 빠른 추론** |

---

## 11. 핵심 통찰

### SSM의 발전 방향

1. **고정 → 선택적**: S4(고정 $A$,$B$,$C$) → Mamba(입력 의존적) → Mamba-3(입력 의존적 $A$ + 복소수)
2. **컨볼루션 → 재귀 → 행렬곱**: S4(FFT) → Mamba(parallel scan) → Mamba-2(SSD chunk matmul)
3. **SSM vs Attention → SSM + Attention**: 경쟁에서 융합으로. 하이브리드가 양쪽 모두 능가

### "SSM vs Attention" 경쟁의 현재 위치

- **순수 SSM** (Falcon Mamba): 충분한 데이터면 경쟁 가능하지만, ICL/리콜에서 약점
- **순수 Transformer**: 성능은 좋지만 추론 비용 $O(L^2)$
- **하이브리드** (Nemotron-H, Hymba): 현재 최적해. 대부분의 처리는 SSM, 정밀 추론은 소수의 Attention 레이어

### Mamba-2의 SSD가 통합한 것들

Mamba-2의 State Space Duality 프레임워크는 다음을 **하나의 수학적 구조**(semi-separable matrix)로 통합했다:

| 모델 | SSD 관점 |
|------|---------|
| Linear Attention | $A = 0$인 SSD (감쇠 없음) |
| RetNet | 고정 감쇠의 SSD |
| RWKV | 특수 감쇠 패턴의 SSD |
| Mamba-1 | 선택적 감쇠의 SSD |
| GLA | 게이트 있는 SSD |

---

## 참고 문헌

| 모델 | arXiv |
|------|-------|
| HiPPO | 2008.07669 |
| S4 | 2111.00396 |
| S4D | 2206.11893 |
| S5 | 2208.04933 |
| H3 | 2212.14052 |
| Hyena | 2302.10866 |
| Linear Attention | 2006.16236 |
| RWKV | 2305.13048 |
| RetNet | 2307.08621 |
| LRU | 2303.06349 |
| Griffin | 2402.19427 |
| xLSTM | 2405.04517 |
| GLA | 2312.06635 |
| DeltaNet | 2406.06484 |
| Gated DeltaNet | 2412.06464 |
| Based | 2402.18668 |
| Mamba-1 | 2312.00752 |
| Mamba-2/SSD | 2405.21060 |
| Mamba-3 | 2603.15569 |
| Jamba | 2403.19887 |
| Zamba | 2405.16712 |
| Hymba | 2411.13676 |
| Mamba-2-Hybrid | 2406.07887 |
| Nemotron-H | 2504.03624 |
| Vision Mamba | 2401.09417 |
| VMamba | 2401.10166 |
| MambaVision | 2407.08083 |
| Falcon Mamba | 2410.05355 |
