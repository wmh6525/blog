---
title: "[논문 리뷰] MBLM — 5M 바이트 컨텍스트를 단일 GPU로 처리하는 계층적 바이트 LLM"
date: 2026-05-15
tags: ["논문리뷰", "MBLM", "바이트", "Mamba", "Transformer", "장문컨텍스트"]
categories: ["ML/AI"]
summary: "MBLM(Multiscale Byte Language Model, IBM Research, ICML 2025 Workshop) 심화 리뷰. MegaByte의 2단계 계층을 N단계로 일반화 + Transformer/Mamba 백본 혼합. PG19에서 5M 바이트 컨텍스트를 단일 A100에서 학습. 순서도 도식과 학습/추론 절차 완전 해부."
math: true
toc: true
draft: false
---

## 논문 정보

- **제목**: Multiscale Byte Language Models — A Hierarchical Architecture for Causal Million-Length Sequence Modeling
- **저자**: Eric Egli, Matteo Manica, Jannis Born
- **소속**: **IBM Research Europe**
- **arXiv**: 2502.14553 (2025.02, ICML 2025 Workshop)
- **코드**: [github.com/ai4sd/multiscale-byte-lm](https://github.com/ai4sd/multiscale-byte-lm) (PyPI: `mblm`)

---

## 1. 한 줄 요약

> **MegaByte의 2단계 계층을 임의의 N단계로 일반화하고 Transformer/Mamba를 스테이지마다 자유 조합 — 단일 A100에서 5,000,000 바이트 컨텍스트 학습.**

---

## 2. 배경 — 바이트 LM 계보

토크나이저 없이 **바이트를 직접 처리**하면 다국어·멀티모달 통합이 자연스럽다. 하지만 바이트 시퀀스는 토큰 대비 **4-6배 길어** Transformer가 비현실적.

| 모델 | 단계 | 백본 | 최대 컨텍스트 | 핵심 |
|------|-----|------|------------|------|
| **MegaByte** (2023) | 2 | Transformer×2 | 1.2M | 고정 패치 계층 |
| **MambaByte** (2024) | 1 | Mamba | — | 선형 시간 SSM |
| **BLT** (2024) | 2 | Transformer | — | 엔트로피 동적 패칭 |
| **bGPT** (2024) | 1 | Transformer | — | 멀티모달 바이너리 사전학습 |
| **★ MBLM** (2025) | **N (무제한)** | **Transformer / Mamba-2 / 혼합** | **5M (단일 GPU)** | N단계 + 백본 자유 조합 |

**MBLM의 위치**: "MegaByte를 N단계로 일반화 + 스테이지마다 백본 선택 + 메모리 효율 청킹"

---

## 3. 아키텍처 순서도

### 3.1 전체 데이터 흐름 (N단계 일반화)

```
                      바이트 입력 x [B, L]
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │ Stage 1 (Global, 가장 거친 단위)              │
        │  ① Byte embedding + Positional encoding      │
        │  ② Nested patches로 reshape                  │
        │     [B, L] → [B, P1, P2, ..., PN, D_N]       │
        │  ③ Stage 차원으로 project → D_1              │
        │  ④ 패딩 start token 추가                      │
        │  ⑤ pack: [K_1, P_1, D_1]                     │
        │  ⑥ Model M_1 forward (Transformer 또는 Mamba) │
        │  ⑦ Linear projection → 다음 stage 임베딩에 더함│
        └──────────────────────┬──────────────────────┘
                               ▼
        ┌─────────────────────────────────────────────┐
        │ Stage 2 (Global)                             │
        │  P_2^in = P_2^emb' + project(P_1^out)        │
        │  ↑ residual cross-stage connection           │
        │  Model M_2 forward (옵션: 청크 c개로 분할     │
        │                     gradient checkpointing)  │
        └──────────────────────┬──────────────────────┘
                               ▼
                       ... Stage 3, ..., N-1 ...
                               │
                               ▼
        ┌─────────────────────────────────────────────┐
        │ Stage N (Local, byte-level)                  │
        │  P_N^in = P_N^emb' + project(P_{N-1}^out)    │
        │  Model M_N forward                           │
        │  Linear head → logits over 256 bytes         │
        └──────────────────────┬──────────────────────┘
                               ▼
                  Reshape → next-byte CE loss
                  또는 sampling
```

**핵심 설계**:
- 단계 1 ~ N-1: **global decoder** — 패치 간 의존성
- 단계 N: **local decoder** — 패치 내부 byte 자회귀
- 어휘 = 256 바이트 (+ `<pad>` = ID 257)
- 최대 컨텍스트: $L_{\max} = \prod_i P_i$

### 3.2 수식 (논문 Eq. 1-8)

**Patch embedder**:

$$x_i^{\text{emb}} \in \mathbb{R}^{B \times L \times D_N} = E_i^{\text{emb}}(x) + E_i^{\text{pos}}(x)$$

스테이지별 독립 바이트+위치 임베딩, 임베딩 차원은 항상 $D_N$.

Reshape → $P_i^{\text{emb}} \in \mathbb{R}^{B \times P_1 \times \cdots \times P_N \times D_N}$
스테이지 차원으로 투영: $W_i^{\text{patch}}$로 $D_i$ 차원
학습 가능한 시작 토큰 $E_i^{\text{pad}} \in \mathbb{R}^{D_i}$ prepend.

**Global model**:

$$P_i^{\text{in}} = P_i^{\text{emb}\prime} + P_{i-1}^{\text{out}} \cdot W_{i-1}^{\text{global}}$$

$$P_i^{\text{out}} = \text{concat}_c(M_i(P_i^{\text{in}}))$$

$c$ = optional chunking (gradient checkpointing용 mini-batch).

**Local model**:

$$Z \in \mathbb{R}^{K_N \times P_N \times V} = P_N^{\text{out}} \cdot W^{\text{head}}$$

**평가 지표**:

$$\text{BPB} = \frac{\ell_{\text{byte}}}{\ln 2}, \quad \text{PPL}_{\text{word}} = \exp\left(\frac{L_B}{L_W} \cdot \ell_{\text{byte}}\right)$$

---

## 4. 백본 선택

스테이지마다 자유 선택 → 약자 표기: **T** = Transformer, **S** = Mamba (SSM).

| 백본 | 세부 |
|------|-----|
| **Transformer-2 (T)** | hidden 1024, 16헤드×64, FFN expansion 2, **RoPE** (global stage만) |
| **Mamba-2 (S)** | model dim 1024, SSM state 128, conv width 4, block expansion 2, head dim 64, **위치 임베딩 없음** (실험으로 검증) |

**조합 예시**:
- **TT** (2D MegaByte 베이스라인)
- **ST** = global Mamba + local Transformer (**저자 추천**)
- **SS** = 양쪽 Mamba (최고 성능)
- **STT, TTT** = 3단계

---

## 5. 학습 데이터

### 5.1 PG19 (텍스트, 사전학습)

- 28,752권의 1919년 이전 출판 도서
- **11.68B 바이트** train / **17.73M** val / **41.29M** test
- 평균 도서 ~411 KB → 긴 컨텍스트 평가에 이상적
- **전처리 없이 raw 바이트 그대로** (NUL, CR 포함)

### 5.2 CLEVR (멀티모달, 파인튜닝)

- 70K 합성 RGB 이미지 (480×320)
- ~700K Q&A 쌍, 28 unique answers, 13 question types
- **500K 바이트 컨텍스트**: flattened pixel bytes + UTF-8 question

---

## 6. 학습 절차

### 6.1 컴퓨트

- **8× NVIDIA A100 80GB**, data-parallel
- 모든 PG19 모델 = **360M 파라미터**로 맞춤 (스테이지별 layer 수 조정)
- 학습 데이터: 30B 바이트 (컨텍스트 ≤ 98K), **200B 바이트** (컨텍스트 ≥ 100K)
- **3D 5M 컨텍스트 모델: 단일 A100에서 15시간**, 100 GB 처리, PG19 test BPB **2.448**

### 6.2 하이퍼파라미터 (Table A4)

| 항목 | 사전학습 | 멀티모달 FT |
|------|---------|-----------|
| Peak LR | **1e-3** | 1e-4 |
| Gradient steps | 48 | 84 |
| Gradient clipping | 1.0 | 1.0 |
| Dropout | 0 | 0.1 |
| 옵티마이저 | AdamW $\beta = (0.9, 0.95)$ | 동일 |
| 스케줄 | 10% linear warmup → cosine | 동일 |
| 정밀도 | fp32 weights + AMP fp16 backward (FlashAttention) | 동일 |

LR 1e-3가 MegaByte 권장값보다 우수 (LR ablation으로 검증).

### 6.3 컨텍스트별 패치 설정 (Table A2)

| 컨텍스트 | 1D | 2D (g, l) | 3D (g1, g2, l) |
|---------|----|-----------|----------------|
| 8,192 | 8192 | 1024, 8 | 256, 8, 4 |
| 16,384 | 16384 | 2048, 8 | 512, 8, 4 |
| 32,768 | 32768 | 4096, 8 | 1024, 8, 4 |
| 98,304 | — | 8192, 12 | — |
| 1,048,576 | — | — | 8192, 16, 8 |
| **5,000,000** | — | — | **1000, 200, 25** |

**Gradient checkpoint chunking** (메모리 절감):
- 2D-100K → 2 chunks (stage 2)
- 3D-1M → 10 (stage 2), 20 (stage 3) → 75-80% GPU 활용

---

## 7. 추론 순서도

```
        프롬프트 → 컨텍스트 길이 L로 right-pad (<pad>=257)
                          │
                          ▼
              ┌───────────────────────┐
              │ 매 바이트 생성 루프      │
              └───────────┬───────────┘
                          ▼
        ┌────────────────────────────────────┐
        │ ① 입력 임베딩 + nested patches reshape│
        └────────────────┬───────────────────┘
                         ▼
        ┌────────────────────────────────────┐
        │ ② 모든 N개 global stage 순차 실행    │
        │    (각 stage는 이전 stage의           │
        │     project된 출력을 자기 임베딩에 더함)│
        └────────────────┬───────────────────┘
                         ▼
        ┌────────────────────────────────────┐
        │ ③ Local stage M_N                    │
        │    마지막 패치의 마지막 위치 logits 생성│
        └────────────────┬───────────────────┘
                         ▼
        ┌────────────────────────────────────┐
        │ ④ 다음 바이트 샘플 → append          │
        └────────────────┬───────────────────┘
                         │
              (생성 완료까지 반복)
```

### 추론의 핵심 한계

> **패치는 lossy 압축 표현 → KV cache 또는 Mamba RNN-mode가 계층 전체에서 깔끔하게 재사용 안 됨.**

모든 Mamba 블록이 매 스텝마다 parallel scan 재실행 → **바이트당 $O(L)$** (이상적인 $O(1)$이 아님).

그래도 **하이브리드(global Mamba + local Transformer)**가 1M 바이트까지 거의 선형 시간 (Fig. 6).

---

## 8. 실험 결과

### 8.1 메모리 스케일링 (단일 A100 80GB, 배치=2)

| 컨텍스트 | 1D-T | 2D-T | 3D-T |
|---------|------|------|------|
| 8K | 30.5 GB | 19.6 GB | 15.9 GB |
| 16K | 56.2 GB | 35.8 GB | 28.2 GB |
| 32K | **OOM** | 68.2 GB | **53.0 GB** |

**스테이지를 하나 추가할 때마다 학습 가능 시퀀스 길이가 거의 2배**.

### 8.2 PG19 — 98K 컨텍스트

| 계층 | (Global, Local) | Test PPL | Test BPB |
|------|-----------|---------|---------|
| **MegaByte (TT 베이스라인)** | T, T | 278.79 | 1.370 |
| MBLM 2D | S, T | 163.29 | 1.240 |
| **MBLM 2D** | **S, S** | **119.37** | **1.164** |

같은 패치 설정·같은 200 GB 데이터에서 MBLM이 MegaByte를 압도.

### 8.3 PG19 — 1M+ 컨텍스트

| 3D 구성 | Test PPL | Test BPB |
|---------|---------|---------|
| TTT | 5420.66 | 2.092 |
| **STT** | **5351.71** | **2.089** |

### 8.4 컨텍스트 외삽 (Fig. 7-8)

8K → 991K 바이트 평가. **장문 컨텍스트 모델은 더 긴 컨텍스트에서 단조 개선 안 됨** — 이는 PG19의 한계 (책 단위 예측은 local context면 충분, Llama-2-7B도 4K에서 saturation).

### 8.5 CLEVR 멀티모달 (Visual QA, Table 4)

| 모델 | Exists | Count | Cmp.Int | Cmp.Attr | Query | **All** |
|------|-------|-------|---------|---------|-------|---------|
| CNN+LSTM | 65.2 | 43.7 | 66.0 | 53.0 | 49.2 | **52.3** |
| 1D Transformer (DISC 3-bit) | 69.0 | 39.7 | 62.2 | 50.8 | 44.6 | **52.1** |
| 1D Mamba (DISC) | 68.7 | 38.5 | 63.4 | 49.9 | 43.4 | 51.6 |
| **1D Mamba (JPEG bytes)** | **72.0** | 39.3 | 64.1 | 51.2 | 36.5 | 50.3 |

**의의**:
- **이미지 인코더 없이** raw 바이트 → CNN+LSTM 수준 달성
- **JPEG 파일 바이트** 입력이 Exists +7%, Compare-Integer +7%
- **PG19 사전학습 → CLEVR로 positive transfer** (bGPT의 negative transfer 결과와 반대)
- 공간 추론은 약함 (1D raster flatten + 단방향 자회귀의 한계)

---

## 9. 다른 바이트 LM과 비교

| | MegaByte | MambaByte | BLT | **MBLM** |
|--|---------|-----------|-----|---------|
| 단계 | 2 (고정) | 1 | 2 | **N (무제한)** |
| 백본 | Transformer | Mamba | Transformer | **자유 선택** |
| 패칭 | 고정 | 없음 | **동적 (엔트로피)** | 고정 |
| 멀티모달 | 텍스트 | 텍스트 | 텍스트 | **텍스트+이미지** |
| 학습 가능 컨텍스트 | 1.2M | — | — | **5M (단일 GPU)** |

**MBLM만의 독보적 장점**:
1. **임의의 N단계** → 메모리 효율 극대화
2. **백본 혼합** → 스테이지별 최적 모델 선택 (예: global=Mamba, local=Transformer)
3. **단일 GPU에서 5M 바이트** 학습 가능 (계층 깊이 + chunking)
4. **첫 byte-level VQA** 성공

---

## 10. 한계

1. **공간 추론 약함** — 2D positional encoding 부재, raster flatten, 단방향
2. **효율적 hierarchical 추론 부재** — 패치 압축 때문에 KV cache/Mamba RNN-mode 재사용 불가 → 바이트당 선형 비용
3. **짧은 패치에서 local Mamba가 4배 느림** (backward) — parallel scan은 $L > 2K$에서만 FlashAttention 능가
4. **컨텍스트 외삽 평가의 한계** — PG19 자체가 4K 이상에서 의미 적음. needle-in-haystack 같은 적절한 long-context 벤치 필요
5. **CLEVR 과적합** — ~64% 에폭 후 성능 하락
6. **billion 파라미터 미검증**, tensor/sequence parallelism 미구현

---

## 11. 핵심 요약

| 질문 | 답 |
|------|-----|
| **무엇** | MegaByte를 N단계로 일반화 + Transformer/Mamba 자유 조합 |
| **데이터** | PG19 (200B 바이트), CLEVR (멀티모달) |
| **학습** | 8× A100, AdamW (β 0.9/0.95), LR 1e-3, cosine 스케줄 |
| **추론** | 매 바이트마다 N stage 순차 forward, chunking으로 메모리 |
| **최대 컨텍스트** | **5,000,000 바이트** (단일 A100, 15시간) |
| **결과** | MegaByte 대비 BPB 15% 개선, 첫 byte-level VQA 성공 |

> **한 줄**: "MegaByte의 2단계를 N단계로 일반화하고 Mamba/Transformer를 자유 조합 — 500만 바이트를 단일 GPU에서 다루는 바이트 LM 계보의 새 정점."

---

## 12. 관련 블로그 포스트

- [BLT-D 논문 리뷰](blt-d-review.md) — 또다른 바이트 LM 계열
- [Diffusion LM 서베이](diffusion-language-models-survey.md)
- [Mamba/SSM 계보 정리](mamba-family-complete-survey.md)
- [LLaDA 심화 리뷰](llada-review.md)

---

## 참고 자료

- [MBLM 논문 (arXiv:2502.14553)](https://arxiv.org/abs/2502.14553)
- [코드: github.com/ai4sd/multiscale-byte-lm](https://github.com/ai4sd/multiscale-byte-lm)
- [MegaByte (arXiv:2305.07185)](https://arxiv.org/abs/2305.07185)
- [MambaByte (arXiv:2401.13660)](https://arxiv.org/abs/2401.13660)
- [BLT (arXiv:2412.09871)](https://arxiv.org/abs/2412.09871)
