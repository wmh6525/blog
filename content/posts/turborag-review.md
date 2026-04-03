---
title: "[논문 리뷰] TurboRAG — KV 캐시 사전 계산으로 RAG 추론 9.4배 가속"
date: 2026-04-03
tags: ["논문리뷰", "RAG", "LLM", "최적화"]
categories: ["ML/AI"]
summary: "TurboRAG 논문 상세 리뷰. 청크별 KV 캐시를 오프라인 사전 계산하고, Independent Attention + Reordered RoPE로 프리필 연산을 제거하여 TTFT를 최대 9.4배 줄인다."
math: true
toc: true
draft: false
---

## 논문 정보

- **제목**: TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text
- **저자**: Songshuo Lu, Hua Wang, Yutian Rong, Zhi Chen, Yaohua Tang
- **소속**: Moore Threads AI
- **학회**: EMNLP 2025
- **코드**: [github.com/MooreThreads/TurboRAG](https://github.com/MooreThreads/TurboRAG)

---

## 1. 문제: RAG의 프리필 병목

기존 RAG 시스템의 추론 파이프라인:

```
질의 → 검색(top-k 청크) → 청크들을 프롬프트에 연결 → 프리필(KV 캐시 계산) → 생성
```

**프리필이 병목인 이유**:
1. 프리필 비용이 입력 길이에 대해 **이차적**으로 증가
2. 동일한 청크가 다른 질의에서 반복 검색되어도 **매번 재계산**
3. 긴 연결 문서의 프리필이 GPU 자원을 독점하여 배치 크기 제한

> TurboRAG의 질문: "프리필을 오프라인으로 미리 해놓을 수는 없을까?"

---

## 2. 두 가지 핵심 관찰

### 관찰 1: 청크 간 Cross-Attention은 극히 희소하다

어텐션 맵을 시각화하면, 각 문서 청크는 **자기 자신 내부**에만 주로 어텐드한다. 서로 다른 청크 간의 어텐션 점수는 매우 낮다.

→ 청크 간 cross-attention을 **제거해도** 성능 하락이 미미하다.

### 관찰 2: RoPE는 상대 위치만 중요하다

RoPE(Rotary Position Embedding)의 어텐션 점수는 두 토큰의 **상대 거리**에만 의존한다:

$$\text{RoPE}(q_t, k_s) = f(q, k, t - s)$$

→ 각 청크를 **독립적으로** 프리필해도, 나중에 위치를 재배열하면 올바른 상대 위치를 복원할 수 있다.

---

## 3. TurboRAG 아키텍처

### 3.1 Independent Attention

문서 간 cross-attention을 완전히 제거한 어텐션 마스크:

```
기존 Causal Attention:        Independent Attention:
┌──────────────┐              ┌──────────────┐
│█             │              │█             │
│██            │              │██            │
│███           │              │ ██           │
│████          │              │ ███          │
│█████         │              │   ██         │
│██████        │              │   ███        │
│███████       │              │█████████     │  ← 질의는 모든 청크에 어텐드
│████████      │              │██████████    │
└──────────────┘              └──────────────┘
  청크 A  청크 B  질의            청크 A  청크 B  질의
```

각 청크는 자기 자신에만 어텐드. 질의/응답은 모든 청크와 자기 자신에 어텐드.

### 3.2 Reordered RoPE

**문제**: 각 청크를 독립적으로 프리필하면 모든 KV 캐시의 position ID가 $[0, 1, \ldots, l]$로 동일. 단순 연결 시 위치가 겹침.

**해결**: KV 캐시를 RoPE 적용 **전**(raw 상태)으로 저장. 연결 후 연속 position ID $[0, 1, \ldots, kl]$로 RoPE 적용.

```python
# Key를 raw 상태로 캐시에 저장 (RoPE 미적용)
key_states, value_states = past_key_value.update(key_states, value_states)

# 연결 후 연속 position ID로 RoPE 적용
full_position_ids = torch.arange(0, past_key_value.seen_tokens)
key_states = apply_rotary_pos_emb(key_states, cos, sin, full_position_ids)
```

### 3.3 전체 파이프라인

```
=== 오프라인 (1회) ===
각 청크 c_i → LLM 프리필 → KV 캐시 저장 (RoPE 미적용 상태)

=== 온라인 (매 질의) ===
질의 → 검색 (top-k 청크)
     → 저장된 KV 캐시 k개 로드
     → 연결 + Reordered RoPE 적용
     → 질의 토큰만 프리필 (~128 토큰)
     → 즉시 생성 시작!
```

**핵심**: 프리필 대상이 전체 컨텍스트(~8192 토큰)에서 질의 토큰(~128 토큰)으로 축소. **연산량 98.46% 감소**.

---

## 4. Fine-tuning

Independent Attention + Reordered RoPE 패턴에 모델을 적응시키기 위한 SFT 수행.

- **베이스 모델**: Qwen2-7B
- **하드웨어**: 32× A100 80GB
- **데이터**: 18개 Document QA 데이터셋 (~190K 샘플) + 일반 대화/추론/코드
- **학습률**: 1e-5, AdamW

---

## 5. 실험 결과

### 5.1 Document QA 정확도 (RGB 벤치마크)

5개 문서 검색, Noise Ratio = 불필요 문서 비율.

| 모델 | NR=0.2 | NR=0.4 | NR=0.6 | NR=0.8 | 평균 |
|------|--------|--------|--------|--------|------|
| GPT-4o | 99.0 | 99.3 | 98.3 | 96.3 | 98.2 |
| Naive RAG | 99.7 | 99.3 | 99.3 | 94.3 | 98.2 |
| TurboRAG (FT 전) | 98.0 | 97.3 | 90.7 | 85.7 | 92.9 |
| **TurboRAG (FT 후)** | **99.0** | **98.3** | **96.0** | **93.7** | **96.8** |

파인튜닝 후 Naive RAG 대비 **1.4%p 이내** 격차.

### 5.2 TTFT 속도 향상 (LongBench)

| 데이터셋 | 컨텍스트 토큰 | Naive TTFT | Turbo TTFT | 속도 향상 |
|---------|------------|-----------|-----------|---------|
| musique | 16,349 | 1,610ms | 171ms | **9.4배** |
| 2wikimqa | 7,553 | 709ms | 101ms | **7.0배** |
| dureader | 10,642 | 1,007ms | 116ms | **8.7배** |
| hotpotqa | 13,453 | 1,333ms | 147ms | **9.1배** |
| **평균** | **11,999** | **1,165ms** | **134ms** | **8.6배** |

### 5.3 배치 확장 (h2d 전송 제외 시)

| 배치 | Naive TTFT | Turbo TTFT | 속도 향상 |
|------|-----------|-----------|---------|
| 1 | 711ms | 44ms | **16.1배** |
| 4 | 2,842ms | 97ms | **29.3배** |
| 8 | 5,812ms | 177ms | **32.8배** |

KV 캐시가 GPU 메모리에 프리페치되면 **최대 32.8배** 가속.

### 5.4 범용 능력 저하 없음

| 벤치마크 | Naive RAG | TurboRAG | 차이 |
|---------|----------|---------|------|
| MMLU | 69.57 | 70.73 | +1.16 |
| GSM-8K | 79.12 | 79.45 | +0.33 |
| MATH | 39.54 | 40.58 | +1.04 |

---

## 6. 스토리지 오버헤드

Qwen2-7B, 청크 512 토큰 기준:

$$\text{KV 캐시/청크} = 2 \times 28 \times 4 \times 128 \times 512 \times 2 \approx \textbf{29.4 MB}$$

원본 텍스트(~2KB) 대비 **~15,000배** 스토리지. 이것이 TurboRAG의 핵심 트레이드오프: **공간 ↔ 시간**.

---

## 7. 한계

1. **스토리지 비용**: 청크당 ~30MB KV 캐시 저장 필요
2. **h2d 전송 병목**: CPU→GPU KV 캐시 전송이 프리페치 없으면 속도 향상 제한 (~4배)
3. **Cross-attention 손실**: 문서 간 상호 참조가 필요한 복잡한 multi-hop 추론에서 약점
4. **Fine-tuning 필수**: 기존 모델에 바로 적용 시 정확도 하락
5. **모델 업데이트 시 재계산**: LLM이 업데이트되면 모든 KV 캐시 재계산 필요
