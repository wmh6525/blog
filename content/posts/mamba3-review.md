---
title: "[논문 리뷰] Mamba-3 (Hymba): Mamba와 Attention의 하이브리드 아키텍처"
date: 2026-03-19
tags: ["논문리뷰", "Mamba", "SSM", "Hymba", "하이브리드"]
categories: ["ML/AI"]
summary: "Mamba 계열의 최신 발전을 정리한다. Hymba(NVIDIA)를 중심으로 SSM-Attention 하이브리드 아키텍처의 설계 원리, Mamba-1/2와의 차이, 그리고 순수 Transformer 대비 성능을 분석한다."
math: true
toc: true
draft: false
---

## 1. Mamba 계열의 진화 흐름

| 세대 | 핵심 기여 | 발표 |
|------|----------|------|
| **Mamba-1** | Selective SSM (입력 의존적 $\Delta$) | 2023.12 |
| **Mamba-2** | SSD (SSM = Structured Attention 증명) | 2024.05 |
| **Hymba** | SSM + Attention 하이브리드, 헤드 내 융합 | 2024.11 |

"Mamba-3"라는 공식 논문은 없지만, NVIDIA가 발표한 **Hymba**가 Mamba 계열의 최신 하이브리드 아키텍처로서 실질적인 다음 단계이다.

---

## 2. Hymba: 왜 하이브리드인가?

### SSM과 Attention의 상보성

| 능력 | SSM (Mamba) | Attention |
|------|------------|-----------|
| 장거리 의존성 | 상태 압축 (정보 손실 가능) | **완전 접근** |
| 계산 복잡도 | **$O(L)$** | $O(L^2)$ |
| In-context learning | 약함 | **강함** |
| 추론 효율 | **$O(1)$ per token** | $O(L)$ (KV cache) |
| 로컬 패턴 | 보통 | 보통 |

어느 쪽도 단독으로 완벽하지 않다. **SSM의 효율성**과 **Attention의 표현력**을 결합하는 것이 자연스러운 방향이다.

---

## 3. Hymba 아키텍처

### 핵심 설계: 헤드 내 SSM-Attention 융합

기존 하이브리드(Jamba 등)는 SSM 레이어와 Attention 레이어를 **교차 배치**했다:

```
[Mamba] → [Attention] → [Mamba] → [Attention] → ...
```

Hymba는 다르다. **하나의 레이어 안에서** SSM과 Attention을 동시에 수행하고 결합한다:

```
Input
  ├── SSM heads (일부 헤드)
  │     └── Mamba-2 SSD 연산
  └── Attention heads (나머지 헤드)
        └── 표준 causal attention

  → 두 출력을 concat/합산 → Output
```

### 장점

- SSM 헤드가 **효율적 장거리 문맥 압축**
- Attention 헤드가 **정밀한 토큰 간 관계** 포착
- 같은 레이어에서 두 관점이 동시에 작동 → 정보 흐름이 풍부

### Learnable Meta Tokens

Hymba는 **학습 가능한 메타 토큰**을 도입한다. 이는 KV cache에 추가되는 고정 토큰으로:
- 모든 위치에서 접근 가능한 **글로벌 메모리** 역할
- Attention의 초기 토큰 편향(attention sink) 문제 해소
- 적은 수(보통 128개)로도 성능 개선

---

## 4. 성능 비교

### 소규모 모델 (1.5B)

| 모델 | Params | 평균 벤치마크 | Cache 크기 |
|------|--------|-------------|-----------|
| Llama-3.2 | 1B | 기준 | 100% |
| Mamba-2 | 1.3B | Llama-3.2 대비 열등 | 매우 작음 |
| **Hymba** | **1.5B** | **Llama-3.2-3B 능가** | **~20%** |

Hymba-1.5B가 2배 큰 Llama-3.2-3B보다 높은 성능을 보인다.

### 캐시 효율

| 모델 | KV Cache (상대) |
|------|----------------|
| Transformer | 100% |
| Mamba | ~0% (상태만) |
| **Hymba** | **~20%** (Attention 헤드만) |

SSM 헤드는 KV cache가 불필요하므로, Attention 헤드 비율만큼만 캐시를 사용한다.

---

## 5. 하이브리드 아키텍처 비교

| 모델 | 방식 | SSM 종류 | 특징 |
|------|------|---------|------|
| **Jamba** (AI21) | 레이어 교차 | Mamba-1 | MoE 결합, 52B |
| **Zamba** (Zyphra) | 공유 Attention | Mamba-1 | Attention 레이어 공유로 효율화 |
| **Griffin** (Google) | 게이트 결합 | Gated Linear Recurrence | RLHF까지 검증 |
| **Hymba** (NVIDIA) | **헤드 내 융합** | Mamba-2 SSD | 메타 토큰, 가장 세밀한 통합 |

---

## 6. SSM-Attention 하이브리드의 설계 원칙

Mamba-1/2/Hymba의 진화에서 도출되는 설계 원칙:

### 원칙 1: SSM 비율 > Attention 비율

대부분의 시퀀스 처리는 SSM으로 충분하다. Attention은 **소수의 헤드/레이어**에만 배치하되, ICL(in-context learning) 같은 정밀한 추론에 집중시킨다.

### 원칙 2: 같은 레이어에서 융합

레이어 교차보다 **헤드 수준 융합**이 더 효과적이다. 같은 입력에 대해 SSM과 Attention이 동시에 다른 관점의 표현을 생성하고 결합한다.

### 원칙 3: 글로벌 메모리

Learnable meta tokens이나 register tokens 같은 **글로벌 메모리**가 SSM의 정보 압축 손실을 보완한다.

### 원칙 4: KV cache 최적화

SSM 헤드는 KV cache 불필요 → 전체 캐시 사용량을 Attention 헤드 비율에 비례하여 줄일 수 있다.

---

## 7. 앞으로의 방향

### 스케일링

- Hymba는 아직 1.5B 수준에서만 검증. **7B, 13B, 70B+**로의 확장 필요
- Jamba-1.5가 52B까지 확장했으나, Hymba의 헤드 내 융합이 대규모에서도 유효한지 미검증

### 학습 효율

- SSD 알고리즘의 chunk-wise 계산이 하이브리드에서도 효율적으로 작동하는가?
- Attention 헤드와 SSM 헤드의 최적 비율은?

### 추론 최적화

- SSM 헤드: 순환 모드 ($O(1)$)
- Attention 헤드: KV cache 필요 ($O(L)$)
- 두 모드를 **동시에 효율적으로** 서빙하는 인프라 필요

### 특화 도메인

- **코드 생성**: 긴 파일 컨텍스트 → SSM의 선형 복잡도 유리
- **다중 문서 QA**: 정밀한 참조 → Attention 필수
- **실시간 스트리밍**: SSM의 $O(1)$ 추론이 핵심

---

## 8. 개인적 생각

Mamba 계열의 진화는 명확한 방향성을 보여준다:

1. **Mamba-1**: "Attention 없이도 된다" → 선택적 SSM의 가능성 증명
2. **Mamba-2**: "사실 SSM과 Attention은 같은 것이다" → 이론적 통합
3. **Hymba**: "둘 다 쓰되, 최적으로 결합하자" → 실용적 해답

"SSM vs Attention" 경쟁은 사실상 **"어떻게 결합하느냐"**의 문제로 수렴하고 있다. Hymba의 헤드 내 융합은 현재까지 가장 세밀한 결합 방식이며, 메타 토큰이라는 글로벌 메모리 개념도 깔끔하다.

장기적으로는 Mamba-2의 SSD 이론이 시사하듯, SSM과 Attention을 **하나의 파라미터화된 연산**으로 통합하는 방향으로 갈 가능성이 있다 — 학습 과정에서 각 레이어/헤드가 스스로 "SSM적"으로 작동할지 "Attention적"으로 작동할지 결정하는 연속적 스펙트럼.

SancMamba 프로젝트에서도 바이트 수준의 긴 시퀀스를 처리하기 위해 Mamba 기반 아키텍처를 사용하고 있는데, Hymba 스타일의 하이브리드를 고려할 만하다 — 특히 의미 단위 경계 탐지에서 Attention의 정밀한 토큰 간 관계 포착이 도움될 수 있다.
