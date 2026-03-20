---
title: "[논문 리뷰] Hymba: SSM과 Attention을 헤드 수준에서 융합한 하이브리드 아키텍처"
date: 2026-03-20
tags: ["논문리뷰", "Mamba", "SSM", "Hymba", "하이브리드"]
categories: ["ML/AI"]
summary: "NVIDIA Hymba 논문 리뷰. SSM과 Attention을 레이어가 아닌 헤드 수준에서 융합하고, 학습 가능한 메타 토큰으로 글로벌 메모리를 도입한다. Hymba-1.5B가 Llama-3.2-3B를 능가하면서 캐시는 ~20%만 사용한다."
math: true
toc: true
draft: false
---

## 논문 정보

- **제목**: Hymba: A Hybrid-head Architecture for Small Language Models
- **저자**: Xin Dong, Yonggan Fu, Shizhe Diao, Wonmin Byeon, Zijia Chen, Ameya Sunil Mahabaleshwarkar, Shih-Yang Liu, Matthijs Douze, Zangwei Zheng, Jan Kautz, Pavlo Molchanov
- **소속**: NVIDIA
- **발표**: 2024.11 (arXiv: 2411.13676)
- **키워드**: Hybrid Architecture, SSM-Attention Fusion, Meta Tokens, Small Language Models

---

## 1. 동기: SSM과 Attention, 각자의 한계

### SSM과 Attention의 상보성

| 능력 | SSM (Mamba) | Attention |
|------|------------|-----------|
| 장거리 의존성 | 상태 압축 (정보 손실 가능) | **완전 접근** |
| 계산 복잡도 | **$O(L)$** | $O(L^2)$ |
| In-context learning | 약함 | **강함** |
| 추론 효율 | **$O(1)$ per token** | $O(L)$ (KV cache) |
| 로컬 패턴 | 보통 | 보통 |

어느 쪽도 단독으로 완벽하지 않다. **SSM의 효율성**과 **Attention의 표현력**을 결합하는 것이 자연스러운 방향이다.

### 기존 하이브리드의 한계

Jamba(AI21), Griffin(Google) 등 기존 하이브리드는 SSM 레이어와 Attention 레이어를 **교차 배치**한다:

```
[Mamba] → [Attention] → [Mamba] → [Attention] → ...
```

이 방식의 문제:
- 같은 입력에 대해 SSM 관점과 Attention 관점이 **다른 레이어에서 분리**되어 처리
- 두 관점의 정보가 합류하려면 다음 레이어까지 기다려야 함
- 최적의 SSM/Attention 레이어 배치 비율을 찾기 어려움

---

## 2. Hymba의 핵심: 헤드 내 융합

### 같은 레이어에서 SSM과 Attention 동시 수행

Hymba는 **하나의 레이어 안에서** SSM과 Attention을 동시에 수행하고 결합한다:

```
Input
  ├── SSM heads (일부 헤드)
  │     └── Mamba-2 SSD 연산
  └── Attention heads (나머지 헤드)
        └── 표준 causal attention

  → 두 출력을 concat/합산 → Output
```

이것은 Multi-Head Attention에서 각 헤드가 다른 subspace를 학습하는 것과 유사하되, 일부 헤드는 **SSM으로**, 나머지는 **Attention으로** 작동한다.

### 장점

- SSM 헤드가 **효율적 장거리 문맥 압축** 담당
- Attention 헤드가 **정밀한 토큰 간 관계** 포착
- 같은 레이어에서 두 관점이 동시에 작동 → **정보 흐름이 풍부**
- 레이어 교차보다 **세밀한 통합** 가능

---

## 3. Learnable Meta Tokens

Hymba의 두 번째 혁신은 **학습 가능한 메타 토큰**이다.

### 개념

입력 시퀀스 앞에 고정된 학습 가능 토큰을 추가한다. 이들은 KV cache에 포함되어:

- 모든 위치에서 접근 가능한 **글로벌 메모리** 역할
- 일종의 "항상 참조 가능한 요약 정보"

### Attention Sink 문제 해소

Transformer에서 관찰되는 **attention sink** 현상 — 모델이 첫 번째 토큰에 비정상적으로 높은 어텐션을 주는 문제 — 을 메타 토큰이 자연스럽게 흡수한다. 메타 토큰이 "기본 어텐션 대상"이 되어 의미 없는 첫 토큰 편향을 방지한다.

### 효과

- 적은 수(보통 128개)로도 유의미한 성능 개선
- 추가 파라미터/계산 비용 미미
- 학습 과정에서 자동으로 유용한 정보를 인코딩

---

## 4. 아키텍처 세부 사항

### 전체 구조

각 Hymba 레이어:

1. **입력 프로젝션**: $X \to Q, K, V$ (Attention용) + SSM 입력
2. **병렬 처리**:
   - SSM 헤드: Mamba-2의 SSD 알고리즘으로 처리
   - Attention 헤드: 표준 causal self-attention (+ meta tokens)
3. **출력 합산**: 두 종류의 헤드 출력을 concat → 프로젝션
4. **FFN**: SwiGLU 기반 feed-forward

### SSM 헤드 vs Attention 헤드 비율

Hymba는 SSM 헤드를 **다수**, Attention 헤드를 **소수**로 배치한다. 대부분의 시퀀스 처리는 SSM으로 충분하고, Attention은 ICL 등 정밀한 추론에만 필요하기 때문이다.

이 비율은 모델 크기와 용도에 따라 조정 가능하다.

---

## 5. 실험 결과

### 소규모 모델 성능 (1.5B)

| 모델 | Params | 평균 벤치마크 | Cache 크기 |
|------|--------|-------------|-----------|
| Llama-3.2 | 1B | 기준 | 100% |
| Mamba-2 | 1.3B | Llama-3.2 대비 열등 | 매우 작음 |
| **Hymba** | **1.5B** | **Llama-3.2-3B 능가** | **~20%** |

Hymba-1.5B가 **2배 큰** Llama-3.2-3B보다 높은 성능을 보인다.

### 캐시 효율

| 모델 | KV Cache (상대) |
|------|----------------|
| Transformer | 100% |
| Mamba (순수 SSM) | ~0% (상태만) |
| **Hymba** | **~20%** (Attention 헤드만) |

SSM 헤드는 KV cache가 불필요하므로, Attention 헤드 비율만큼만 캐시를 사용한다. 이는 긴 시퀀스에서 **메모리 절약**에 큰 이점이 된다.

### 벤치마크별 성능

Hymba는 다양한 벤치마크에서 일관된 우위를 보인다:

- **상식 추론**: SSM의 장거리 문맥 + Attention의 정밀 추론
- **수학/코드**: Attention 헤드가 정확한 토큰 매칭 담당
- **긴 문맥 이해**: SSM 헤드의 효율적 압축 + 메타 토큰의 글로벌 정보

---

## 6. 하이브리드 아키텍처 비교

| 모델 | 방식 | SSM 종류 | 특징 |
|------|------|---------|------|
| **Jamba** (AI21) | 레이어 교차 | Mamba-1 | MoE 결합, 52B 확장 |
| **Zamba** (Zyphra) | 공유 Attention | Mamba-1 | Attention 레이어 공유로 효율화 |
| **Griffin** (Google) | 게이트 결합 | Gated Linear Recurrence | RLHF까지 검증 |
| **Hymba** (NVIDIA) | **헤드 내 융합** | Mamba-2 SSD | 메타 토큰, 가장 세밀한 통합 |

Hymba의 차별점:
1. **레이어가 아닌 헤드 수준 융합** — 같은 입력에 두 관점 동시 적용
2. **메타 토큰** — 글로벌 메모리로 SSM 정보 손실 보완
3. **Mamba-2 SSD 기반** — 최신 SSM 알고리즘 활용

---

## 7. 설계 원칙 정리

Hymba에서 도출되는 SSM-Attention 하이브리드 설계 원칙:

### 원칙 1: SSM 비율 > Attention 비율

대부분의 시퀀스 처리는 SSM으로 충분하다. Attention은 **소수의 헤드**에만 배치하되, ICL 같은 정밀한 추론에 집중시킨다.

### 원칙 2: 같은 레이어에서 융합

레이어 교차보다 **헤드 수준 융합**이 더 효과적이다. 같은 입력에 대해 SSM과 Attention이 동시에 다른 관점의 표현을 생성하고 결합한다.

### 원칙 3: 글로벌 메모리 도입

Learnable meta tokens 같은 **글로벌 메모리**가 SSM의 정보 압축 손실을 보완한다.

### 원칙 4: KV cache 최적화

SSM 헤드는 KV cache 불필요 → 전체 캐시 사용량을 Attention 헤드 비율에 비례하여 줄일 수 있다.

---

## 8. 한계와 열린 질문

1. **스케일링**: 1.5B에서 검증 — 7B, 13B, 70B+에서의 유효성 미확인
2. **최적 헤드 비율**: SSM과 Attention 헤드의 최적 비율이 과제/규모에 따라 달라질 수 있음
3. **메타 토큰 수**: 128개가 최적인지, 과제에 따라 조정해야 하는지
4. **추론 인프라**: SSM 헤드(순환 모드)와 Attention 헤드(KV cache)를 동시에 효율적으로 서빙하는 인프라 필요

---

## 9. 개인적 생각

Hymba는 "SSM vs Attention" 논쟁에 대한 가장 실용적인 답을 제시한다: **둘 다 쓰되, 최적으로 결합하라**.

가장 인상적인 점:

1. **헤드 수준 융합**이라는 간단하면서도 효과적인 아이디어. Multi-Head Attention의 각 헤드가 이미 다른 subspace를 학습하므로, 일부를 SSM으로 대체하는 것은 자연스러운 확장이다.

2. **메타 토큰**이 attention sink 문제까지 해결하는 부수 효과. 단순한 글로벌 메모리로 도입했는데 기존 Transformer의 알려진 문제도 함께 해결된다.

3. **1.5B로 3B를 이기면서 캐시는 1/5** — 소형 모델에서의 효율성이 모바일/엣지 배포에 직접적으로 유용하다.

Mamba-3와 함께 보면, SSM 계열의 발전이 두 축으로 진행되고 있음을 알 수 있다:
- **Mamba-3**: SSM 자체의 이론적 기반 강화 (이산화, 복소수, MIMO)
- **Hymba**: SSM과 Attention의 최적 결합 방법론

장기적으로는 이 두 축이 합쳐져, Mamba-3 수준의 개선된 SSM을 Hymba 스타일로 Attention과 융합하는 방향으로 갈 것으로 예상된다.
