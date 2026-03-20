---
title: "[논문 리뷰] Mamba: 선택적 상태 공간 모델로 Transformer에 도전하다"
date: 2026-03-19
tags: ["논문리뷰", "Mamba", "SSM", "State Space Model"]
categories: ["ML/AI"]
summary: "Mamba 논문 리뷰. 입력 의존적 상태 전이(Selective SSM)로 Transformer의 어텐션 메커니즘 없이 동등한 성능을 달성한다. 선형 시간 복잡도와 하드웨어 친화적 설계가 핵심이다."
math: true
toc: true
draft: false
---

## 논문 정보

- **제목**: Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- **저자**: Albert Gu, Tri Dao
- **발표**: ICLR 2024 (arXiv: 2312.00752)
- **키워드**: State Space Model, Selective Mechanism, Linear Complexity

---

## 1. 배경: SSM의 한계와 Mamba의 등장

### State Space Model (SSM)이란?

연속 시간 시스템을 이산화한 시퀀스 모델:

$$h'(t) = A \cdot h(t) + B \cdot x(t)$$
$$y(t) = C \cdot h(t)$$

- $h(t)$: 은닉 상태 (state)
- $A$: 상태 전이 행렬
- $B$: 입력 투영
- $C$: 출력 투영

이산화 후:

$$h_t = \bar{A} \cdot h_{t-1} + \bar{B} \cdot x_t$$

$$y_t = C \cdot h_t + D \cdot x_t$$

### 기존 SSM의 문제: LTI (Linear Time-Invariant)

S4 등 기존 SSM은 $A$, $B$, $C$가 학습 후 **고정**이었다. 즉 모든 입력에 동일한 상태 전이 규칙이 적용된다.

이 LTI 속성 덕분에 **컨볼루션으로 변환**해 학습 시 병렬화가 가능했지만, 동시에 **내용 기반 추론(content-based reasoning)**이 불가능했다.

> 핵심 한계: "이 토큰이 중요한가?"를 판단할 수 없다.

---

## 2. Mamba의 핵심: 선택적 상태 전이

Mamba의 혁신은 $B$, $C$, $\Delta$를 **입력 의존적**으로 만든 것이다:

$$B(t) = \text{Linear}\_B(x(t))$$

$$C(t) = \text{Linear}\_C(x(t))$$

$$\Delta(t) = \text{softplus}(\text{Linear}\_\Delta(x(t)))$$

이산화:

$$\bar{A}(t) = \exp(\Delta(t) \cdot A)$$

$$\bar{B}(t) = (\Delta(t) \cdot A)^{-1} (\exp(\Delta(t) \cdot A) - I) \cdot \Delta(t) \cdot B(t)$$

상태 전이:

$$h(t) = \bar{A}(t) \cdot h(t-1) + \bar{B}(t) \cdot x(t)$$

$$y(t) = C(t) \cdot h(t) + D \cdot x(t)$$

### $\Delta$의 의미: 선택적 메모리

$\Delta(t)$는 **매 토큰마다 다른 시간 스케일**을 적용한다:

- **$\Delta$ 큼** → $\bar{A}(t) \approx 0$ → 이전 상태를 **잊고** 새 입력을 강하게 반영
- **$\Delta$ 작음** → $\bar{A}(t) \approx I$ → 이전 상태를 **보존**하고 새 입력을 약하게 반영

이것이 Transformer의 어텐션과 유사한 **내용 기반 선택** 메커니즘이다:
- 중요한 토큰 → $\Delta$ 크게 → 상태에 강하게 기록
- 불필요한 토큰 → $\Delta$ 작게 → 상태 유지, 입력 무시

---

## 3. 하드웨어 친화적 설계

### 문제: 선택적 SSM은 컨볼루션 불가

LTI SSM은 $A$, $B$, $C$가 고정이므로 글로벌 컨볼루션으로 변환해 $O(L \log L)$로 계산할 수 있었다. 그러나 입력 의존적 파라미터는 이 트릭을 깨뜨린다.

### 해결: Parallel Scan + Kernel Fusion

Mamba는 **parallel scan (prefix sum)** 알고리즘으로 순환 연산을 병렬화한다:

1. **커널 퓨전**: 이산화, 상태 전이, 출력 계산을 하나의 GPU 커널로 통합
2. **SRAM 활용**: HBM ↔ SRAM 간 IO를 최소화
3. **순환 모드**: 추론 시 $O(1)$ per step (상태만 유지)

결과적으로:
- **학습**: $O(L)$ 시간, FlashAttention과 유사한 IO 효율
- **추론**: $O(1)$ per token (RNN 모드), Transformer의 KV cache 없음

---

## 4. Mamba 블록 아키텍처

```
Input (B, L, D)
    │
    ├── Linear (D → 2·E)  ──→ SiLU ──→ Conv1d ──→ SSM ──→ ×
    │                                                        │
    └── Linear (D → 2·E)  ──→ SiLU ─────────────────────→ gate
                                                             │
                                                         Linear (E → D)
                                                             │
                                                          Output
```

핵심 구성:
- **Expand**: 입력 차원을 확장 (보통 2배)
- **Conv1d**: 짧은 범위의 로컬 패턴 포착
- **Selective SSM**: 장거리 의존성 + 선택적 메모리
- **Gated connection**: 정보 흐름 조절

---

## 5. 실험 결과

### 언어 모델링

| 모델 | Params | Pile PPL |
|------|--------|----------|
| Transformer++ | 1.4B | - |
| RWKV-4 | 1.5B | - |
| H3 | 1.3B | - |
| **Mamba** | **1.4B** | **최고 성능** |

Mamba-1.4B가 Transformer++ 2.8B와 동등한 성능을 달성. 즉 **절반의 파라미터로 동일 품질**.

### Selective Copying Task

LTI SSM이 실패하는 대표적 과제. 시퀀스 중간에 있는 특정 토큰만 기억하고 나머지는 무시해야 한다.

| 모델 | 정확도 |
|------|-------|
| S4 (LTI) | 실패 |
| H3 | 부분 성공 |
| **Mamba** | **완벽** |

### 추론 속도 (Throughput)

Mamba-1.4B의 추론 throughput은 Transformer++의 **5배** 수준. 시퀀스 길이가 길어질수록 격차가 커진다 (선형 vs 이차).

---

## 6. 왜 Mamba가 중요한가?

### Transformer 대비 장점

| 항목 | Transformer | Mamba |
|------|------------|-------|
| 시간 복잡도 | $O(L^2)$ | $O(L)$ |
| 추론 per token | $O(L)$ (KV cache) | $O(1)$ |
| 메모리 (추론) | KV cache 증가 | 고정 크기 state |
| 긴 시퀀스 | 비용 급증 | 선형 확장 |
| 하드웨어 효율 | IO bound | Kernel fusion |

### Attention과의 관계

Mamba의 selective mechanism은 사실 **소프트 어텐션의 일반화**로 볼 수 있다:
- Attention: 모든 이전 토큰에 대해 가중합 (명시적)
- Mamba: 상태 벡터에 선택적으로 정보 압축 (암묵적)

차이: Attention은 과거 전체에 접근 가능하지만 비용이 $O(L^2)$. Mamba는 고정 크기 상태로 압축하므로 $O(L)$이지만, 정보 손실 가능성이 있다.

---

## 7. 후속 연구

### Mamba-2 (2024)

- Structured State Space Duality (SSD): SSM = structured masked attention임을 증명
- 더 효율적인 병렬 알고리즘
- Mamba-1 대비 2-8배 빠른 학습

### Jamba (AI21, 2024)

- Transformer + Mamba 하이브리드
- Attention 레이어와 Mamba 레이어를 교차 배치
- 52B 파라미터, 256K 컨텍스트

### Vision Mamba, Video Mamba 등

- 이미지/비디오 도메인으로 확장
- ViT 대비 선형 복잡도로 고해상도 처리

---

## 8. 한계와 열린 질문

1. **In-context learning**: Transformer 대비 ICL 능력이 다소 약하다는 보고
2. **정보 압축 손실**: 고정 크기 상태로는 모든 과거 정보를 완벽히 보존 불가
3. **하이브리드의 필요성**: Jamba처럼 Attention과 결합할 때 최고 성능
4. **학습 안정성**: 대규모(70B+)에서의 학습 안정성 추가 검증 필요

---

## 9. 개인적 생각

Mamba는 "Attention is All You Need" 이후 가장 임팩트 있는 아키텍처 논문 중 하나다. 특히:

- **$\Delta$를 통한 선택적 메모리** 아이디어가 깔끔하다. "무엇을 기억하고 무엇을 잊을지"를 입력 자체가 결정하게 한 것은 생물학적 뉴런의 게이팅과도 유사하다.
- **하드웨어 수준의 최적화**를 논문에서 직접 다룬 것이 실용적이다.
- SSM → Selective SSM의 전환이 LTI → LTV (Linear Time-Varying)라는 오래된 제어이론 개념의 재발견이라는 점이 흥미롭다.

다만 순수 Mamba만으로는 Transformer를 완전히 대체하기 어렵고, **하이브리드 아키텍처**가 현실적 방향이라는 것이 현재까지의 합의인 것 같다.

바이트 수준 언어 모델(SancMamba)의 기반으로 Mamba를 선택한 이유도 여기에 있다 — 바이트 시퀀스는 토큰 시퀀스보다 3~6배 길어지므로, 선형 복잡도가 필수적이다.
