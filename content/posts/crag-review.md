---
title: "[논문 리뷰] CRAG — 검색 결과가 잘못되었을 때 스스로 교정하는 RAG"
date: 2026-04-03
tags: ["논문리뷰", "RAG", "LLM", "CRAG"]
categories: ["ML/AI"]
summary: "CRAG(Corrective Retrieval Augmented Generation) 논문 상세 리뷰. 경량 T5 평가기로 검색 품질을 Correct/Incorrect/Ambiguous 3단계로 분류하고, 각 상황에 맞는 교정 전략(지식 정제, 웹 검색 폴백, 결합)을 적용하여 RAG의 취약점을 보완한다."
math: true
toc: true
draft: false
---

## 논문 정보

- **제목**: Corrective Retrieval Augmented Generation
- **저자**: Shi-Qi Yan (USTC), Jia-Chen Gu (UCLA), Yun Zhu (Google DeepMind), Zhen-Hua Ling (USTC)
- **arXiv**: 2401.15884 (2024.01)
- **코드**: [github.com/HuskyInSalt/CRAG](https://github.com/HuskyInSalt/CRAG)

---

## 1. 문제: RAG는 검색이 틀리면 오히려 더 나빠진다

기존 RAG의 근본적 취약점:

$$P(Y|X) = P(D|X) \cdot P(Y, D|X)$$

검색기($R$)와 생성기($G$)가 **낮은 오류 허용도로 밀접 결합**되어 있다. 검색이 실패하면 무관한 문서가 프롬프트에 주입되어, 검색 없이 생성하는 것보다 **오히려 할루시네이션이 악화**될 수 있다.

### 구체적 예시

**질의**: "Death of a Batman의 각본가는?"

**검색 결과**: 1989년 배트맨 영화 관련 문서 (브루스 웨인의 부모 이야기)

**RAG 답변**: "Hamm" (오답) — 무관한 문서에서 이름을 잘못 추출

> **핵심 문제**: 기존 RAG는 검색 결과의 품질과 무관하게 **무차별적으로** 문서를 프롬프트에 삽입한다.

---

## 2. CRAG의 해결: 3단계 교정 메커니즘

### 전체 파이프라인

```
질의 x → 검색기 → 문서 D = {d_1, ..., d_k}
                    ↓
              검색 평가기 E
                    ↓
        ┌──────────┼──────────┐
        ↓          ↓          ↓
    [CORRECT]  [AMBIGUOUS]  [INCORRECT]
        ↓          ↓          ↓
   지식 정제    정제 + 웹검색   웹 검색
        ↓          ↓          ↓
        └──────────┼──────────┘
                   ↓
              생성기 G → 응답 y
```

### 알고리즘

```
1. 각 (질의, 문서) 쌍에 대해 관련성 점수 계산: score_i = E(x, d_i)
2. 전체 점수에서 신뢰도 판정:

   IF max(scores) > 상한 임계값 → CORRECT
   IF max(scores) < 하한 임계값 → INCORRECT
   ELSE → AMBIGUOUS

3. 판정에 따른 조치:
   CORRECT    → k = Knowledge_Refine(x, D)
   INCORRECT  → k = Web_Search(Rewrite(x))
   AMBIGUOUS  → k = Knowledge_Refine(x, D) + Web_Search(Rewrite(x))

4. 생성기에 k를 입력하여 응답 생성
```

---

## 3. 검색 평가기 (Retrieval Evaluator)

### 아키텍처

- **모델**: T5-large (**0.77B** 파라미터)
- Self-RAG의 비평 모델(LLaMA-2 7B, 인스트럭션 튜닝 필요)보다 **10배 작음**

### 학습

- **데이터**: PopQA (14K 샘플)
- **양성**: 질의 + 골드 Wikipedia 문서 (label = 1)
- **음성**: 질의 + 유사하지만 무관한 검색 결과 (label = -1)
- **출력**: 관련성 점수 $[-1, 1]$

### 평가기 정확도

| 방법 | 정확도 |
|------|-------|
| ChatGPT (직접 프롬프트) | 58.0% |
| ChatGPT (CoT) | 62.4% |
| ChatGPT (few-shot) | 64.7% |
| **T5 평가기 (CRAG)** | **84.3%** |

**0.77B T5가 ChatGPT를 22%p 능가** — 도메인 특화 파인튜닝의 힘.

### 신뢰도 임계값

| 데이터셋 | 상한 | 하한 |
|---------|------|------|
| PopQA | 0.59 | -0.99 |
| PubHealth, ARC | 0.50 | -0.91 |
| Biography | 0.95 | -0.91 |

---

## 4. 세 가지 교정 전략

### 4.1 CORRECT → 지식 정제 (Knowledge Refinement)

관련 문서라도 **노이즈가 포함**되어 있으므로, 핵심 정보만 추출한다.

**Decompose-then-Recompose**:

```
문서 → [분해] → 개별 스트립 (문장 단위)
          ↓
      [필터링] → T5 평가기로 각 스트립의 관련성 점수 계산
          ↓       (임계값 -0.5 이하 제거)
      [재조합] → 상위 5개 스트립 연결 → 내부 지식 (k_in)
```

### 4.2 INCORRECT → 웹 검색 폴백

모든 검색 문서를 **폐기**하고 웹 검색으로 대체:

```
질의 → ChatGPT가 검색 키워드로 재작성
    → Google Search API (top-5 URL)
    → Wikipedia 페이지 우선
    → HTML 파싱 (<p> 태그)
    → 동일한 지식 정제 적용
    → 외부 지식 (k_ex)
```

**쿼리 재작성 프롬프트**: "다음 대화와 질문에서 웹 검색 쿼리로 사용할 키워드를 최대 3개 추출하세요."

### 4.3 AMBIGUOUS → 결합

내부 지식 + 외부 지식 **모두 활용**:

$$k = k_{in} + k_{ex}$$

**설계 근거**: 이진 분류(Correct/Incorrect)만 사용하면 평가기의 정확도에 과도하게 의존. Ambiguous 전략이 시스템을 **안정화**시킨다.

---

## 5. 실험 결과

### 5.1 주요 벤치마크 (SelfRAG-LLaMA2-7b 생성기)

| 방법 | PopQA | Bio (FactScore) | PubHealth | ARC-Challenge |
|------|-------|----------------|----------|--------------|
| RAG | 52.8 | 59.2 | 39.0 | 53.2 |
| Self-RAG | 54.9 | 81.2 | 72.4 | 67.3 |
| **CRAG** | **59.8** | **74.1** | **75.6** | **68.6** |
| **Self-CRAG** | **61.8** | **86.2** | **74.8** | 67.2 |

**CRAG vs RAG 개선폭**:
- PopQA: **+7.0%**
- PubHealth: **+36.6%** (가장 극적)
- ARC: **+15.4%**
- Bio: **+14.9** FactScore

### 5.2 바닐라 LLaMA2-hf-7b (인스트럭션 튜닝 없음)

| 방법 | PopQA | PubHealth | ARC |
|------|-------|----------|-----|
| RAG | 50.5 | 48.9 | 43.4 |
| Self-RAG | 29.0 | 0.7 | 23.9 |
| **CRAG** | **54.9** | **59.5** | **53.7** |

**핵심**: Self-RAG는 바닐라 LLaMA2에서 처참하게 실패(0.7%!)하지만, CRAG는 **Fine-tuning 없이도** 잘 작동한다.

### 5.3 연산 오버헤드

| 방법 | TFLOPs/토큰 | 실행 시간(s) |
|------|-----------|-----------|
| RAG | 26.5 | 0.363 |
| **CRAG** | **27.2** | **0.512** |
| Self-RAG | 26.5~132.4 | 0.741 |
| Self-CRAG | 27.2~80.2 | 0.908 |

CRAG는 RAG 대비 **0.7 TFLOPs, ~0.15초**만 추가. Self-RAG(최대 132.4 TFLOPs)보다 훨씬 가벼움.

---

## 6. Ablation Study

### 각 전략 제거 효과 (PopQA, SelfRAG-LLaMA2-7b)

| 구성 | CRAG | Self-CRAG |
|------|------|---------|
| 전체 | 59.8 | 61.8 |
| -Correct | 58.3 (-1.5) | 59.6 (-2.2) |
| -Incorrect | 59.5 (-0.3) | 60.8 (-1.0) |
| -Ambiguous | 59.0 (-0.8) | 61.5 (-0.3) |

### 각 지식 활용 연산 제거 효과

| 구성 | CRAG |
|------|------|
| 전체 | 59.8 |
| -지식 정제 | 54.2 (**-5.6**) |
| -쿼리 재작성 | 56.2 (-3.6) |
| -스트립 선택 | 58.6 (-1.2) |

**지식 정제(Decompose-then-Recompose)가 가장 큰 기여** (-5.6%p).

### 웹 검색 보충 vs CRAG

| 구성 | 정확도 |
|------|-------|
| RAG | 52.8 |
| RAG + 웹 검색 | 53.8 (+1.0) |
| **CRAG** | **59.8 (+7.0)** |

단순히 웹 검색을 추가하는 것(+1.0)보다 **CRAG의 교정 메커니즘이 훨씬 효과적**(+7.0).

---

## 7. Self-RAG vs CRAG 비교

| 항목 | Self-RAG | CRAG |
|------|---------|------|
| **접근법** | LLM에 반성 토큰 학습 | 외부 평가기 + 교정 파이프라인 |
| **Generator 수정** | **인스트럭션 튜닝 필수** | **불필요** (plug-and-play) |
| **평가 모델** | LLaMA-2 7B (비평 모델) | T5-large **0.77B** (10배 작음) |
| **바닐라 LLM 호환** | 매우 나쁨 (0.7% PubHealth) | **잘 작동** |
| **연산 비용** | 최대 132.4 TFLOPs | **27.2 TFLOPs** |
| **결합 가능** | - | **Self-CRAG**로 결합 시 추가 개선 |

**CRAG의 핵심 장점**: 어떤 생성기와도 조합 가능한 **플러그 앤 플레이** 교정 레이어.

---

## 8. 전체 아키텍처에서의 위치

```
기존 RAG:   검색기 ──────────────────→ 생성기
CRAG:       검색기 → [검색 평가기] → [교정 레이어] → 생성기
                      ↓                ↓
                  Correct/Incorrect/   지식 정제 or
                  Ambiguous 판정      웹 검색 폴백
```

CRAG는 **검색기와 생성기 사이에 삽입**되는 교정 레이어이므로, 기존 RAG 파이프라인에 최소 침습적으로 통합 가능하다.

---

## 9. 한계

1. **외부 평가기 파인튜닝 필수**: T5 평가기를 도메인별로 학습해야 함
2. **도메인 전이 한계**: PopQA에서 학습한 평가기가 과학 질의(ARC)에서 88.3%를 Ambiguous로 분류
3. **엔터티 정렬 편향**: 평가기가 의미적 관련성보다 **엔터티 매칭**에 의존 (재현 연구에서 발견)
4. **경험적 임계값**: 신뢰도 임계값을 데이터셋별로 수동 설정
5. **외부 API 의존**: ChatGPT (쿼리 재작성), Google Search API
6. **Ambiguous 정확도**: Ambiguous 케이스에서 19.3%만 정확 (가장 약한 고리)

---

## 10. Adaptive RAG과의 관계

| | CRAG | Self-RAG | Adaptive RAG |
|--|------|---------|-------------|
| **초점** | 검색 **후** 교정 | 검색 **여부** 결정 + 비평 | 검색 **전략** 선택 |
| **시점** | 검색 결과 평가 시 | 전체 생성 과정 | 질의 분류 시 |
| **상호 보완** | Adaptive RAG → CRAG 순서로 적용 가능 | CRAG와 결합 = Self-CRAG | CRAG의 전단계로 사용 가능 |

**완전한 파이프라인**: Adaptive RAG(질의 복잡도 라우팅) → 검색 → **CRAG(교정)** → Self-RAG(반성적 생성)
