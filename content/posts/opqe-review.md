---
title: "[논문 리뷰] OPQE — HyDE에 강화학습을 결합한 On-Policy Pseudo-document Query Expansion"
date: 2026-04-22
tags: ["논문리뷰", "RAG", "검색", "HyDE", "강화학습", "PPO", "OPQE"]
categories: ["ML/AI"]
summary: "OPQE 논문 상세 리뷰. HyDE의 가상 문서 생성에 PPO 기반 on-policy RL을 결합하여 검색 성능을 직접 최적화한다. RL 쿼리 재작성(DeepRetrieval)과 달리 sparse/dense 검색기 모두에서 일관된 성능 향상."
math: true
toc: true
draft: false
---

## 논문 정보

- **제목**: Rethinking On-policy Optimization for Query Augmentation
- **약어**: OPQE = **O**n-**P**olicy **P**seudo-document **Q**uery **E**xpansion
- **저자**: Zhichao Xu, Shengyao Zhuang, Xueguang Ma, Bingsen Chen, Yijun Tian, Fengran Mo, Jie Cao, Vivek Srikumar
- **arXiv**: 2510.17139 (2025.10, 2026.03 개정)
- **코드**: [github.com/BaleChen/RethinkQAug](https://github.com/BaleChen/RethinkQAug)

---

## 1. 한 줄 요약

> **HyDE의 가상 문서 생성 아이디어를 유지하면서, 그 생성 과정 자체를 PPO 강화학습으로 검색 성능에 맞춰 직접 최적화한다.**

---

## 2. 배경: HyDE와 그 한계

### HyDE (Hypothetical Document Embeddings)

[HyDE](https://arxiv.org/abs/2212.10496) (Gao et al., 2022)의 핵심 아이디어:

```
질의: "트랜스포머의 어텐션이란?"
       ↓ LLM에게 "이 질의의 가상 답변을 작성하라" 프롬프트
가상 문서: "어텐션 메커니즘은 쿼리와 키의 내적으로..."
       ↓ 가상 문서를 임베딩
       ↓ 임베딩으로 유사도 검색
검색 결과
```

**왜 효과적인가**:
- 짧은 질의 vs 긴 문서의 **비대칭** 문제를 해결
- 질의↔문서 매칭을 **문서↔문서 매칭**으로 전환 (Contriever 학습 분포와 일치)
- LLM 내부 지식으로 어휘 불일치 완화

### HyDE의 한계

1. **순수 프롬프팅** — 학습 없음, LLM의 zero-shot 능력에 의존
2. **유창한 ≠ 검색에 좋은**: "그럴듯해 보이는 문서"가 항상 좋은 검색 표현은 아님
3. **Sparse 검색기(BM25)에서 부정적 효과**: 장황한 가상 문서가 **잡음 단어**를 추가 → 어휘 매칭 방해
4. **태스크/코퍼스 적응 불가**: LLM의 고정된 지식에 갇힘

---

## 3. 대안: DeepRetrieval (RL 쿼리 재작성)

[DeepRetrieval](https://arxiv.org/abs/2503.00223)은 다른 방향:

```
질의 → LLM이 RL로 학습된 쿼리 재작성 → 검색
```

**장점**: 도메인 특화 BM25에서 강함 (예: FiQA 금융)

**단점**:
- 가상 문서의 풍부한 신호 포기
- **Dense 검색기에서 오히려 성능 하락** (키워드 최적화가 임베딩 정렬을 깨뜨림)
- 데이터셋 의존적

### OPQE의 위치

> **HyDE의 강점 (가상 문서) + DeepRetrieval의 강점 (RL 최적화)** = OPQE

```
[프롬프팅만] HyDE/SPQE
                       ↘
                         OPQE (가상 문서 + RL)
                       ↗
[RL 재작성] DeepRetrieval
```

---

## 4. OPQE 아키텍처

### 4.1 전체 흐름

```
질의 q
  ↓
LLM 정책 π_θ
  ↓ (rollout)
<think>...추론...</think>
<answer>가상 문서 d'</answer>
  ↓ (concat)
[질의 q + 가상 문서 d']
  ↓
검색기 R (BM25 또는 Contriever)
  ↓
검색 결과
  ↓ → 보상 r 계산 → PPO 업데이트 → π_θ
```

### 4.2 정책의 액션 공간

DeepRetrieval은 "재작성된 질의" 를 출력 → 짧은 액션 공간
**OPQE는 "가상 문서"를 출력** → 풍부한 액션 공간 + HyDE의 강점 보존

---

## 5. RL 학습 상세

### 5.1 알고리즘: PPO

KL 정규화 PPO 목적함수:

$$\mathcal{L}_{KL}(\theta) = \mathbb{E}_{q' \sim \pi_\theta(\cdot | q)} \left[ r_{\text{retrieval}}(q') - \beta \cdot D_{KL}(\pi_\theta(\cdot|q) \| \pi_{\text{ref}}(\cdot|q)) \right]$$

- $\pi_\theta$: 학습 중인 정책 LLM
- $\pi_{\text{ref}}$: frozen 참조 정책 (KL 발산 페널티)
- $\beta$: KL 강도

### 5.2 보상 함수 (곱셈 결합)

$$r(q') = r_{\text{format}}(q') \cdot r_{\text{retrieval}}(q')$$

**Format 보상** (이진 0/1):
- `<think>...</think><answer>...</answer>` 구조 강제

**Retrieval 보상** (태스크별):

| 과제 | 보상 |
|------|------|
| Evidence-seeking (NQ, TriviaQA, SQuAD) | **순위 기반 계단 보상**: rank ≤ 5 → 5.0, rank > 3000 → −2.5 |
| Ad-hoc retrieval (BEIR, MS MARCO) | **NDCG@10** |
| Tool retrieval (ToolRet) | **Completeness@10** |

### 5.3 핵심 신호: concat(query, pseudo_doc)으로 검색

```
검색 입력 = [원본 질의] + [LLM이 생성한 가상 문서]
              ↓
검색기 (BM25 / Contriever)
              ↓
검색 결과의 품질이 보상으로 LLM에게 피드백
```

LLM은 **검색 성능을 높이는 가상 문서를 생성하도록** 학습된다.

### 5.4 학습 설정

| 항목 | 값 |
|------|-----|
| 베이스 모델 | Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct |
| RL 알고리즘 | PPO (GAE + KL) |
| Learning rate | 1e-6 |
| Batch size | 128 |
| Rollout temperature | 0.6 |
| GPU | 7-14× A100 40GB |
| 프레임워크 | verl (DeepRetrieval 코드 베이스) |

### 5.5 Warm-Start 효과

논문 Figure 1의 핵심 발견:

```
DeepRetrieval (재작성):  낮은 시작점 → 천천히 상승
SPQE/HyDE (프롬프트만):   높은 시작점 → 학습 안 함
OPQE:                  높은 시작점 (HyDE처럼) → RL로 추가 상승
```

가상 문서가 이미 좋은 출발점이므로 RL이 빠르게 수렴.

---

## 6. 실험 결과

### 6.1 평가 데이터

| 과제 | 데이터셋 | 코퍼스 | 메트릭 |
|------|---------|--------|--------|
| Evidence-seeking | NQ, TriviaQA, SQuAD | Wikipedia-2018 (21M) | Hit@20 |
| Ad-hoc | FEVER, HotpotQA, NFCorpus, SciFact, MS MARCO, DL19, DL20 | BEIR | NDCG@10 |
| Tool retrieval | ToolRet (Web/Code/Custom) | 200K+ | Completeness@10 |

### 6.2 평균 결과

**Evidence-seeking (BM25, Hit@20)**:
| 방법 | 점수 |
|------|------|
| Base (질의만) | 70.2 |
| DeepRetrieval-3B | 75.0 |
| DeepRetrieval-7B | 75.6 |
| **SPQE (HyDE-like)** | **78.2** |

**Ad-hoc (BM25, NDCG@10)**:
| 방법 | 점수 |
|------|------|
| Base | 46.6 |
| DeepRetrieval-7B | 49.0 |
| **SPQE** | **51.6** |

**Ad-hoc (Dense Contriever, NDCG@10)**:
| 방법 | 점수 |
|------|------|
| Base | 56.0 |
| DeepRetrieval-7B | 57.5 |
| SPQE | 56.6 |
| **OPQE-7B** | **58.1** |

**Tool Retrieval (Completeness@10)**:
| 방법 | BM25 | Dense |
|------|------|------|
| Base | 29.4 | 29.6 |
| DeepRetrieval-7B | - | **29.4 (악화!)** |
| **SPQE** | **35.1** | **34.8** |

---

## 7. 핵심 발견 4가지

### Finding 1: 컴퓨트 보정 시 결론이 뒤집힌다

큰 LLM(GPT-4o-mini, Qwen3-32B, GPT-OSS-120B)으로 단순 프롬프팅(SPQE/HyDE)을 하면, **비싼 RL 방법(DeepRetrieval)을 종종 능가**한다.

→ "RL이 진짜 필요한가?"의 의문 제기

### Finding 2: RL의 효과는 검색기 의존적

| 검색기 | RL 쿼리 재작성 효과 |
|--------|---------------|
| BM25 (sparse) | **개선** (어휘 매칭 강화) |
| Dense (embedding) | **악화** (키워드 최적화가 임베딩 정렬 깨뜨림) |

### Finding 3: OPQE만 양쪽에서 일관된 개선

```
DeepRetrieval: BM25 ↑, Dense ↓
SPQE/HyDE:     BM25 ↑, Dense ≈
OPQE:          BM25 ↑, Dense ↑  ← 유일한 양쪽 개선
```

가상 문서 생성을 액션 공간으로 유지한 덕분에 dense 악화 패턴 회피.

### Finding 4: 가상 문서가 효과적인 3가지 이유

1. **어휘 불일치 완화**: LLM 내부 지식으로 동의어/관련 용어 확장
2. **비대칭 → 대칭 변환**: 짧은 질의↔긴 문서 → 긴 가상 문서↔긴 실제 문서 (Contriever 학습 분포와 일치)
3. **풍부한 컨텍스트**: 재작성된 짧은 질의보다 더 많은 신호 제공

---

## 8. HyDE vs DeepRetrieval vs OPQE 종합 비교

| 항목 | HyDE / SPQE | DeepRetrieval | **OPQE** |
|------|-----------|--------------|---------|
| **방법** | 프롬프팅 (zero-shot) | RL 쿼리 재작성 | **RL 가상 문서 생성** |
| **학습** | 없음 | PPO | **PPO** |
| **액션 공간** | 가상 문서 | 짧은 재작성 쿼리 | **가상 문서** |
| **출력 길이** | 길다 | 짧다 | **길다** |
| **BM25 효과** | 큼 | 큼 | **큼** |
| **Dense 효과** | 보통 | **악화** | **개선** |
| **태스크 적응** | 불가 | 가능 | **가능** |
| **계산 비용** | 추론만 | 학습 필요 | 학습 필요 |
| **시작점 (warm start)** | - | 낮음 | **높음** |

---

## 9. 의의와 한계

### 의의

1. **HyDE 계보의 진화**: 프롬프팅 → 학습으로 자연스러운 다음 단계
2. **양 검색기 호환**: Sparse/Dense 모두에서 작동
3. **Warm-start 학습**: HyDE 출발점에서 RL 미세조정 → 효율적
4. **Format reward**: `<think>` 추론 구조 강제로 안정성 확보

### 한계

1. **다중 GPU PPO 학습 필요**: 7-14× A100
2. **백본 비교 불공정**: SPQE는 큰 프롬프팅 LLM, OPQE는 학습된 3B/7B → 직접 비교 어려움
3. **논문의 메타 메시지가 미묘함**: 사실 논문은 "단순 프롬프팅이 과소평가되었다"가 주된 주장. OPQE는 RL을 쓸 거면 이게 최고라는 정도
4. **데이터 의존성**: 학습 데이터에 따른 일반화 성능 변동

---

## 10. RAG 학습 패러다임에서의 위치

```
[프롬프팅 기반]
  ├── HyDE — 가상 문서 zero-shot
  ├── Query2Doc — 유사 접근
  └── Promptagator — few-shot 질의 생성

[RL 기반 검색 최적화]
  ├── DeepRetrieval — 쿼리 재작성 RL
  ├── ★ OPQE — 가상 문서 RL ← 여기
  ├── Search-R1 — 검색 시점 결정 RL
  └── CCS — 정답 없는 RL

[지도학습 기반]
  ├── Self-RAG — 반성 토큰 SFT
  ├── CRAG — T5 평가기 SFT
  └── RAFT — 방해 문서 혼합 SFT
```

---

## 11. 실무 시사점

### 언제 OPQE를 쓸까?

```
☐ 도메인이 좁고 LLM의 zero-shot HyDE가 부족함
☐ Sparse + Dense 검색을 모두 사용
☐ A100 multi-GPU 자원 보유
☐ 학습 데이터 (질의, 정답 문서) 보유
→ OPQE 채택
```

### 언제 그냥 HyDE를 쓸까?

```
☐ 학습 인프라 부족
☐ 일반 도메인 (Wikipedia 등)
☐ 큰 LLM (GPT-4 등) 사용 가능
☐ 빠른 프로토타입
→ SPQE/HyDE로 충분 (논문의 메타 메시지)
```

### 언제 DeepRetrieval을 쓸까?

```
☐ Sparse(BM25) 검색기만 사용
☐ 매우 좁은 도메인 (예: 금융, 의료)
☐ 짧은 키워드 쿼리가 자연스러움
→ DeepRetrieval 고려
```

---

## 12. 핵심 한 줄 요약

> **OPQE = HyDE(가상 문서) + PPO(검색 보상) — sparse와 dense 검색기 양쪽에서 일관되게 작동하는 첫 RL 기반 query augmentation.**

---

## 13. 관련 블로그 포스트

- [Search-R1 상세 리뷰](search-r1-review.md) — 검색 시점 RL
- [CCS 상세 리뷰](cycle-consistent-search-review.md) — 정답 없는 RL
- [DPO 상세 리뷰](dpo-review.md) — RL 단순화의 또 다른 길
- [RAG 동향 총정리](rag-survey-2026.md) — HyDE 등 전반
- [RAG End-to-end 학습 서베이](rag-end-to-end-training-survey.md) — REPLUG, RA-DIT 등

---

## 참고 자료

- [OPQE 논문 (arXiv:2510.17139)](https://arxiv.org/abs/2510.17139)
- [OPQE 코드 (github.com/BaleChen/RethinkQAug)](https://github.com/BaleChen/RethinkQAug)
- [HyDE 원논문 (arXiv:2212.10496)](https://arxiv.org/abs/2212.10496)
- [DeepRetrieval (arXiv:2503.00223)](https://arxiv.org/abs/2503.00223)
- [Query2Doc, Promptagator 등은 합성 데이터 서베이 참조](rag-synthetic-data-survey.md)
