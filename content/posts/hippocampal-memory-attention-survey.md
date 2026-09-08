---
title: "[서베이] 해마를 어떻게 구현할 것인가 — attention 옆에 기억을 따로 두는 방법들"
date: 2026-09-08
tags: ["서베이", "해마", "에피소드기억", "메모리증강", "Attention", "Hopfield", "NeuroAI", "논문리뷰"]
categories: ["ML/AI"]
summary: "의미 표상을 attention과 별도로 저장해 해마 기능을 모델링하는 방법을 정리한다. 출발점은 'attention이 이미 CA3'라는 세 편의 결과(Ramsauer의 Hopfield 동치, Bricken의 SDM 근사, Whittington의 장소·격자세포 재현)다. 따라서 새로 만들 것은 패턴 완성이 아니라 패턴 분리(DG)·색인·시간 인접 검색·공고화다. Memorizing Transformers, EM-LLM, Titans, HippoRAG, Larimar를 해마 하위영역에 대응시키고, 대부분의 시스템이 빠뜨리는 DG 층의 최소 구현을 제시한다."
math: true
toc: true
draft: false
---

## 0. 문제 설정

트랜스포머에 "사람의 해마 같은 것"을 붙이고 싶다고 하자. 자연스러운 첫 발상은 이렇다.

> attention을 계산할 때 의미 표상을 **어딘가 따로 저장**해 두고, 나중에 꺼내 쓴다.

방향은 맞다. 그런데 설계를 시작하기 전에 확인해야 할 것이 하나 있다. **attention이 이미 해마의 일부를 하고 있다.** 무엇을 새로 만들어야 하는지는 그것을 확정한 뒤에야 정해진다.

이 글의 구조:

| 절 | 내용 |
|---|---|
| §1 | attention = CA3 패턴 완성 — 세 편의 독립적 결과 |
| §2 | 해마를 계산 요소로 분해 (색인 이론과 CLS) |
| §3 | 저장 계열 4가지 — KV 외부화 · 사건 분절 · 파라미터 기록 · 색인 그래프 |
| §4 | 해마 하위영역 ↔ 기존 시스템 대응표 |
| §5 | 대부분이 빠뜨리는 것 — 패턴 분리(DG)와 그 최소 구현 |
| §6 | 설계 결정 5가지 |
| §7 | 무엇부터 해볼 것인가 |

---

## 1. attention은 이미 CA3다

### 1.1 Ramsauer et al. 2020 — Hopfield 업데이트와의 동치

**"Hopfield Networks is All You Need"** — [arXiv:2008.02217](https://arxiv.org/abs/2008.02217).

연속 상태(continuous-state) modern Hopfield network를 정의하고 다음을 보인다.

| 성질 | 내용 |
|---|---|
| 용량 | **지수적으로 많은** 패턴 저장 |
| 검색 | **1회 업데이트**로 패턴 복원 |
| 오차 | 지수적으로 작은 검색 오차 |
| ★ 핵심 | **이 업데이트 규칙이 트랜스포머 attention과 동치** |

에너지 극소점은 세 종류로 나뉜다.

```
① 전역 고정점    — 모든 패턴을 평균     → 정보 없음
② 준안정 상태    — 일부 패턴을 평균     → 부분집합 검색
③ 단일 패턴 고정점 — 하나의 패턴 저장    → ★ 패턴 완성 ★
```

그리고 실측: 트랜스포머·BERT는 **하위 층에서는 전역 평균 영역**, **상위 층에서는 준안정 상태**로 동작한다. 층마다 기억 장치로서의 성격이 다르다는 뜻이다 — §6.2의 "어느 층에서 뽑을 것인가"가 여기서 나온다.

**해마와의 연결**: 부분 단서로 전체 기억을 복원하는 **패턴 완성(pattern completion)** 은 해마 **CA3**의 순환 어트랙터가 하는 일이다. Hopfield 네트워크는 그 어트랙터의 고전적 모델이다. 따라서 attention은 CA3의 계산을 이미 수행하고 있다.

### 1.2 Bricken & Pehlevan 2021 — Sparse Distributed Memory 근사

**"Attention Approximates Sparse Distributed Memory"** — NeurIPS 2021, [arXiv:2111.05498](https://arxiv.org/abs/2111.05498).

Kanerva의 **SDM**은 생물학적으로 그럴듯한 연상기억 모델이다. 이 논문은 **특정 데이터 조건에서 attention이 SDM에 근사**함을 보이고, 그 조건이 **사전학습된 GPT-2에서 실제로 충족됨**을 확인한다.

§1.1이 "attention = 어트랙터 검색"이라면, 이쪽은 "attention = 주소 기반 연상기억"이다. 둘 다 같은 방향을 가리킨다.

### 1.3 Whittington, Warren & Behrens 2022 — 장소세포와 격자세포가 나온다

**"Relating transformers to models and neural representations of the hippocampal formation"** — ICLR 2022, [arXiv:2112.04035](https://arxiv.org/abs/2112.04035).

**이 글에서 가장 실용적인 논문이다.** 결과가 놀랍도록 저렴하기 때문이다.

```
  트랜스포머 + ★순환 위치 인코딩(recurrent position encoding)★
        │
        └─▶ 해마 형성체의 정밀하게 조율된 공간 표상이 재현됨
              · 장소세포(place cells)
              · 격자세포(grid cells)
```

저자들의 평가:

- 이 결과가 현재의 신경과학 해마 모델들과 밀접하게 관련되며
- 트랜스포머 버전은 신경과학 버전 대비 **극적인 성능 향상**을 보이고
- 이것이 **해마 색인 이론(hippocampal indexing theory)의 한 구현**이며
- 트랜스포머 위치 인코딩의 역할에 대한 새로운 통찰을 준다

**설계 함의**: 거대한 외부 메모리를 붙이기 **전에**, 위치 인코딩 방식만 바꿔도 해마스러운 표상이 나온다. 가장 싼 실험이다.

### 1.4 §1의 결론

```
  attention이 이미 하는 것        →  CA3 패턴 완성, 연상 검색
  ─────────────────────────────────────────────────────────
  따로 만들어야 하는 것            →  패턴 분리 (DG)
                                    색인 (indexing)
                                    시간 인접 검색
                                    공고화 (consolidation)
```

"의미 표상을 따로 저장한다"는 아이디어를 **패턴 완성을 또 만드는 것**으로 구현하면 중복이다. 나머지 넷을 겨냥해야 한다.

---

## 2. 해마를 계산 요소로 분해

### 2.1 해부학적 회로

```
        신피질 (경험의 내용이 실제로 사는 곳)
              │  ▲
              ▼  │
     EC (내후각피질) ──── 인터페이스. 격자세포. 구조 정보
              │
              ▼
     DG (치상회)  ──── ★ 패턴 분리 ★
              │        확장 투사 + 희소 코딩
              │        비슷한 경험을 서로 다른 코드로 갈라놓는다
              ▼
     CA3        ──── 패턴 완성. 순환 어트랙터        ← §1: attention이 이미 함
              │
              ▼
     CA1 / 해마이행부 ── 비교·출력. 신피질로 되돌림
              │
              ▼
     [ 재생 · 공고화 ] ── 해마 → 신피질 (느린 학습)
```

### 2.2 해마 색인 이론 — 설계의 핵심 전제

**Teyler & DiScenna 1986** 원안, **Teyler & Rudy 2007** 개정 — *Hippocampus*, ["The hippocampal indexing theory and episodic memory: Updating the index"](https://onlinelibrary.wiley.com/doi/abs/10.1002/hipo.20350).

핵심 주장:

> 해마는 행동 에피소드의 개별 특징들이 만들어낸 **신피질 활동에 대한 정보를 포착**하도록 기능적으로 설계되고 해부학적으로 배치되어 있다. 그리고 해마가 그 신피질 영역들로 **되돌아가는 투사**를 갖기 때문에, 해마가 저장한 정보는 **부분 단서로 기억을 인출하는 색인(index)** 역할을 할 수 있다.

2007년 개정판의 결론은 이 이론이 "매우 잘 늙었다(aged very well)"는 것이었다.

**이것이 왜 결정적인가**:

```
  ❌ 흔한 구현:  의미 표상을 ★복제해서★ 외부 저장소에 넣는다
                  → 저장소가 신피질의 사본이 된다. 용량이 선형 증가.

  ✅ 색인 이론:  해마는 ★내용을 저장하지 않는다★
                  포인터 + 결합(binding)만 저장
                  회상 시 그 포인터로 신피질 패턴을 ★재활성화★
```

"의미 표상을 따로 저장한다"는 요구를 **가장 해마답게** 만족시키는 방법은, 표상을 복사하는 것이 아니라 **원문 위치에 대한 포인터와, 그것을 다시 찾게 해줄 분리된 키**를 저장하는 것이다.

### 2.3 상보 학습 시스템 (CLS) — 왜 두 개가 필요한가

**Kumaran, Hassabis & McClelland 2016**, "What Learning Systems do Intelligent Agents Need? Complementary Learning Systems Theory Updated" — *Trends in Cognitive Sciences*. ([PubMed](https://pubmed.ncbi.nlm.nih.gov/27315762/))

| 시스템 | 담당 | 학습 속도 |
|---|---|---|
| **신피질** | 일반화된 지식, 구조 | 느림 (교차 학습 필요) |
| **해마** | 특정 경험(instance) | 빠름 (1회 노출) |

논문의 논지:

> 자연·인공 학습 시스템 **둘 다** 특정 경험을 저장하는 두 번째 시스템의 이점을 얻는다. 이 시스템으로부터의 **재생(replay)** 이 교차 학습을 지원하며, 재생은 **보상이나 참신성(novelty)에 의해 조절**될 수 있다.

**왜 두 개가 필요한가**: 하나의 망에 빠르게 쓰면 **파국적 망각**이 일어난다. 빠른 시스템이 경험을 붙잡아 두었다가 느린 시스템에 **교차해서 재생**해 주어야 구조 학습이 망가지지 않는다.

그리고 마지막 문장이 중요하다 — 재생이 **참신성에 의해 조절**된다. §3.2와 §3.3의 놀람(surprise) 기반 게이팅이 여기서 신경과학적 근거를 얻는다.

### 2.4 TEM — 구조와 내용의 분리

**Whittington et al. 2020, "The Tolman-Eichenbaum Machine"** — *Cell*. ([링크](https://www.cell.com/cell/fulltext/S0092-8674(20)31388-X))

§1.3 논문의 신경과학 원본이다. 핵심 원리:

```
  내측 EC  →  ★구조★ 지식의 기저 (관계, 전이 규칙)
  외측 EC  →  ★감각★ 표상 (내용)
  해마     →  둘을 ★결합(conjunction)★
```

> **경험 사이의 관계를 각 경험의 내용으로부터 인수분해(factorization)** 하는 것이, 구조 지식을 새로운 상황으로 일반화하는 강력한 기제를 제공한다.

학습 후 TEM의 EC 세포는 격자·띠·경계·물체벡터 세포를, 해마 세포는 환경 간 재사상(remapping)하는 장소·랜드마크 세포를 보인다. 그리고 검증 가능한 예측을 냈다 — **해마 재사상은 무작위가 아니며 구조 지식은 환경을 가로질러 보존된다.** 동시 기록된 장소·격자세포에서 확인되었다.

**설계 함의**: 저장 키를 만들 때 **"무엇"과 "어디/언제/어떤 관계"를 분리**하라. 합쳐 놓으면 새 상황으로 일반화되지 않는다.

---

## 3. 저장 계열 4가지

### 3.1 계열 A — KV를 외부로 빼기

#### kNN-LM (Khandelwal et al., ICLR 2020)

**"Generalization through Memorization: Nearest Neighbor Language Models"** — [arXiv:1911.00172](https://arxiv.org/abs/1911.00172).

계보의 출발점. 사전학습 LM을 **kNN 모델과 선형 보간**한다. 이웃은 사전학습 LM 임베딩 공간의 거리로 계산하며, 데이터스토어는 **어떤 텍스트 컬렉션이든** 될 수 있다.

- WIKITEXT-103에서 perplexity **15.79** (2.9점 개선), **추가 학습 없음**
- 데이터스토어만 바꾸면 **도메인 적응**이 된다

#### ★ Memorizing Transformers (Wu et al., ICLR 2022)

**"Memorizing Transformers"** — [arXiv:2203.08913](https://arxiv.org/pdf/2203.08913). Yuhuai Wu, Markus Rabe, DeLesley Hutchins, Christian Szegedy (Google).

**질문에 가장 직접적으로 답하는 논문이다.**

```
  [ 특정 층 하나에서 ]

   지역 문맥 ────▶ 표준 dense self-attention          ──┐
                                                        ├──▶ ★학습된 게이트★
   외부 메모리 ──▶ 과거 (key, value) 쌍에 대한 근사 kNN ──┘        │
                   ★ 비미분(non-differentiable) ★                 ▼
                                                              최종 출력
```

| 특성 | 내용 |
|---|---|
| 저장 대상 | 과거 입력의 **내부 표상 (K, V 쌍)** |
| 그래디언트 | **흐르지 않음**. 메모리는 학습 대상이 아님 |
| 결합 | 학습된 게이트가 지역/장기 중 무엇을 볼지 **동적으로 결정** |
| 규모 | **262K 토큰**까지, 계산 부담 증가 미미 |
| 효과 | 코드·수학 논문 등 여러 벤치마크에서 perplexity 개선 |

**"attention과 검색을 통합한다(unifying attention and retrieval)"** 는 것이 논문의 자기 규정이다.

구현: [lucidrains/memorizing-transformers-pytorch](https://github.com/lucidrains/memorizing-transformers-pytorch)

**해마 대응**: 게이트가 CA1(비교·출력)에 가깝다. 다만 **DG가 없다** — 저장 전 패턴 분리 단계가 빠져 있다. §5에서 다룬다.

### 3.2 계열 B — 사건 단위로 분절해 저장

#### ★ EM-LLM (Fountas et al., ICLR 2025)

**"Human-inspired Episodic Memory for Infinite Context LLMs"** — [arXiv:2407.09450](https://arxiv.org/abs/2407.09450) · [프로젝트](https://em-llm.github.io/) · [코드](https://github.com/em-llm/EM-LLM-model). Huawei Noah's Ark Lab + UCL.

**이 글의 목적에 가장 잘 맞는 최신 연구다.** 그리고 **파인튜닝이 필요 없다.**

```
[ 인코딩 ]
  ① 베이지안 놀람(Bayesian surprise)으로 토큰 스트림의 ★사건 경계★ 탐지
  ② 그래프 이론적 경계 정련(boundary refinement)                  ← 온라인 수행
        └─▶ 토큰들이 응집적인 ★에피소드 사건★ 으로 조직됨

[ 인출 — 2단계 ]
  ③ 유사도 기반 검색
  ④ ★시간적 인접성(temporal contiguity)★ 기반 검색
```

성능:

| 항목 | 결과 |
|---|---|
| LongBench · ∞-Bench | SOTA 검색 모델 **InfLLM을 일관되게 상회** |
| full-context 모델 대비 | **대부분 과제에서 상회** |
| 규모 | **1,000만 토큰** 검색 — full-context로는 계산 불가능한 규모 |
| ★ 인지적 타당성 | 사건 분절이 **인간이 지각하는 사건 경계와 강하게 상관** |

**④ 시간적 인접성이 왜 중요한가**: 인간의 자유회상에서는 한 항목을 떠올리면 **시간적으로 이웃했던 항목이 함께 떠오른다**(temporal contiguity effect). 순수 유사도 검색은 이 구조를 만들지 못한다. 해마 기능을 표현하는 것이 목적이라면 이 축을 빼면 안 된다.

**①의 놀람 기반 분절**은 이 블로그의 [Bayesian Surprise](../bayesian-surprise/) 글과 직접 이어진다. 그리고 §2.3에서 본 CLS의 "재생은 참신성에 의해 조절된다"와 같은 원리다.

### 3.3 계열 C — 파라미터에 쓰는 기억

#### Titans (Behrouz, Zhong & Mirrokni, Google, NeurIPS 2025)

**"Titans: Learning to Memorize at Test Time"** — [arXiv:2501.00663](https://arxiv.org/pdf/2501.00663).

앞의 둘이 "밖에 쌓고 찾아온다"면, 이쪽은 **가중치에 새긴다**.

```
  신경 장기기억 모듈 (LMM)
        │
        ├─ ★순전파 도중 자기 가중치를 갱신★ 하여 기억
        ├─ 무엇을 기억할지 = 그래디언트 기반 ★놀람 지표 + 모멘텀★
        └─ 적응적 망각(adaptive forgetting)으로 오버플로 방지

  역할 분담이 명시적:
        attention   = 단기기억 (정확한 의존성 모델링)
        신경 메모리  = 장기기억 (지속적)
```

**놀람의 정의가 구체적이다** — 연상기억 손실에 대한 **입력에 대한 신경망의 그래디언트**로 측정한다. 기대를 위반한 사건이 더 기억에 남는다는 인간 장기기억의 성질에서 착안했다.

**해마 대응**: 외부 저장소가 아니라 **공고화(consolidation)** 에 가깝다. §2.3의 CLS에서 해마→신피질 이전에 해당한다.

#### Larimar (Das et al., IBM, ICML 2024)

**"Larimar: Large Language Models with Episodic Memory Control"** — [arXiv:2403.11901](https://arxiv.org/abs/2403.11901) · [코드](https://github.com/IBM/larimar).

분산 에피소드 기억을 붙인 뇌 착안 구조. 특징:

- **1회(one-shot) 지식 갱신** — 재학습·파인튜닝 불필요
- 순차 편집(sequential editing) 상황에서도 경쟁 베이스라인과 대등한 정확도
- 베이스 LLM에 따라 **4~10배 속도 향상**
- **선택적 사실 망각**과 입력 문맥 길이 일반화 기제 제공
- 구조가 단순하고 **LLM-agnostic**

**해마 대응**: "빠른 1회 학습"이라는 CLS의 해마 역할에 정확히 대응한다.

#### 계보: DNC (Graves et al., Nature 2016)

**"Hybrid computing using a neural network with dynamic external memory"** — *Nature* 538:471–476. ([링크](https://www.nature.com/articles/nature20101))

외부 메모리 계열의 원조. 신경망 컨트롤러가 **읽고 쓸 수 있는 외부 메모리 구조**를 갖고, 여러 주소 지정 방식(미사용 셀에 쓰기, 셀 갱신, 내용 기반 조회)으로 접근한다. 지금 계열 A~C가 하는 일의 원형이 여기 있다.

### 3.4 계열 D — 색인 그래프

#### HippoRAG (Gutiérrez et al., NeurIPS 2024)

**"HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models"** — [arXiv:2405.14831](https://arxiv.org/pdf/2405.14831).

**해마 색인 이론(§2.2)을 문자 그대로 구현한 시스템이다.**

```
  인공 신피질      =  LLM
  부해마영역(PHR)  =  인코더            ← 패턴 분리 담당
  인공 해마        =  오픈 지식그래프(KG) ← 색인
        │
        └─ 인출: ★Personalized PageRank★ 로 그래프 위에서 패턴 완성
```

논문이 명시하듯, 해마 색인 이론의 두 목표가 **패턴 분리**(서로 다른 경험의 표상을 고유하게 유지)와 **패턴 완성**(부분 자극에서 전체 기억 인출)이고, 시스템의 각 부분이 그것에 대응한다.

| 결과 | 수치 |
|---|---|
| 멀티홉 QA | SOTA 대비 **최대 20% 향상** |
| 단일 스텝 검색 vs 반복 검색 | 동등하거나 더 나은 성능에 **10~20배 저렴, 6~13배 빠름** |

#### HippoRAG 2 (2025)

**"From RAG to Memory: Non-Parametric Continual Learning for Large Language Models"** — [arXiv:2502.14802](https://arxiv.org/pdf/2502.14802).

1편의 한계를 명시한다: **엔티티 중심 접근이 색인·추론 양쪽에서 맥락 손실**을 일으키고 의미 매칭이 어려웠다. 2편은 오프라인 색인 / 온라인 인출의 2단계 구조를 유지하면서 이 부분을 개선한다.

이 블로그의 [LightRAG](../lightrag-review/), [CRAG](../crag-review/), [RAG 서베이 2026](../rag-survey-2026/)과 같은 계열이지만, **설계 동기가 신경과학에서 왔다는 점**이 다르다.

---

## 4. 대응표 — 해마 하위영역 ↔ 시스템

| 해마 요소 | 기능 | 대응 기제 | 대표 구현 |
|---|---|---|---|
| **EC** | 신피질 인터페이스, 구조 표상 | 위치·구조 인코딩 | Whittington 2022 (순환 위치 인코딩), TEM |
| **DG** | **패턴 분리** | **확장 투사 + 희소화** | ⚠️ **대부분 빠짐** (§5) |
| **CA3** | 패턴 완성 | **attention 자체** | Ramsauer 2020, Bricken 2021 |
| **CA1** | 비교·출력 게이팅 | 학습된 게이트 | Memorizing Transformers |
| **색인** | 신피질 패턴 포인터 | KG + PPR / 사건 색인 | HippoRAG, EM-LLM |
| **사건 분절** | 경험의 단위화 | 놀람 기반 경계 탐지 | EM-LLM, Titans |
| **시간 인접** | 자유회상 구조 | 시간 인접 검색 | EM-LLM |
| **공고화** | 해마 → 신피질 | 테스트시 가중치 갱신 / 1회 편집 | Titans, Larimar |
| **빠른 1회 학습** | 즉시 기억 | 에피소드 메모리 제어 | Larimar |

```
[ 조합 지도 ]

   입력
     │
     ▼
  ┌─────────────────────────────────────────────────┐
  │  트랜스포머 (신피질 + CA3)                        │
  │   · 순환 위치 인코딩 → EC 구조 표상    (Whittington)│
  │   · attention        → CA3 패턴 완성   (Ramsauer)  │
  └───────────┬─────────────────────────────────────┘
              │ 특정 층의 의미 표상
              ▼
      ┌───────────────┐
      │ 놀람 게이팅     │  기억할 가치가 있는가?      (EM-LLM/Titans)
      └───────┬───────┘
              │ yes
              ▼
      ┌───────────────┐
      │ ★ 패턴 분리 ★  │  DG — 확장 + 희소화        (§5)
      └───────┬───────┘
              ▼
      ┌───────────────┐
      │ 색인 저장       │  포인터 + 분리된 키         (색인 이론)
      │                │  내용 복제 ✗
      └───────┬───────┘
              │
              ▼  인출 시
      ┌───────────────┐
      │ 2단계 검색     │  유사도 + ★시간 인접★       (EM-LLM)
      └───────┬───────┘
              ▼
      ┌───────────────┐
      │ 게이트 결합     │  CA1                       (Memorizing T.)
      └───────┬───────┘
              ▼
      ┌───────────────┐
      │ 공고화          │  주기적 재생 → 가중치        (Titans/Larimar)
      └───────────────┘
```

---

## 5. 대부분이 빠뜨리는 것 — 패턴 분리 (DG)

### 5.1 왜 필요한가

§4 표에서 유일하게 "대부분 빠짐"으로 표시된 항목이다. 그런데 **해마에서 DG는 정보가 CA3에 도달하기 전 반드시 거치는 관문**이다.

```
  DG 없이 저장하면:

    비슷한 경험 A, A'  →  표상이 거의 겹침
                        →  검색 시 서로를 끌어당김
                        →  ★ 간섭(interference) ★
                        →  A를 찾으려 했는데 A'가 나오거나, 둘의 평균이 나온다

  §1.1의 Hopfield 용어로:
    겹치는 패턴들은 ★준안정 상태(부분집합 평균)★ 에 빠진다
    단일 패턴 고정점으로 수렴하지 못한다
```

DG의 해부학적 특징이 정확히 이 문제를 겨냥한다.

| DG 특징 | 계산적 역할 |
|---|---|
| EC보다 뉴런 수가 훨씬 많다 | **확장 투사(expansion)** — 고차원으로 올려 분리 여지를 만든다 |
| 활성률이 매우 낮다 | **희소 코딩(sparsity)** — 겹침 확률을 낮춘다 |
| 결과 | 유사 입력의 발화 패턴 **직교화(orthogonalization)** |

HippoRAG가 PHR 인코더에 패턴 분리 역할을 명시적으로 배정한 것도 같은 이유다.

### 5.2 최소 구현

학습 없이도 시작할 수 있다.

```python
import torch, torch.nn.functional as F

class PatternSeparator(torch.nn.Module):
    """DG 모사: 확장 투사 + k-WTA 희소화 → 색인 키 생성.

    W는 랜덤 고정(Johnson-Lindenstrauss)만으로도 동작하고,
    학습시키면 더 좋아진다. 핵심은 '확장 + 희소화' 두 단계다.
    """
    def __init__(self, d_in, d_expand, k, learnable=False):
        super().__init__()
        W = torch.randn(d_in, d_expand) / d_in ** 0.5
        self.W = torch.nn.Parameter(W, requires_grad=learnable)
        self.k = k                      # 활성 뉴런 수. d_expand의 1~5% 권장

    def forward(self, h):               # h: (..., d_in) 의미 표상
        z = h @ self.W                  # ① 확장:  d_in -> d_expand (>> d_in)
        val, idx = z.topk(self.k, dim=-1)
        s = torch.zeros_like(z).scatter_(-1, idx, val)   # ② k-WTA 희소화
        return F.normalize(s, dim=-1)   # 이것을 '색인 키'로 저장한다
```

**분리가 실제로 되는지 확인하는 법** — 저장 전후의 코사인 유사도 분포를 비교한다.

```python
def separation_gain(h, sep):
    """유사 표상 쌍의 유사도가 실제로 낮아졌는지 측정."""
    a = F.normalize(h, dim=-1) @ F.normalize(h, dim=-1).T      # 원 표상
    b = sep(h) @ sep(h).T                                      # 분리 후
    off = ~torch.eye(len(h), dtype=torch.bool, device=h.device)
    return a[off].mean().item(), b[off].mean().item()          # (전, 후)
```

두 번째 값이 첫 번째보다 뚜렷하게 작아야 한다. 안 그러면 `d_expand`를 키우거나 `k`를 줄인다.

**실측** — `d_in=512`, `d_expand=8192`, 공통 성분을 공유하는 매우 유사한 표상 32개(평균 코사인 유사도 0.89)에 적용한 결과다. 랜덤 고정 `W`, 학습 없음.

| `k` (활성 뉴런) | 희소율 | 분리 전 | 분리 후 | 감소폭 |
|---:|---:|---:|---:|---:|
| 16 | 0.20% | 0.8935 | **0.5055** | −0.3880 |
| 64 | 0.78% | 0.8935 | 0.5569 | −0.3366 |
| 256 | 3.13% | 0.8935 | 0.6645 | −0.2290 |

학습된 가중치 없이 **랜덤 투사 + k-WTA만으로** 유사도가 0.89 → 0.51까지 떨어진다. 그리고 `k`가 작을수록 분리가 강해지는데, 이것이 §5.3의 트레이드오프를 그대로 보여준다.

### 5.3 주의 — 분리와 완성의 트레이드오프

과하게 분리하면 **패턴 완성이 안 된다.** 부분 단서로 원본을 찾아야 하는데, 코드가 너무 희소하면 단서와 저장된 키가 겹치지 않는다.

```
   k 작음 (희소↑)  →  분리 좋음, 완성 어려움   (단서가 빗나감)
   k 큼   (희소↓)  →  완성 쉬움, 간섭 발생     (DG가 없는 것과 비슷)
```

실무적으로는 **저장 키는 희소하게, 질의는 덜 희소하게** (질의 시 `k`를 2~3배로) 두면 완충이 된다. 뇌에서도 DG→CA3 mossy fiber는 희소하고 강한 반면, EC→CA3 직접 경로는 약하고 넓게 퍼져 있어 유사한 비대칭을 보인다.

---

## 6. 설계 결정 5가지

### 6.1 무엇을 저장할 것인가

| 선택지 | 대표 | 특징 |
|---|---|---|
| 특정 층의 (K, V) | Memorizing Transformers | 가장 단순. attention에 바로 꽂힘 |
| 사건 청크 + 경계 | EM-LLM | 인지적으로 타당. 시간 구조 보존 |
| 엔티티·명제 그래프 | HippoRAG | 멀티홉에 강함. 구축 비용 있음 |
| 가중치 델타 | Titans, Larimar | 공고화. 검색 불필요 |

**색인 이론(§2.2)을 따른다면**: 원문 span 오프셋 + 분리된 키만 저장하고, 회상 시 원문을 재활성화한다. 의미 표상을 통째로 복제하지 않는다.

### 6.2 어느 층에서 뽑을 것인가

Memorizing Transformers는 **상위 쪽 한 층**만 쓴다. 그리고 §1.1의 Ramsauer 결과가 근거를 준다 — 하위 층은 **전역 평균 영역**(정보가 뭉개짐), 상위 층은 **준안정 상태**(부분집합 검색)에서 동작한다.

최종층은 다음 토큰 예측에 특화되어 의미보다 표층에 가까우므로, **중상위 중간층**이 대체로 안전하다.

### 6.3 언제 쓸 것인가 — 놀람 게이팅

전부 저장하면 (a) 메모리가 터지고 (b) 검색이 흐려진다.

**놀람(surprise) 기반 게이팅**이 EM-LLM과 Titans가 독립적으로 도달한 답이고, §2.3의 CLS가 신경과학적 근거를 준다(재생은 참신성에 의해 조절된다).

| 구현 | 놀람의 정의 |
|---|---|
| EM-LLM | **베이지안 놀람** — 사건 경계 탐지에 사용 |
| Titans | **연상기억 손실의 입력에 대한 그래디언트** + 모멘텀 |

### 6.4 어떻게 인출할 것인가

유사도만으로는 부족하다. EM-LLM의 2단계 — **유사도 + 시간적 인접성** — 를 권한다. 인간 자유회상의 시간 인접 효과를 재현하는 최소 장치다.

### 6.5 무엇을 잊을 것인가

명시적 삭제 정책이 없으면 오래된 기억이 검색을 오염시킨다. Titans의 **적응적 망각**, Larimar의 **선택적 사실 망각**이 참고 사례다.

---

## 7. 무엇부터 해볼 것인가

비용 순으로 정렬했다.

```
① 순환 위치 인코딩으로 바꿔본다                        [가장 저렴]
   근거: Whittington 2022 — 이것만으로 장소·격자세포 재현
   외부 메모리를 붙이기 전에 확인할 것

② EM-LLM을 그대로 붙여본다                            [파인튜닝 불필요]
   놀람 분절 + 시간 인접 검색이 이미 구현되어 있다
   해마 기능의 상당 부분을 즉시 확보

③ 저장 경로에 패턴 분리 층을 끼운다                     [§5]
   대부분의 시스템이 빠뜨린 부분
   separation_gain으로 실제 분리를 확인할 것

④ 색인 구조로 전환한다                                 [설계 변경]
   내용 복제 → 포인터 + 결합
   HippoRAG 2가 참고 구현

⑤ 공고화를 붙인다                                     [가장 비쌈]
   Titans 방식(테스트시 가중치 갱신) 또는
   Larimar 방식(1회 편집)
```

**①~③까지가 비용 대비 효과가 가장 크다.** ①은 인코딩만 바꾸는 것이고, ②는 학습이 없으며, ③은 층 하나다.

---

## 8. 정리

```
  "attention할 때 의미 표상을 따로 저장한다"
              │
              ├─ 그런데 attention은 이미 CA3 패턴 완성을 한다  (§1)
              │     Ramsauer(Hopfield 동치) · Bricken(SDM) · Whittington(장소세포)
              │
              ▼
  따라서 따로 만들 것은:
              │
              ├─ ★ 패턴 분리 (DG) ★   ← 대부분 빠뜨림. 없으면 간섭   (§5)
              ├─ 색인 (내용 복제 ✗, 포인터 ○)                      (§2.2)
              ├─ 놀람 기반 저장 게이팅                              (§6.3)
              ├─ 시간 인접 인출                                    (§6.4)
              └─ 공고화 (해마 → 신피질)                            (§2.3)
```

---

## 참고문헌

### attention = 연상기억
- Ramsauer, H. et al. (2020). Hopfield Networks is All You Need. [arXiv:2008.02217](https://arxiv.org/abs/2008.02217)
- Bricken, T., Pehlevan, C. (2021). Attention Approximates Sparse Distributed Memory. NeurIPS 2021. [arXiv:2111.05498](https://arxiv.org/abs/2111.05498) · [코드](https://github.com/TrentBrick/attention-approximates-sdm)
- Whittington, J.C.R., Warren, J., Behrens, T.E.J. (2022). Relating transformers to models and neural representations of the hippocampal formation. ICLR 2022. [arXiv:2112.04035](https://arxiv.org/abs/2112.04035)

### 해마 이론
- Teyler, T.J., Rudy, J.W. (2007). The hippocampal indexing theory and episodic memory: Updating the index. *Hippocampus*. [링크](https://onlinelibrary.wiley.com/doi/abs/10.1002/hipo.20350)
- Kumaran, D., Hassabis, D., McClelland, J.L. (2016). What Learning Systems do Intelligent Agents Need? Complementary Learning Systems Theory Updated. *TiCS*. [PubMed](https://pubmed.ncbi.nlm.nih.gov/27315762/)
- Whittington, J.C.R. et al. (2020). The Tolman-Eichenbaum Machine. *Cell*. [링크](https://www.cell.com/cell/fulltext/S0092-8674(20)31388-X)

### 외부 메모리 / 검색 증강
- Graves, A. et al. (2016). Hybrid computing using a neural network with dynamic external memory. *Nature* 538:471–476. [링크](https://www.nature.com/articles/nature20101)
- Khandelwal, U. et al. (2020). Generalization through Memorization: Nearest Neighbor Language Models. ICLR 2020. [arXiv:1911.00172](https://arxiv.org/abs/1911.00172) · [코드](https://github.com/urvashik/knnlm)
- Wu, Y., Rabe, M.N., Hutchins, D., Szegedy, C. (2022). Memorizing Transformers. ICLR 2022. [arXiv:2203.08913](https://arxiv.org/pdf/2203.08913) · [구현](https://github.com/lucidrains/memorizing-transformers-pytorch)

### 에피소드 기억 / 공고화
- Fountas, Z. et al. (2025). Human-inspired Episodic Memory for Infinite Context LLMs. ICLR 2025. [arXiv:2407.09450](https://arxiv.org/abs/2407.09450) · [코드](https://github.com/em-llm/EM-LLM-model)
- Behrouz, A., Zhong, P., Mirrokni, V. (2025). Titans: Learning to Memorize at Test Time. NeurIPS 2025. [arXiv:2501.00663](https://arxiv.org/pdf/2501.00663)
- Das, P. et al. (2024). Larimar: Large Language Models with Episodic Memory Control. ICML 2024. [arXiv:2403.11901](https://arxiv.org/abs/2403.11901) · [코드](https://github.com/IBM/larimar)

### 해마 착안 검색
- Gutiérrez, B.J. et al. (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. NeurIPS 2024. [arXiv:2405.14831](https://arxiv.org/pdf/2405.14831)
- HippoRAG 2 (2025). From RAG to Memory: Non-Parametric Continual Learning for LLMs. [arXiv:2502.14802](https://arxiv.org/pdf/2502.14802)

---

## 관련 블로그 포스트

- [Bayesian Surprise](../bayesian-surprise/) — §3.2·§6.3의 놀람 기반 게이팅
- [AI와 실제 지능 서베이](../ai-and-natural-intelligence-survey/) — 표상 정합 축과 이어짐
- [LightRAG](../lightrag-review/) · [CRAG](../crag-review/) · [RAG 서베이 2026](../rag-survey-2026/) — 계열 D의 RAG 쪽 계보
- [Prospective Configuration](../prospective-configuration-review/) · [Meta-PCN](../meta-pcn-review/) — 예측 부호화와 뇌의 학습 규칙
- [HiPPO](../hippo-review/) — 이름만 비슷한 상태공간 압축 이론. 장기 의존성의 **다른** 접근
