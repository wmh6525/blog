---
title: "[서베이] 인공지능과 실제 지능 — '기계와 뇌는 같은 것을 하는가'를 다룬 연구 40편"
date: 2026-09-07
tags: ["서베이", "NeuroAI", "인지과학", "LLM", "뇌", "표상정합", "ARC-AGI", "논문리뷰"]
categories: ["ML/AI"]
summary: "AI와 생물학적 지능의 관계를 다룬 연구를 8개 축으로 정리한다. 시각·언어 표상 정합(Yamins 2014, Schrimpf 2021, Goldstein 2022, Tuckute 2024의 인과적 개입), 이해 논쟁(Bender & Koller, Mitchell & Krakauer, Vafa의 불완전 세계모델), 기계심리학(Binz & Schulz, Webb vs Hodel, Strachan ToM, Centaur), 측정 논쟁(ARC-AGI-3에서 인간 100% vs AI 0.51%), 데이터 효율 격차(4~5자릿수), 학습 규칙, 수렴 가설. 세 편에서 독립적으로 나타난 '섭동 비대칭'과 '예측≠설명' 단층선을 종합한다."
math: true
toc: true
draft: false
---

## 0. 이 글의 범위

"AI가 진짜 지능인가"는 답이 없는 질문이다. 하지만 **"인공 시스템과 생물학적 시스템이 같은 계산을 하는가"** 는 답이 있는 질문이고, 지난 10여 년간 실제로 측정되어 왔다.

이 글은 그 측정을 8개 축으로 정리한다.

| 축 | 질문 | 대표 연구 |
|---|---|---|
| 1 | 왜 이것이 연구 프로그램이 되었나 | Zador et al. 2023 (NeuroAI) |
| 2 | 시각 표상이 일치하는가 | Yamins & DiCarlo 2014 ↔ Bowers 2023, Feather 2023 |
| 3 | 언어 표상이 일치하는가 | Schrimpf 2021, Goldstein 2022, **Tuckute 2024 (인과적 개입)** |
| 4 | LLM은 이해하는가 | Bender & Koller 2020, Mitchell & Krakauer 2023, Vafa 2024 |
| 5 | 인지심리 검사를 통과하는가 | Binz & Schulz 2023, Webb 2023 ↔ Hodel & West 2023, Strachan 2024 |
| 6 | 무엇을 지능의 증거로 인정할 것인가 | Chollet 2019, ARC-AGI-2/3, Schaeffer 2023 |
| 7 | 데이터 효율 격차 | Frank 2023, BabyLM |
| 8 | 학습 규칙이 같은가 / 표상이 수렴하는가 | 비역전파 계열, Huh 2024 |

마지막 §10~§11에서 **여러 축을 가로지르는 두 가지 패턴**을 종합한다. 이 글에서 새로 주장하는 부분은 그 두 절이며, 나머지는 원 논문의 내용을 정리한 것이다.

> **인용 원칙**: 수치와 주장은 원 논문·공식 페이지에서 확인한 것만 적었다. 확인하지 못한 것은 그렇다고 표시했다.

---

## 1. 왜 이것이 연구 프로그램이 되었나 — NeuroAI

### 1.1 Zador et al. 2023

**"Catalyzing next-generation Artificial Intelligence through NeuroAI"** — *Nature Communications* 14:1597 (2023-03-22). Zador, Escola, Richards, Ölveczky, **Bengio, LeCun**, DiCarlo, Ganguli, Hawkins, Körding, Lillicrap, Olshausen, Sejnowski, Simoncelli, Tsao 등 **27인 공저**.

이 논문은 논증이라기보다 **분야 선언문**에 가깝다. 핵심 제안은 **체화된 튜링 테스트(embodied Turing test)** 다.

```
[ 기존 튜링 테스트 ]
  대화로 인간과 구별되지 않는가
        │
        └─ 인간 고유 능력에 초점

[ 체화된 튜링 테스트 (제안) ]
  동물 수준의 감각운동 상호작용을 할 수 있는가
        │
        └─ 5억 년간 진화한, 모든 동물이 공유하는 능력에 초점
```

**논지의 전환점**: "인간만 하는 것"이 아니라 **"다람쥐도 하는 것"** 을 목표로 삼자는 것이다. 언어는 최근에 생겼고, 나뭇가지 사이를 뛰어다니는 능력은 5억 년의 산물이다. 후자가 훨씬 어렵고, 그것을 못 하는 한 지능을 이해했다고 말할 수 없다는 입장이다.

**이 관점이 §7(데이터 효율)과 §6(ARC-AGI-3)에서 실증적으로 되돌아온다.** 상호작용·탐색 능력이 정확히 현재 시스템이 가장 못하는 부분이기 때문이다.

---

## 2. 축 2: 시각 — 표상 정합의 원형과 그 반론

### 2.1 출발점: Yamins & DiCarlo 2014

**"Performance-optimized hierarchical models predict neural responses in higher visual cortex"** — *PNAS* 111(23):8619–8624.

이 논문이 전체 분야의 방법론적 원형이다.

```
① 물체 분류 과제에서 인간 수준 성능을 내는 심층망을 찾는다
        │
        └─ ★ 신경 데이터에 맞추도록 훈련하지 않았다 ★
        │
② 그 망의 중간층 활성으로 V4·IT 뉴런 반응을 예측한다
        │
        └─ 잘 예측된다
```

**핵심 논리**: 신경 데이터를 목표로 훈련하지 않았는데도 신경 반응을 예측한다면, 그 예측력은 **과제 최적화 자체에서 나온 것**이다. 즉 "뇌도 같은 과제를 풀도록 최적화되었기 때문에 비슷한 표상에 도달했다"는 해석이 가능해진다.

이 논리 구조 — **과제 성능 최적화 → 부수적 신경 예측력** — 가 이후 §3의 언어 연구로 그대로 이식된다.

### 2.2 반론 (1): 예측력이 곧 기제는 아니다

**Bowers et al. 2023, "Deep problems with neural network models of human vision"** — *Behavioral and Brain Sciences* 46 (13인 공저, 타깃 논문 + 다수 논평).

DNN이 최고의 생물학적 시각 모델이라는 결론은 대개 세 가지 근거에 의존한다.

1. 이미지 분류 정확도가 가장 높다
2. 인간의 분류 **오류 패턴**을 가장 잘 예측한다
3. 이미지에 대한 **뇌 신호**를 가장 잘 예측한다

Bowers 등의 반론:

> **행동·뇌 데이터셋은 "어떤 특징이 좋은 예측에 기여하는지"에 대한 가설을 검정하지 않는다.** 그리고 DNN은 심리학 연구의 결과를 거의 설명하지 못한다.

**핵심 지적**: 예측 성능 벤치마크는 **모델 순위**를 매길 뿐, **왜 그 모델이 이기는지**를 묻지 않는다. 심리물리학이 수십 년간 축적한 현상들(형태 우선성, 부분-전체 관계, 배열 효과 등)은 DNN이 재현하지 못한다.

### 2.3 반론 (2): 메타머 — 불변성이 다르다

**Feather et al. 2023, "Model metamers reveal divergent invariances between biological and artificial neural networks"** — *Nature Neuroscience*.

가장 날카로운 실험 설계다.

```
[ 메타머 (metamer) ]
  모델의 특정 층에서 활성이 자연 자극과 ★동일하게 매칭된★ 자극

  모델 입장:  자연 이미지 == 메타머   (구별 불가)
  인간 입장:  자연 이미지 != 메타머   (후기 층 메타머는 대개 ★알아볼 수 없음★)
```

시각·청각 양쪽에서, 지도학습·비지도학습 SOTA 모델의 **후기 층에서 생성한 메타머는 인간에게 완전히 인식 불가능한 경우가 많았다.**

**왜 중요한가**: 두 시스템이 같은 표상을 갖는다면 **같은 것을 구별하지 못해야 한다.** 불변성(invariance)이 다르다는 것은 표상이 다르다는 직접 증거다. 그리고 논문이 명시하듯 **메타머 인식률은 기존 뇌 기반 벤치마크 점수와도, 적대적 취약성과도 분리(dissociate)된다.** 즉 Brain-Score가 높아도 메타머는 여전히 인식 불가일 수 있다 — **기존 벤치마크가 못 잡는 독립적 실패 양상**이다.

### 2.4 이 축의 정리

| | 주장 | 증거 유형 |
|---|---|---|
| 긍정 | 과제 최적화 망이 시각 피질 반응을 예측한다 | **예측 정확도** |
| 부정 1 | 예측이 좋아도 특징 가설은 검정되지 않았다 | **설명력 부재** |
| 부정 2 | 불변성 구조가 인간과 다르다 | **반증적 자극(메타머)** |

이 3층 구조 — 예측 성능 / 설명력 / 반증 설계 — 가 §3~§5의 모든 논쟁에서 반복된다.

---

## 3. 축 3: 언어 — LLM과 뇌

### 3.1 Schrimpf et al. 2021 — 다음 단어 예측이 뇌 적합도를 결정한다

**"The neural architecture of language: Integrative modeling converges on predictive processing"** — *PNAS* 118(45):e2105646118. Schrimpf, Blank, Tuckute, Kauf, Hosseini, **Kanwisher, Tenenbaum, Fedorenko**.

주요 결과:

| 발견 | 내용 |
|---|---|
| 예측력 | 가장 강력한 트랜스포머가 문장에 대한 신경 반응의 **설명 가능 분산의 거의 100%** 를 예측 |
| 일반화 | 서로 다른 데이터셋과 **영상 양식(fMRI, ECoG)** 을 가로질러 일반화 |
| 결정 요인 ★ | brain score와 행동 적합도가 **다음 단어 예측 성능과 강하게 상관** — 그런데 **다른 언어 과제 성능과는 아니다** |
| 구조 | 트랜스포머 > 순환망 > 단어 임베딩. 대용량 > 소용량 |

**세 번째 행이 논문의 제목("predictive processing로 수렴")을 낳았다.** 뇌 적합도가 다음 단어 예측 성능과만 상관하고 다른 과제와는 상관하지 않는다면, 뇌의 언어 영역이 하는 일도 **예측적 처리**일 가능성이 커진다.

### 3.2 Goldstein et al. 2022 — 세 가지 공유 계산 원리

**"Shared computational principles for language processing in humans and deep language models"** — *Nature Neuroscience* 25(3).

**설계**: 참가자 9명이 30분 팟캐스트를 듣는 동안 **ECoG**(피질전도) 기록.

자기회귀 언어모델과 인간 뇌가 공유하는 세 원리:

```
① 단어가 나오기 ★전에★ 연속적으로 다음 단어를 예측한다
        │
② 예측을 실제 들어온 단어와 대조해 ★사후 놀람(surprise)★ 을 계산한다
        │
③ 자연 맥락 속 단어를 ★맥락적 임베딩★ 으로 표상한다
```

**②가 특히 중요하다.** 놀람 신호는 예측 부호화(predictive coding) 계열 이론의 핵심 양이다. 이 블로그의 [Bayesian Surprise](../bayesian-surprise/), [Prospective Configuration](../prospective-configuration-review/), [Meta-PCN](../meta-pcn-review/) 리뷰와 직접 이어지는 지점이다.

### 3.3 ★ Tuckute et al. 2024 — 상관에서 개입으로

**"Driving and suppressing the human language network using large language models"** — *Nature Human Behaviour*.

**이 축에서 방법론적으로 가장 중요한 논문이다.** §3.1~§3.2는 전부 **상관 연구**다. 이 논문은 **개입**을 한다.

```
① fMRI로 1,000개 다양한 문장에 대한 뇌 반응 측정
        │
② GPT 기반 인코딩 모델이 각 문장의 반응 크기를 예측하도록 학습
        │
③ ★ 모델을 이용해 "반응을 최대로 끌어올릴 문장"과
     "억제할 문장"을 새로 설계 ★
        │
④ ★ 새로운 피험자 ★ 에게 제시 → 실제로 언어 영역 활동이 강하게
     구동되고 억제됨을 확인
```

**왜 결정적인가**: 예측은 우연히 맞을 수 있지만, **설계한 개입이 새로운 사람에게서 의도대로 작동하는 것**은 훨씬 강한 증거다. 모델이 언어 영역의 반응 함수를 실제로 포착했다는 뜻이다.

부수 발견: 반응 강도의 핵심 결정 요인은 입력의 **놀람도(surprisal)와 적형성(well-formedness)** 이었다. §3.1의 예측적 처리 결론과 일치한다.

논문의 표현대로, 신경망 모델이 인간 언어를 **모방**하는 데 그치지 않고 고차 피질 영역의 활동을 **비침습적으로 제어**할 수 있음을 보였다.

### 3.4 정교화: 다음 단어 예측만으로는 부족하다

**Antonello & Huth 계열, "Language models and brains align due to more than next-word prediction and word-level information"** — EMNLP 2024 (arXiv:2212.00596).

§3.1의 결론을 반박하는 게 아니라 **좁힌다**.

> 다음 단어 예측이 정합에 **필요한가, 아니면 단지 충분한가?**

단어 수준 정보와 예측 정확도를 **통제한 뒤에도** 언어 영역에서 **잔여 정합(residual alignment)** 이 남는다. 즉 모델은 다음 단어 예측으로 환원되지 않는 다른 속성도 포착하고 있다.

관련해서 최근 연구들은 **선형 매핑의 한계**도 지적한다. 전통적 인코딩 모델은 단일 양식 특징에서 선형 사상을 쓰는데, 비선형·다중양식 접근이 예측 성능을 유의하게 개선한다는 결과가 있다.

**함의**: "brain score가 높다"는 진술은 **어떤 매핑 함수를 허용했는지에 의존한다.** 매핑이 유연할수록 점수는 오르지만 주장은 약해진다. §10에서 다시 다룬다.

### 3.5 이 축의 종합 리뷰

**Tuckute, Kanwisher, Fedorenko 2024, "Language in Brains, Minds, and Machines"** — *Annual Review of Neuroscience* 47:277–301.

이 축을 처음 읽는다면 여기서 시작하는 것이 좋다. 언어모델이 뇌 인코딩·디코딩을 가능하게 할 만큼 유사하게 언어 정보를 표상한다는 증거를 정리하고, **모델의 어떤 속성**(구조 / 과제 성능 / 학습)이 신경 반응 포착에 결정적인지를 다룬다.

### 3.6 결정적 구분: 언어 ≠ 사고

**Fedorenko, Piantadosi & Gibson 2024, "Language is primarily a tool for communication rather than thought"** — *Nature* (2024-06).

이 논문은 LLM 논쟁의 프레임 자체를 바꾼다.

> 신경과학 증거에 따르면 현생 인류에게 **언어는 소통의 도구**이며, 사고를 위해 언어를 쓴다는 유력한 견해에 반한다. 언어는 인간 인지의 정교함을 **낳는 것이 아니라 반영할 뿐**이다.

```
[ 함의 ]

  LLM이 인간 언어 네트워크를 잘 모델링한다  ← §3.1~§3.3에서 실제로 그렇다
              │
              │  그런데 언어 네트워크는 사고를 하는 곳이 아니다
              ▼
  LLM이 인간의 ★사고★ 를 모델링한다는 결론은 따라 나오지 않는다
```

**이것이 §3 전체의 가장 중요한 단서(caveat)다.** 언어 정합의 성공은 인상적이지만, 그 성공이 보장하는 범위는 정확히 "언어 처리"까지다. §4~§5의 이해·추론 논쟁이 별개 축으로 존재해야 하는 이유가 여기 있다.

---

## 4. 축 4: 이해 논쟁

### 4.1 Bender & Koller 2020 — 문어(octopus) 논증

**"Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data"** — ACL 2020.

사고실험: 초지능 문어가 해저 통신 케이블만 도청해 두 사람의 대화 패턴을 완벽히 학습한다. 그 문어는 대화를 이어갈 수 있지만, 한 사람이 곰의 공격을 받아 도움을 구할 때 **무엇을 해야 할지 알 수 없다.**

> **형식(form)만으로 훈련된 시스템은 원리상 의미(meaning)를 학습할 방법이 없다.**

**이 논증의 성격**: 경험적 주장이 아니라 **원리적(a priori) 주장**이다. 따라서 모델 성능 향상으로 반박되지 않으며, 반박하려면 "형식만으로도 지시(reference)가 성립할 수 있다"를 보여야 한다 — 그것이 §4.3이다.

### 4.2 Mitchell & Krakauer 2023 — 논쟁의 지도

**"The debate over understanding in AI's large language models"** — *PNAS* 120(13).

찬반 양측 논증을 정리한 뒤, 저자들의 입장:

> **이해란 개념에 대한 인과적 지식**이다. 개념은 외부 세계와 '자기'에 대한 내적 모델이며, 그들 사이의 위계적 관계다.

그리고 결론은 판정이 아니라 **연구 프로그램의 제안**이다:

> 확장된 **지능의 과학(science of intelligence)** 을 발전시켜, **서로 다른 이해의 양식들**과 그 강점·한계, 그리고 이질적 인지 형태들을 통합하는 문제를 조명해야 한다.

**핵심 이동**: "LLM이 이해하는가(예/아니오)"에서 **"이해에는 몇 가지 종류가 있고 LLM은 어느 것을 갖는가"** 로 질문을 바꾼다. 이 재구성이 §10의 종합에서 실질적으로 유용하다.

### 4.3 Mollo & Millière 2023 — 벡터 접지 문제

**"The Vector Grounding Problem"** — arXiv:2304.01481 / *Philosophy and the Mind Sciences*.

고전 기호 접지 문제(Harnad)의 현대판. 저자들의 기여는 **접지를 종류별로 분해한 것**이다.

| 접지 유형 | 필수인가 |
|---|---|
| **지시적 접지(referential grounding)** — 표상과 세계 내 지시체의 연결 | **이것만 본질적** |
| 감각운동 접지 | 아니다 |
| 사회적 접지 | 아니다 |
| 의사소통적 의도 접지 | 아니다 |

그리고 목적의미론(teleosemantics)에서 두 조건을 도출한다: 내적 상태가 (1) 세계와 적절한 **인과-정보적 관계**에 있고, (2) 그 정보를 담는 **기능을 부여한 선택의 역사**를 가질 것.

**결론이 흥미롭다.** 저자들은 LLM이 지시적 접지를 **달성할 수 있다**고 주장한다 — (a) 세계 관여적 기능을 명시적으로 확립하는 **선호 미세조정(RLHF 등)** 을 통해, 그리고 (b) 제한된 영역에서는 **사전학습만으로도**.

즉 §4.1의 원리적 부정에 대한 **원리적 반론**이다. "형식만"이라는 전제가 RLHF 이후에는 더 이상 참이 아니라는 것.

### 4.4 Vafa et al. 2024 — 세계 모델을 실제로 측정하다

**"Evaluating the World Model Implicit in a Generative Model"** — NeurIPS 2024 (arXiv:2406.03689).

철학 논쟁을 **측정 가능한 문제로 바꾼** 논문이다. 기저 현실이 **결정적 유한 오토마타(DFA)** 로 기술되는 경우로 한정하고, 언어이론의 **Myhill–Nerode 정리**에서 세계모델 복원 지표를 유도한다.

가장 인상적인 실험:

```
[ 뉴욕시 택시 데이터 ]

  훈련: 실제 택시 운행을 턴바이턴 방향 지시 시퀀스로 변환
        트랜스포머가 다음 방향을 예측하도록 학습

  성능: 두 교차로 사이의 ★유효한 경로★ 를 찾을 뿐 아니라
        대개 ★최단 경로★ 를 찾아낸다               ← 놀랍도록 잘한다

  진단: 그런데 모델의 암묵적 뉴욕 거리 지도를 복원해 보면
        ★실제 지도와 거의 닮지 않았다★              ← 세계모델은 비정합적
```

**이것이 이 논쟁 전체에서 가장 깔끔한 결과다.**

> **과제 성능이 높다는 것과 정합적인 세계 모델을 갖는다는 것은 별개다.**

성능만 보면 "이 모델은 뉴욕 지리를 안다"고 말하고 싶어진다. 하지만 지도를 꺼내 보면 존재하지 않는 도로가 잔뜩 있다. §2.3의 메타머와 정확히 같은 종류의 증거다 — **성능 벤치마크가 못 잡는 것을 잡는 반증적 진단 설계.**

---

## 5. 축 5: 기계심리학 — 인지 검사를 돌려보다

### 5.1 Binz & Schulz 2023 — 인지심리학 도구로 GPT-3 검사

**"Using cognitive psychology to understand GPT-3"** — *PNAS* 120(6).

심리학 문헌의 정전(canonical) 실험 배터리를 그대로 적용: 의사결정, 정보 탐색, 숙고, 인과 추론.

| 결과 | 내용 |
|---|---|
| ✅ 인상적 | 삽화(vignette) 기반 과제를 인간과 비슷하거나 더 잘 푼다 |
| ✅ 인상적 | **다중 슬롯머신 과제에서 인간을 능가**. 모델 기반 강화학습의 흔적을 보인다 |
| ❌ 실패 | **삽화에 작은 섭동을 주면 크게 빗나간다** |
| ❌ 실패 | 지향적 탐색(directed exploration)의 흔적이 없다 |
| ❌ 실패 | 인과 추론 과제에서 **처참하게 실패** |

**세 번째 행을 기억해 두자.** §5.2, §5.3에서 독립적으로 다시 나타난다.

### 5.2 유추 추론 — 3라운드 논쟁

이 논쟁은 **좋은 과학적 교환의 사례**다.

**1라운드 — Webb, Holyoak & Lu 2023**, "Emergent analogical reasoning in large language models", *Nature Human Behaviour*.
GPT-3가 대부분의 설정에서 인간과 대등하거나 능가하는 **추상적 패턴 귀납** 능력을 보인다. 제로샷 유추 문제를 광범위하게 푼다.

**2라운드 — Hodel & West 2023**, "Response: Emergent analogical reasoning in large language models" (arXiv:2308.16118).
문자열 유추의 **가장 단순한 변형**에 대한 반례를 제시. GPT-3는 변형된 문제를 못 푸는 반면 **인간 성능은 모든 변형에서 일관되게 유지된다.** 제로샷 추론은 특별한 주장이므로 특별한 증거가 필요하며, **암기 배제 설계**가 필요하다는 지적.

**3라운드 — Lewis & Mitchell 2024 / Webb et al. 2024**.
Lewis & Mitchell은 라틴 알파벳 영역에서 GPT 계열이 문자열 유추를 약 **60% 정확도**로 풀며, 이는 그들이 검사한 성인보다 다소 낮다고 보고했다. 그리고 **치환된 알파벳(counterfactual task)** 에서는 성능이 더 나빠진다. Webb 등도 후속 논문에서 치환 알파벳 + 큰 간격 조건에서 GPT-3·GPT-4의 성능 저하를 확인했다.

```
[ 논쟁의 수렴 ]

  원 과제                 → 인간 대등 이상
  사소한 변형             → 성능 저하
  반사실적 변형(치환 알파벳) → 뚜렷한 저하    ← ★ 양측이 동의 ★

  인간                    → 모든 변형에서 일관
```

**의미**: 논쟁이 "된다/안 된다"에서 **"어떤 변형 축에서 얼마나 무너지는가"** 로 정밀해졌다. 이것이 진전이다.

### 5.3 마음이론 (Theory of Mind)

**Strachan et al. 2024**, "Testing theory of mind in large language models and humans", *Nature Human Behaviour*.

**설계 규모**: GPT 계열과 LLaMA2 계열을, **인간 참가자 1,907명** 표본과 비교. 오신념, 간접 요청 해석, 아이러니, **실언(faux pas)** 인식 등 배터리.

| 모델 | 결과 |
|---|---|
| GPT-4 | 간접 요청·오신념·오도(misdirection)에서 **인간 수준 또는 그 이상**. 단 **faux pas만 실패** |
| GPT-3.5 | 동일하게 faux pas만 실패 |
| LLaMA2-70B | 전반적으로 최하위인데 **faux pas만 인간을 능가** |

**후속 조작이 이 논문의 백미다.** LLaMA2의 우위는 **허상**이었다 — 무지(ignorance)를 귀속하는 편향의 결과일 가능성이 크다. 반대로 GPT의 부진은 추론 실패가 아니라 **결론 확정을 지나치게 보수적으로 회피하는 태도**에서 비롯되었다.

**방법론적 교훈**: 벤치마크 점수의 **방향(높다/낮다) 자체가 능력을 뜻하지 않는다.** 왜 그 점수가 나왔는지를 조작 실험으로 분해해야 한다. 이것은 이 블로그의 [LLM Harness 최적화·오류](../llm-harness-optimization-errors/)에서 다룬 평가 하네스 문제와 정확히 같은 구조다.

관련: **Ullman 2023**, "Large language models fail on trivial alterations to theory-of-mind tasks" — §5.1·§5.2와 같은 섭동 패턴.

### 5.4 Centaur — 인간 인지의 파운데이션 모델

**Binz et al. 2025**, "A foundation model to predict and capture human cognition", *Nature* 644 (2025-08-28).

**규모**: **Psych-101** 데이터셋 — 참가자 **6만 명 이상**, 선택 **1,000만 건 이상**, 실험 **160종**의 시행 단위(trial-by-trial) 데이터로 언어모델을 미세조정.

주장:

- 보류된 참가자의 행동을 **기존 인지 모델보다 잘 예측**
- 처음 보는 표지 이야기(cover story), 구조적 과제 변형, **완전히 새로운 영역**으로 일반화
- 신경 활동을 포착하도록 **명시적으로 훈련한 적이 없는데도** 내부 표상이 더 인간 정합적으로 변함

**반론 — Bowers et al. 2025, "Centaur: A model without a theory."**

제목이 논지를 전부 말한다. §2.2에서 Bowers가 시각에 대해 했던 비판과 **같은 형태**다.

```
  예측력이 아무리 높아도
        │
        └─ 그것만으로는 ★이론★ 이 아니다
              │
              └─ 왜 인간이 그렇게 행동하는지 설명하지 않는다
```

**이 대립이 §10의 종합 주제다.**

---

## 6. 축 6: 측정 논쟁 — 무엇을 지능의 증거로 인정할 것인가

### 6.1 Chollet 2019 — 지능의 정의를 바꾸다

**"On the Measure of Intelligence"** — 여기서 **ARC(Abstraction and Reasoning Corpus)** 가 제안되었다.

정의:

> **지능은 미지의 과제에 대한 기술 습득 효율(skill-acquisition efficiency)이다.**

**이것이 왜 급진적인가**: 기존 벤치마크는 **기술(skill)** 을 측정한다. 체스를 잘 두는가, 번역을 잘하는가. Chollet은 기술이 아니라 **새 기술을 얼마나 빨리 얻는가**를 측정하자고 한다.

```
  기술 측정      : 학습 후 성능        → 데이터·계산으로 살 수 있다
  습득 효율 측정 : 얼마나 적은 경험으로 → 살 수 없다 (사전 지식이 결정)
```

이 정의는 §7의 데이터 효율 격차와 **같은 이야기**다. 다른 언어로 표현되었을 뿐이다.

### 6.2 ARC-AGI-2 (2025)

**"ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems"** — arXiv:2505.11831.

추상 추론에 대해 더 세밀한 신호를 주도록 과제를 재구성한 차세대 벤치마크. 리더보드 수치는 계속 변하므로 여기 고정하지 않는다(공식 리더보드 참조).

### 6.3 ★ ARC-AGI-3 (2026) — 격차가 가장 크게 드러난 지점

**"ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence"** — arXiv:2603.24621 / [arcprize.org](https://arcprize.org/arc-agi/3).

**2019년 ARC 도입 이래 첫 형식 변경**이다.

| | ARC-AGI-1/2 | **ARC-AGI-3** |
|---|---|---|
| 형식 | 정적 퍼즐 | **대화형 환경** (턴 기반) |
| 요구 능력 | 규칙 추론 | **탐색 · 목표 추론 · 환경 동역학의 내부 모델 구축 · 계획** |
| 지시 | 주어짐 | **명시적 지시 없음** |
| 규모 | — | 추상 추론 환경 **135종** |

**결과 (2026년 3월 기준)**:

```
  인간          ████████████████████████████████████████  100%
  프론티어 AI   ▏                                          0.51%
```

채점은 **RHAE**(Relative Human Action Efficiency) — 인간 기준선 대비 레벨당 행동 효율을 게임별로 정규화한 지표다. 인간 기준선은 각 게임을 **처음 접하는** 참가자들의 통제 실험으로 설정한다.

**이 숫자가 이 글 전체에서 가장 중요할 수 있다.** §3에서 본 대로 언어 정합에서는 모델이 "설명 가능 분산의 거의 100%"를 예측한다. 그런데 **지시 없이 낯선 환경을 탐색해 목표를 알아내는** 과제에서는 1% 미만이다.

그리고 이것은 §1의 **체화된 튜링 테스트**가 겨냥한 바로 그 능력이다. Zador 등이 2023년에 "동물이 5억 년간 진화시킨 능력"이라고 지목한 것을 2026년에 측정했더니 0.51%가 나온 셈이다.

### 6.4 Schaeffer et al. 2023 — 창발은 신기루인가

**"Are Emergent Abilities of Large Language Models a Mirage?"** — NeurIPS 2023 **Outstanding Paper**.

주장:

> 특정 과제·모델군에서 **고정된 모델 출력을 분석할 때**, 창발적 능력은 모델 행동의 근본적 변화가 아니라 **연구자의 지표 선택** 때문에 나타난다.

```
  비선형·불연속 지표 (예: 정확 일치)  → 갑작스러운 창발처럼 보임
  선형·연속 지표      (예: 토큰 편집거리) → 매끄럽고 예측 가능한 향상
```

**함의**: "규모가 커지면 질적으로 새로운 능력이 나타난다"는 서사의 상당 부분이 **측정 도구의 산물**일 수 있다. 지능 논쟁에서 이것이 중요한 이유는, 창발이 "기계가 갑자기 이해하기 시작했다"는 주장의 주요 근거로 쓰여왔기 때문이다.

### 6.5 추론 모델 논쟁 (2025)

**Shojaee et al. 2025, "The Illusion of Thinking"** (Apple, arXiv:2506.06941).
하노이 탑, 강 건너기, 블록 월드 등 **오염 없는 퍼즐 환경**에서 난이도를 정밀 조절. 복잡도가 임계를 넘으면 대형 추론 모델(LRM) 성능이 **붕괴**한다고 보고.

**반론 — Opus & Lawsen 2025, "The Illusion of the Illusion of Thinking"** (arXiv:2506.09250).
보고된 붕괴가 추론 실패가 아니라 **평가 방법론의 산물**일 수 있다는 지적. 핵심은 **출력 절단(token limit)을 추론 실패로 오독**했다는 것 — 많은 하노이 탑 실패에서 모델은 **길이 제약 때문에 멈춘다고 명시적으로 말하고 있었다.**

**교훈**: 이 논쟁 자체가 §5.3(Strachan의 조작 실험), §6.4(지표 선택)와 같은 교훈을 반복한다.

> **벤치마크 점수는 능력에 대한 진술이 아니라 "능력 × 평가 하네스"에 대한 진술이다.**

이 블로그의 [LLM Harness 최적화·오류](../llm-harness-optimization-errors/)가 다룬 주제가 그대로 지능 논쟁의 중심에 있다.

---

## 7. 축 7: 데이터 효율 격차 — 남은 진짜 격차

### 7.1 Frank 2023 — 4~5자릿수

**"Bridging the data gap between children and large language models"** — *Trends in Cognitive Sciences* (2023-08).

```
  아동     :  수백만 단어         → 제로샷 일반화, 맥락 내 학습 나타남
  LLM      :  수십억 단어         → 같은 행동이 나타남

  격차     :  ★ 4~5 자릿수 (10,000~100,000배) ★
```

Frank가 제시하는 후보 설명:

1. 아동의 **선재하는 개념 지식**
2. **다중양식 접지(multimodal grounding)**
3. 입력의 **상호작용적·사회적 성격**

### 7.2 BabyLM Challenge

**"Findings of the BabyLM Challenge"** — arXiv:2504.08165 (2025-04), 2회차는 arXiv:2412.05149.

**설계**: 고정된 데이터 예산에서 언어모델 훈련을 최적화하는 공동 경진. 문법 능력, 다운스트림 성능, 일반화를 평가.

전제:

> 아동은 **1억 단어 미만**의 입력으로 언어를 습득한다. LLM은 통상 **3~4자릿수 더 많은 데이터**를 요구하며, 그러고도 많은 평가에서 인간만큼 하지 못한다.

30편 이상의 제출에서 데이터 효율적 언어모델 훈련에 대한 구체적 권고를 도출했다.

**이 축이 중요한 이유**: §2~§5의 논쟁은 "결과가 비슷한가"를 묻는다. 이 축은 **"같은 결과를 얼마의 비용으로 얻는가"** 를 묻는다. 그리고 이 축에서는 **격차가 명확하고 논쟁이 없다.** 아무도 LLM이 인간만큼 데이터 효율적이라고 주장하지 않는다.

§6.1의 Chollet 정의를 따르면, 데이터 효율 격차는 곧 **지능 격차**다.

---

## 8. 축 8: 학습 규칙과 표상 수렴

### 8.1 뇌는 역전파를 하지 않는다

가장 오래된 불일치다. 역전파는 전역적 오차 신호의 정확한 역방향 전파를 요구하는데, 생물학적 신경계에서 이를 지지하는 증거는 약하다(가중치 수송 문제, 시간적 국소성, 별도 순방향/역방향 위상 등).

이 축은 이 블로그에서 이미 여러 편으로 다루었다.

| 접근 | 포스트 |
|---|---|
| 예측 부호화 / 전향적 배치 | [Prospective Configuration 리뷰](../prospective-configuration-review/) · [코드 분석](../prospective-config-code-analysis/) |
| Forward-Forward | [Forward-Forward 리뷰](../forward-forward-review/) |
| 평형 전파 | [DSF-EqProp 코드 분석](../dsf-eqprop-code-analysis/) |
| 메타 예측부호화망 | [Meta-PCN 리뷰](../meta-pcn-review/) |
| 비역전파 전반 | [비역전파 방법 서베이](../no-bp-methods-survey/) · [비역전파 LM 서베이](../no-bp-lm-survey/) |

**이 축의 성격이 다른 축과 다르다는 점**에 주목할 만하다. §2~§5는 **결과의 유사성**을 묻는다. 이 축은 **과정의 유사성**을 묻는다. 그리고 두 시스템이 같은 결과에 다른 과정으로 도달할 수 있다는 것은 잘 알려진 사실이다(다중 실현 가능성).

### 8.2 Huh et al. 2024 — 플라톤적 표상 가설

**"The Platonic Representation Hypothesis"** — ICML 2024 position paper. Huh, Cheung, Wang, Isola.

주장:

> 서로 다른 신경망이 데이터를 표상하는 방식이 **시간이 지날수록, 여러 영역을 가로질러 점점 정렬되고 있다.** 비전 모델과 언어 모델이 커질수록 데이터 포인트 사이의 거리를 점점 더 비슷하게 측정한다.

측정: 커널 기반 정렬 지표 — 두 모델이 유사한 데이터 포인트를 특징 공간에서 가깝게 배치하면 정렬된 것으로 본다.

해석: 이 수렴이 **현실의 공유된 통계 모델**을 향해 가고 있으며, 이는 플라톤의 이데아 개념과 유사한 **이상적 표상 공간**의 존재를 시사한다.

**지능 논쟁에서의 위치**: 만약 충분히 강력한 시스템이 **모두 같은 표상으로 수렴**한다면, 인공 시스템과 생물학적 시스템의 정합(§2, §3)은 우연이 아니라 **필연**이 된다. 반대로 이 가설이 틀렸다면, 관측된 정합은 아키텍처·데이터의 우연한 공통점일 수 있다.

**주의**: 이것은 **position paper**이며 증명이 아니다. 반론과 조건부 검증이 계속 나오고 있다.

---

## 9. 한눈에 보는 지도

```
                        "기계와 뇌는 같은 것을 하는가"
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
   [결과가 같은가]              [과정이 같은가]            [비용이 같은가]
        │                           │                           │
   ┌────┴─────┐              ┌──────┴──────┐                    │
   │          │              │             │                    │
 표상 정합   행동 일치     학습 규칙    세계 모델           데이터 효율
   §2 §3      §5            §8.1         §4.4                  §7
   │          │              │             │                    │
   ├ 시각     ├ 인지검사     ├ 역전파      ├ 택시 실험          ├ 4~5 자릿수
   │  Yamins  │  Binz       │  불일치     │  경로는 최적       │  격차
   │  ↕반론   ├ 유추        ├ 예측부호화  │  지도는 비정합      │
   │  Bowers  │  Webb↔Hodel │  FF/EqProp  │                    ├ BabyLM
   │  Feather ├ ToM         │             │                    │  <1억 단어
   │          │  Strachan   └─────────────┘                    │
   ├ 언어     └ Centaur                                        └ ★논쟁 없음★
   │  Schrimpf   ↕Bowers
   │  Goldstein
   │  ★Tuckute (개입)
   │
   └ ⚠ Fedorenko 2024: 언어 ≠ 사고
        → 언어 정합의 성공이 사고의 정합을 함의하지 않는다

                    [측정 자체가 논쟁 대상] §6
                    Chollet 정의 · ARC-AGI-3 (인간 100% vs AI 0.51%)
                    Schaeffer 창발 신기루 · Illusion of Thinking 공방
```

---

## 10. 종합 (1) — 여러 축을 가로지르는 패턴: 섭동 비대칭

> 이 절과 §11은 원 논문들의 주장이 아니라, 그것들을 놓고 필자가 정리한 해석이다.

### 10.1 같은 현상이 세 축에서 독립적으로 나타났다

| 축 | 원 과제 성능 | 변형 과제 성능 | 인간 |
|---|---|---|---|
| **인지심리 삽화** (Binz & Schulz 2023) | 인간 대등 이상 | **작은 섭동에 크게 빗나감** | 유지 |
| **유추 추론** (Webb 2023 ↔ Hodel & West 2023) | 인간 대등 이상 | **가장 단순한 변형에서 실패** | **모든 변형에서 일관** |
| **마음이론** (Strachan 2024, Ullman 2023) | 다수 항목 인간 수준 | **사소한 변경에 실패** | 유지 |
| **세계 모델** (Vafa 2024) | 최단 경로를 찾아냄 | **지도는 실제와 거의 무관** | — |
| **지각 불변성** (Feather 2023) | 벤치마크 점수 높음 | **메타머가 인간에게 인식 불가** | — |

이 다섯 줄은 **서로 다른 연구 공동체**(인지심리, 유추 연구, ToM 연구, ML 이론, 지각신경과학)에서 **독립적으로** 나온 결과다. 그런데 형태가 같다.

### 10.2 패턴의 정식화

```
  성능(원 과제)     ≈ 인간
  성능(변형 과제)   ≪ 인간
  ───────────────────────────
  ⇒ 두 시스템의 성능이 겹치는 영역은 있지만,
     그 성능을 지지하는 ★표상의 일반화 구조★ 가 다르다
```

**왜 이것이 유용한 진단인가**: "이해하는가"는 이분법이라 결론이 나지 않는다. 하지만 **"어떤 변형 축을 따라 성능이 유지되는가"** 는 측정 가능하고, 실제로 §5.2에서 논쟁이 이 형태로 수렴하며 진전했다.

**함의**: 능력을 주장하는 논문은 **변형 조건을 함께 보고해야 한다.** 원 과제 성능만 있는 주장은 §10.1의 다섯 사례 때문에 이제 약한 증거로 취급되어야 한다.

---

## 11. 종합 (2) — 이 분야의 단층선: 예측 ≠ 설명

### 11.1 같은 비판이 다른 대상에 반복된다

| 대상 | 예측 성능 주장 | 비판 |
|---|---|---|
| 시각 DNN | 신경 반응·오류 패턴 예측 1위 | Bowers 2023: *"어떤 특징이 기여하는지 검정하지 않는다"* |
| 언어 모델 | 설명 가능 분산 거의 100% | Antonello & Huth: 매핑 유연성에 의존 / Fedorenko: 언어 ≠ 사고 |
| Centaur | 기존 인지 모델보다 예측 우수 | Bowers 2025: *"이론 없는 모델"* |

**Bowers가 2023년 시각에 대해 한 비판과 2025년 Centaur에 대해 한 비판은 같은 논증이다.** 대상만 바뀌었다.

### 11.2 무엇이 걸려 있는가

```
  [예측주의 입장]
    모델이 데이터를 잘 예측하면 그것이 곧 좋은 모델이다
    벤치마크 점수로 진보를 측정할 수 있다
         │
         └─ 강점: 측정 가능, 누적적, 반증 가능
         └─ 약점: 왜 그런지 설명하지 않는다

  [설명주의 입장]
    예측은 필요조건일 뿐, 기제에 대한 가설을 검정해야 한다
    반증적 자극(메타머), 조작 실험, 변형 조건이 필요하다
         │
         └─ 강점: 실제 기제에 접근, §10의 실패를 잡아낸다
         └─ 약점: 정량화·누적이 어렵다
```

**이 단층선이 이 분야의 논쟁이 좀처럼 종결되지 않는 이유다.** 양측이 다른 증거 기준을 쓰고 있기 때문에, 새 결과가 나와도 각자의 기준으로 해석된다.

### 11.3 이 대립을 실제로 넘어선 사례

**Tuckute et al. 2024**(§3.3)가 유일하게 양측 기준을 동시에 만족시킨 사례로 보인다.

```
  예측주의 요건:  인코딩 모델이 1,000 문장의 반응 크기를 예측한다      ✅
  설명주의 요건:  그 모델로 ★설계한 개입★ 이
                  ★새로운 피험자★ 에게서 예측대로 작동한다              ✅
```

**상관을 넘어 개입으로 간 것**이 결정적이다. 예측 모델이 우연히 맞을 수는 있어도, 그 모델로 설계한 자극이 새 사람의 뇌를 의도대로 구동·억제하는 것은 우연으로 설명하기 어렵다.

**이것이 이 분야가 나아가야 할 방법론적 방향으로 보인다** — 벤치마크 점수를 올리는 대신, 모델을 이용해 **검증 가능한 개입을 설계**하는 것.

---

## 12. 현재 시점의 정직한 요약

### 12.1 합의된 것

| # | 명제 | 근거 |
|---|---|---|
| 1 | 과제 최적화 신경망은 시각·언어 피질 반응을 **상당히 잘 예측**한다 | §2.1, §3.1, §3.2 |
| 2 | 언어 영역의 예측력은 **다음 단어 예측 성능과 결합**되어 있다 | §3.1 |
| 3 | 그 예측 모델로 **인간 뇌 활동을 인과적으로 구동·억제**할 수 있다 | §3.3 |
| 4 | 그럼에도 **불변성 구조**는 인간과 다르다 | §2.3 |
| 5 | 과제 성능이 높아도 **정합적 세계 모델**을 보장하지 않는다 | §4.4 |
| 6 | 원 과제에서 인간과 대등해도 **변형에서 무너진다** | §10 |
| 7 | **데이터 효율 격차는 4~5자릿수이며 논쟁이 없다** | §7 |
| 8 | 지시 없는 **대화형 탐색 과제에서 격차가 가장 크다** (인간 100% vs AI 0.51%) | §6.3 |

### 12.2 합의되지 않은 것

- LLM이 **이해**하는가 (§4 전체 — 그리고 Mitchell & Krakauer의 지적대로 이 질문은 아마 잘못 세워져 있다)
- 형식만으로 **지시(reference)** 가 성립할 수 있는가 (§4.1 ↔ §4.3)
- 예측 성능이 **모델의 타당성**을 얼마나 뒷받침하는가 (§11)
- 창발이 실재인가 **지표의 산물**인가 (§6.4)
- 표상이 보편적으로 **수렴**하는가 (§8.2 — position paper 단계)
- 뇌와 다른 **학습 규칙**을 쓰는 것이 문제인가, 아니면 다중 실현의 사례일 뿐인가 (§8.1)

### 12.3 이 분야를 읽는 실용적 규칙

논문을 만났을 때 다음 순서로 점검하면 대부분 위치를 잡을 수 있다.

```
① 무엇을 증거로 쓰는가?
     예측 정확도 / 행동 일치 / 반증적 자극 / 인과 개입 / 표본 효율

② 변형 조건을 보고했는가?
     원 과제만 있으면 §10 때문에 할인해서 읽는다

③ 매핑을 얼마나 유연하게 허용했는가?
     선형? 비선형? 유연할수록 점수는 오르고 주장은 약해진다 (§3.4)

④ 점수의 방향을 조작 실험으로 분해했는가?
     Strachan 2024가 LLaMA2의 "우위"를 허상으로 밝힌 방식 (§5.3)

⑤ 평가 하네스가 결과를 만들지는 않았는가?
     Illusion of Thinking 공방, 창발 신기루 (§6.4~§6.5)
```

---

## 13. 읽는 순서 추천

**처음 1편만 읽는다면**
[Mitchell & Krakauer 2023, PNAS](https://www.pnas.org/doi/10.1073/pnas.2215907120) — 논쟁 전체의 지도

**뇌-모델 정합 축을 보려면 (순서대로)**
1. [Tuckute, Kanwisher & Fedorenko 2024, *Annu. Rev. Neurosci.*](https://www.annualreviews.org/content/journals/10.1146/annurev-neuro-120623-101142) — 종합 리뷰
2. [Schrimpf et al. 2021, PNAS](https://www.pnas.org/doi/10.1073/pnas.2105646118) — 핵심 결과
3. [Tuckute et al. 2024, *Nat. Hum. Behav.*](https://www.nature.com/articles/s41562-023-01783-7) — 인과 개입
4. [Bowers et al. 2023, *BBS*](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/abs/deep-problems-with-neural-network-models-of-human-vision/ABCE483EE95E80315058BB262DCA26A9) + [Feather et al. 2023, *Nat. Neurosci.*](https://www.nature.com/articles/s41593-023-01442-0) — 반론

**격차를 보려면**
1. [Frank 2023, *TiCS*](https://www.cell.com/trends/cognitive-sciences/abstract/S1364-6613(23)00203-6) — 데이터 효율
2. [ARC-AGI-3 기술 보고서](https://arxiv.org/abs/2603.24621) — 상호작용 격차
3. [Vafa et al. 2024, NeurIPS](https://arxiv.org/abs/2406.03689) — 세계 모델

**평가 방법론을 보려면**
[Schaeffer et al. 2023, NeurIPS](https://arxiv.org/abs/2304.15004) → [Strachan et al. 2024](https://www.nature.com/articles/s41562-024-01882-z) → [Illusion of Thinking](https://arxiv.org/abs/2506.06941) + [반론](https://arxiv.org/html/2506.09250v1)

---

## 참고문헌

### 분야 선언 / 리뷰
- Zador, A. et al. (2023). Catalyzing next-generation Artificial Intelligence through NeuroAI. *Nature Communications* 14:1597. [링크](https://www.nature.com/articles/s41467-023-37180-x)
- Tuckute, G., Kanwisher, N., Fedorenko, E. (2024). Language in Brains, Minds, and Machines. *Annual Review of Neuroscience* 47:277–301. [링크](https://www.annualreviews.org/content/journals/10.1146/annurev-neuro-120623-101142)
- Mitchell, M., Krakauer, D.C. (2023). The debate over understanding in AI's large language models. *PNAS* 120(13). [링크](https://www.pnas.org/doi/10.1073/pnas.2215907120)

### 표상 정합 — 시각
- Yamins, D.L.K. et al. (2014). Performance-optimized hierarchical models predict neural responses in higher visual cortex. *PNAS* 111(23):8619–8624. [링크](https://www.pnas.org/doi/10.1073/pnas.1403112111)
- Bowers, J.S. et al. (2023). Deep problems with neural network models of human vision. *Behavioral and Brain Sciences* 46. [링크](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/abs/deep-problems-with-neural-network-models-of-human-vision/ABCE483EE95E80315058BB262DCA26A9)
- Feather, J. et al. (2023). Model metamers reveal divergent invariances between biological and artificial neural networks. *Nature Neuroscience*. [링크](https://www.nature.com/articles/s41593-023-01442-0)

### 표상 정합 — 언어
- Schrimpf, M. et al. (2021). The neural architecture of language: Integrative modeling converges on predictive processing. *PNAS* 118(45):e2105646118. [링크](https://www.pnas.org/doi/10.1073/pnas.2105646118) · [코드](https://github.com/mschrimpf/neural-nlp)
- Goldstein, A. et al. (2022). Shared computational principles for language processing in humans and deep language models. *Nature Neuroscience* 25(3). [링크](https://www.nature.com/articles/s41593-022-01026-4)
- Tuckute, G. et al. (2024). Driving and suppressing the human language network using large language models. *Nature Human Behaviour*. [링크](https://www.nature.com/articles/s41562-023-01783-7)
- Antonello, R., Huth, A. et al. Language models and brains align due to more than next-word prediction and word-level information. EMNLP 2024. [링크](https://arxiv.org/abs/2212.00596)
- Fedorenko, E., Piantadosi, S.T., Gibson, E.A.F. (2024). Language is primarily a tool for communication rather than thought. *Nature*. [링크](https://www.nature.com/articles/s41586-024-07522-w)

### 이해 · 접지 · 세계 모델
- Bender, E.M., Koller, A. (2020). Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data. ACL 2020. [링크](https://aclanthology.org/2020.acl-main.463/)
- Coelho Mollo, D., Millière, R. (2023). The Vector Grounding Problem. [arXiv:2304.01481](https://arxiv.org/abs/2304.01481)
- Vafa, K. et al. (2024). Evaluating the World Model Implicit in a Generative Model. NeurIPS 2024. [arXiv:2406.03689](https://arxiv.org/abs/2406.03689) · [코드](https://github.com/keyonvafa/world-model-evaluation)

### 기계심리학
- Binz, M., Schulz, E. (2023). Using cognitive psychology to understand GPT-3. *PNAS* 120(6). [링크](https://www.pnas.org/doi/10.1073/pnas.2218523120)
- Webb, T., Holyoak, K.J., Lu, H. (2023). Emergent analogical reasoning in large language models. *Nature Human Behaviour*. [링크](https://www.nature.com/articles/s41562-023-01659-w)
- Hodel, D., West, J. (2023). Response: Emergent analogical reasoning in large language models. [arXiv:2308.16118](https://arxiv.org/abs/2308.16118)
- Webb, T. et al. (2024). Evidence from counterfactual tasks supports emergent analogical reasoning in large language models. [arXiv:2404.13070](https://arxiv.org/pdf/2404.13070)
- Strachan, J.W.A. et al. (2024). Testing theory of mind in large language models and humans. *Nature Human Behaviour*. [링크](https://www.nature.com/articles/s41562-024-01882-z)
- Binz, M. et al. (2025). A foundation model to predict and capture human cognition. *Nature* 644. [링크](https://www.nature.com/articles/s41586-025-09215-4) · 반론: Bowers et al. (2025), *Centaur: A model without a theory*

### 측정 · 벤치마크
- Chollet, F. (2019). On the Measure of Intelligence. [arXiv:1911.01547](https://arxiv.org/abs/1911.01547)
- ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems. [arXiv:2505.11831](https://arxiv.org/abs/2505.11831)
- ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence. [arXiv:2603.24621](https://arxiv.org/abs/2603.24621) · [arcprize.org/arc-agi/3](https://arcprize.org/arc-agi/3)
- Schaeffer, R., Miranda, B., Koyejo, S. (2023). Are Emergent Abilities of Large Language Models a Mirage? NeurIPS 2023 (Outstanding Paper). [arXiv:2304.15004](https://arxiv.org/abs/2304.15004)
- Shojaee, P. et al. (2025). The Illusion of Thinking. [arXiv:2506.06941](https://arxiv.org/abs/2506.06941) · 반론 [arXiv:2506.09250](https://arxiv.org/html/2506.09250v1)

### 데이터 효율
- Frank, M.C. (2023). Bridging the data gap between children and large language models. *Trends in Cognitive Sciences*. [링크](https://www.cell.com/trends/cognitive-sciences/abstract/S1364-6613(23)00203-6)
- Warstadt, A., Mueller, A. et al. Findings of the BabyLM Challenge. [arXiv:2504.08165](https://arxiv.org/abs/2504.08165) · 2회차 [arXiv:2412.05149](https://arxiv.org/abs/2412.05149)

### 수렴 / 표상 정합 일반
- Huh, M., Cheung, B., Wang, T., Isola, P. (2024). The Platonic Representation Hypothesis. ICML 2024. [arXiv:2405.07987](https://arxiv.org/abs/2405.07987)
- Sucholutsky, I. et al. (2023). Getting aligned on representational alignment. [관련 논의](https://cocosci.princeton.edu/papers/Sucholutsky2024a.pdf)

---

## 관련 블로그 포스트

- [Prospective Configuration 리뷰](../prospective-configuration-review/) · [코드 분석](../prospective-config-code-analysis/) — 뇌의 학습 규칙
- [Forward-Forward 리뷰](../forward-forward-review/) — 역전파 없는 학습
- [비역전파 방법 서베이](../no-bp-methods-survey/) · [비역전파 LM 서베이](../no-bp-lm-survey/)
- [Meta-PCN 리뷰](../meta-pcn-review/) — 예측 부호화망
- [Bayesian Surprise](../bayesian-surprise/) — §3.2의 놀람 신호와 직결
- [LLM Harness 최적화·오류](../llm-harness-optimization-errors/) — §6의 평가 방법론 문제
