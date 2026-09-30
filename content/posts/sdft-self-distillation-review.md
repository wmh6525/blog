---
title: "[리뷰] SDFT — 데모만으로 온폴리시 학습하기: Self-Distillation Enables Continual Learning"
date: 2026-09-30
tags: ["논문리뷰", "SDFT", "Self-Distillation", "On-Policy Distillation", "지속학습", "Catastrophic Forgetting", "SDPO", "파인튜닝"]
categories: ["ML/AI"]
summary: "arXiv 2601.19897(MIT·ETH) 리뷰. 이 논문은 SFT가 잊는 이유를 오프폴리시에서 찾는다. 해법으로 같은 모델에 데모를 in-context로 보여준 것을 교사로 삼아, 학생 자신의 샘플 위에서 증류한다. 보상 없이 데모만으로 온폴리시 학습을 하는 셈이다. Qwen2.5-7B에서 세 과제 모두 SFT보다 새 과제를 더 배우고(과학 70.2 vs 66.2), 기존 능력 평균도 base 65.5에 가깝게 지킨다(64.5~65.4 vs SFT 53.4~60.2). 다만 짚을 점이 있다. 논문 이론은 reverse KL인데 실제 결과는 모두 forward KL로 학습됐다(v2·README 정정). 3B에서는 오히려 SFT보다 나쁘고, 비용은 FLOPs 2.5배·wall-clock 4배다. 후속 연구에서 나온 비판(다양성 붕괴, 콜드스타트, 낡은 사실 갱신 실패, 이동 교사 누적 드리프트)과 형제 논문 SDPO, 메모리 레이어와의 결합 가능성까지 정리한다."
math: true
toc: true
draft: false
---

## 논문 정보

| 항목 | 내용 |
|---|---|
| 제목 | **Self-Distillation Enables Continual Learning** |
| arXiv | [2601.19897](https://arxiv.org/abs/2601.19897) — v1 2026-01-27, v2 2026-08-07 (cs.LG) |
| 저자 | Idan Shenfeld (교신), Mehul Damani, Jonas Hübotter, Pulkit Agrawal |
| 소속 | MIT · Improbable AI Lab · ETH Zurich |
| 코드 | [idanshen/Self-Distillation](https://github.com/idanshen/Self-Distillation) (HF TRL 기반) · [프로젝트 페이지](https://self-distillation.github.io/SDFT.html) |
| 게재 | arXiv 프리프린트. ICML 2026으로 소개되곤 하지만 arXiv·PDF·프로젝트 페이지 어디에서도 확인되지 않는다 |

제1저자는 [RL's Razor](https://arxiv.org/abs/2509.04259)의 Shenfeld다. 이 논문은 그 논문의 직접적인 후속이다. RL's Razor가 "RL이 덜 잊는 건 온폴리시라서"라고 진단했다면, SDFT는 **"보상이 없고 데모만 있어도 온폴리시로 학습할 수 있다"**는 처방이다.

---

## 1. 한 줄 요약

> **데모를 보여준 자기 자신을 교사로 삼아, 자기가 생성한 답 위에서 그 교사를 따라가게 한다.**

```
            ┌─────────────── 같은 가중치 (교사는 EMA) ───────────────┐
            │                                                     │
  질문 x ──▶ 학생 π_θ(·|x)            질문 x + 데모 c ──▶ 교사 π(·|x, c)
            │                                                     │
            └──▶ 샘플 y ~ π_θ  ──────────────▶  토큰별로 교사 분포와 맞춤
                  (온폴리시)                     (KL, 교사는 stop-grad)
```

SFT는 데모 $c$를 **정답 토큰열로 외운다**. SDFT는 데모를 **교사의 문맥으로만** 쓴다. 학습 신호는 학생이 스스로 만든 궤적 위에서, "데모를 봤다면 나는 이 토큰을 얼마나 다르게 냈을까"로부터 나온다.

---

## 2. 문제 — SFT는 왜 잊는가

### 2.1 RL's Razor의 망각 법칙

RL's Razor(Shenfeld, Pari, Agrawal, 2025.09)의 핵심 주장은 이렇다.

- 망각량은 알고리즘이 아니라 **새 과제 분포에서 잰 base와 fine-tuned 정책의 KL**, 즉 $\mathbb{E}_{x\sim\tau}\,\mathrm{KL}(\pi_0 \,\|\, \pi)$로 예측된다
- 과제를 푸는 해가 여럿일 때, 온폴리시 RL은 **KL이 최소인 해**로 편향된다. SFT는 데이터가 가리키는 곳이면 base에서 임의로 먼 해로도 간다
- 이차 적합 $R^2$: 토이(ParityMNIST)에서 0.96, LLM 실험에서 0.71
- **결정적 실험**: 해석적으로 KL이 최소인 정답 라벨로 SFT를 하면("Oracle SFT") RL보다도 덜 잊는다. 즉 RL 자체가 아니라 **암묵적 KL 최소화**가 원인이다

### 2.2 그럼 RL을 쓰면 되지 않나

RL은 **보상**이 필요하다. 그런데 실무의 많은 상황에는 데모만 있다(전문가 풀이, 도구 호출 예시, 의료 답변). 역강화학습(IRL)으로 보상을 복원할 수는 있지만, 강한 구조적 가정이 필요하고 스케일하지 않는다.

그래서 저자들이 던진 질문은 이렇다. **데모만 가지고 온폴리시·KL 최소 쪽으로 학습할 수 있는가?**

### 2.3 답의 재료 — 두 선행 아이디어

| 선행 | 내용 | SDFT가 가져온 것 |
|---|---|---|
| **Context distillation** (Snell et al. 2022) | 지시문·예시를 문맥에 넣은 모델의 출력을, 문맥 없는 같은 모델에 학습시킨다 | "문맥을 가진 자기 자신 = 교사" |
| **On-policy distillation** (GKD, Agarwal et al. 2023 · DAgger) | 학생이 생성한 시퀀스 위에서 교사 분포를 따라가게 한다 | "학생 자신의 샘플 위에서" |

Context distillation은 **교사 샘플**로 학습하는 오프폴리시였고, 문맥도 고정된 프리픽스였다. SDFT는 두 가지를 바꿨다. 학습은 **온폴리시**로, 문맥은 **질의마다 다른 데모**로.

---

## 3. 방법

### 3.1 교사 프롬프트

```
<Question>
This is an example for a response to the question:
<Demonstration>
Now answer with a response of your own, including the thinking process:
```

"네 방식으로 답하라"는 지시 덕분에 교사가 데모를 그대로 베끼지 않는다고 저자들은 말한다. 교사의 출력은 **데모의 내용을 흡수했지만 말투와 추론 스타일은 base 모델 자신의 것**이 된다. 이것이 SDFT가 base에 가까이 머무는 이유다.

### 3.2 목적함수 — 논문에 적힌 것

$$
\mathcal{L}(\theta) = D_{\mathrm{KL}}\big(\pi_\theta(\cdot\mid x)\,\big\|\,\pi(\cdot\mid x, c)\big) = \mathbb{E}_{y\sim\pi_\theta}\left[\log\frac{\pi_\theta(y\mid x)}{\pi(y\mid x,c)}\right]
$$

학생 샘플 위의 **reverse KL**이다. 그래디언트는 토큰마다 어휘 전체에 대해 해석적으로 계산하고, 교사는 stop-gradient로 둔다.

$$
\nabla\mathcal{L} = \mathbb{E}_{y\sim\pi_\theta}\sum_t\sum_{y_t\in V}\pi_\theta(y_t\mid y_{<t},x)\,\log\frac{\pi_\theta(y_t\mid y_{<t},x)}{\pi(y_t\mid y_{<t},x,c)}\,\nabla\log\pi_\theta(y_t\mid y_{<t},x)
$$

### 3.3 ⚠️ 실제로 학습에 쓰인 것 — forward KL

**이 논문을 읽을 때 가장 중요한 정정이다.**

- v2 §3 "Practical Implementation"에 한 문장이 추가됐다: *"Although the theory points to Reverse KL as a suitable loss, we found in practice that Forward KL yields the best performance."*
- 저장소 README(2026-04-07 업데이트): *"all the results in our paper were produced using on-policy sampling, but per-token forward KL loss (similar to the GKD paper)… this is the default argument in this repo."*
- 설정의 `alpha`는 GKD와 같다. 0.0이 forward KL(기본값), 1.0이 reverse KL, 그 사이는 JSD다

정리하면 **샘플은 학생에서 뽑고(온폴리시), 손실은 토큰별 forward KL**이다. 사실상 "자기 자신을 교사로 쓰는 GKD"다. 아래 §4의 이론(IRL 해석)은 reverse KL에 대한 것이므로, **학습된 모델을 엄밀하게 설명하지는 않는다.** v1만 읽은 2차 자료들(이 블로그의 [저망각 파인튜닝 서베이](../low-forgetting-finetuning-survey-2026/) 포함)은 "reverse-KL로 맞춘다"고 적고 있으니 주의하자.

### 3.4 교사 가중치 — EMA가 필수

| 교사 | 결과 |
|---|---|
| 고정된 base | 안정적이지만 **일관되게 성능이 낮다**. 학생이 배운 것을 교사가 반영하지 못한다 |
| 현재 학생 그대로 | **심하게 불안정**. 온폴리시 루프에서 작은 흔들림이 증폭되어 발산한다 |
| **학생의 EMA** | 채택. $\phi \leftarrow \alpha\theta + (1-\alpha)\phi$, $\alpha \in \{0.01, 0.02, 0.05\}$. 저장소 기본값 0.01, 매 스텝 동기화 |

### 3.5 기타 구현 선택

- **KL 추정기 3종 비교**(App. A.1): 샘플 기반 토큰 수준 / 해석적 토큰 수준 / Rao-Blackwellized(비편향). **해석적 토큰 수준**이 가장 안정적이고 성능도 좋다. RB는 측정 가능한 이득이 없었다
- **롤아웃은 프롬프트당 1개**. 여러 개 뽑아도 이득이 미미했다. GRPO와 대비되는 비용상 장점이다
- **처음 몇 토큰은 손실에서 제외**(저장소 기본 3토큰). 학생이 "Based on the text…", "Following the example…" 같은 교사의 말버릇을 물려받는 문제 때문이다
- vLLM 샘플러와 트레이너 사이의 불일치를 importance sampling으로 보정
- **Full fine-tuning**, 실험당 H200 1장. AdamW, lr {5e-6, 1e-5, 5e-5}, 배치 {16, 32, 64}
- SDFT는 보통 2에폭(스킬) 또는 4에폭(지식)을 돈다. SFT는 1에폭 이후 빠르게 과적합해 이득이 없었다

---

## 4. 이론 — 자기 증류 = 역강화학습

### 4.1 유도

신뢰 영역 RL에서 출발한다.

$$
\pi_{k+1} = \arg\max_\pi\ \mathbb{E}[r] - \beta\,\mathrm{KL}(\pi\,\|\,\pi_k) \quad\Rightarrow\quad \pi^*_{k+1} \propto \pi_k\, e^{r/\beta}
$$

이를 뒤집으면 보상은 $r = \beta(\log\pi^*_{k+1} - \log\pi_k) + C$가 된다. 여기서 **In-Context 가정**을 둔다.

$$
\pi^*_{k+1}(y\mid x) \approx \pi(y\mid x, c)
$$

즉 "데모를 본 모델이 곧 다음 신뢰 영역 스텝의 최적 정책"이라는 가정이다. 그러면 **암묵적 보상**이 나온다.

$$
r(y, x, c) = \log\pi(y\mid x,c) - \log\pi_k(y\mid x),\qquad r_t = \log\frac{\pi(y_t\mid\cdot,c)}{\pi_k(y_t\mid\cdot)}
$$

이 보상으로 정책 그래디언트를 계산하면, 기대값에서 reverse-KL 그래디언트와 같아진다. 저자들의 표현으로 SDFT는 *"자신의 더 현명한, 데모를 아는 버전과 비교해 추론한 보상을 최대화하는 온폴리시 RL"*이다. 보상이 토큰 단위로 나오므로 GRPO의 궤적 단위 보상보다 신용 할당이 촘촘하다.

### 4.2 가정의 두 조건과 경험적 확인

In-Context 가정이 성립하려면 두 조건이 필요하다.

1. **최적성**: 데모를 본 교사가 과제를 거의 완벽하게 푼다
2. **최소 이탈**: 교사가 base에서 KL로 가깝다

ToolAlpaca, Qwen2.5-7B-Instruct에서의 확인 결과:

| 항목 | 값 |
|---|---|
| base 정답률 | 42% |
| 데모를 본 교사 정답률 | **100%** (50개 궤적을 수작업으로 확인, 도구 호출과 CoT 모두 정상) |
| base 대비 KL — SFT 모델 | 1.26 nats |
| base 대비 KL — 교사 | **0.68 nats** |

교사가 SFT 결과물보다 base에 **거의 절반 거리**로 가깝다. RL's Razor의 논리대로라면 그만큼 덜 잊는다.

> 다시 강조하자면, 이 등가성은 **reverse KL**에 대한 것이다. 실제로 쓴 forward KL에서도 "암묵적 보상 RL"이라는 해석이 근사적으로 성립하는지는 논문이 다루지 않는다.

---

## 5. 실험

### 5.1 설정

| 항목 | 내용 |
|---|---|
| 주 모델 | **Qwen2.5-7B-Instruct** (규모 실험: 3B/7B/14B, 추론 모델 실험: Olmo-3-7B-Think) |
| 스킬 과제 | **Science Q&A** (SciKnowEval 화학 L-3, 데모는 GPT-4o 생성) · **Tool Use** (ToolAlpaca) · **Medical** (HuatuoGPT-o1, 영어 약 2만 개, GPT-5-mini 채점) |
| 지식 과제 | 모델 컷오프 이후인 **2025년 자연재해 위키 문서 9개**(약 20만 토큰), GPT-5가 만든 QA |
| 망각 측정 | HellaSwag, TruthfulQA, MMLU, IFEval, Winogrande, HumanEval 평균 |
| baseline | SFT · **DFT**(오프라인 데이터를 importance sampling으로 온폴리시처럼 취급) · **SFT + Re-invoke**(SFT 후 base를 교사로 일반 프롬프트에서 온폴리시 증류, Thinking Machines 방식) |
| 없는 것 | **GRPO 등 RL baseline, 리플레이, LoRA, KL 페널티 SFT, 외부 교사 온폴리시 증류** |

### 5.2 스킬 학습 — Table 5

base의 기존 능력 평균은 **65.5**다(HellaSwag 62.0 / HumanEval 65.8 / IFEval 74.3 / MMLU 71.7 / TruthfulQA 47.9 / Winogrande 71.1).

| 과제 (base) | 방법 | **새 과제** | IFEval | HumanEval | TruthfulQA | **기존 평균** |
|---|---|---|---|---|---|---|
| **Science** (32.1) | SFT | 66.2 | 35.3 | 54.8 | 36.8 | 53.4 |
| | SFT + re-invoke | 66.0 | 52.9 | 63.4 | 45.2 | 60.2 |
| | DFT | 54.8 | 60.4 | 67.0 | 38.8 | 60.2 |
| | **SDFT** | **70.2** | 66.8 | 68.9 | 46.5 | **64.5** |
| **Tool Use** (42.9) | SFT | 63.2 | 49.8 | 50.0 | 37.5 | 56.0 |
| | SFT + re-invoke | 63.1 | 59.1 | 68.9 | 49.1 | 63.7 |
| | DFT | 64.2 | 60.2 | 61.4 | 40.2 | 60.8 |
| | **SDFT** | **70.6** | 71.9 | 68.3 | 47.3 | **65.4** |
| **Medical** (30.1) | SFT | 35.5 | 56.6 | 62.1 | 39.8 | 60.2 |
| | SFT + re-invoke | 35.6 | 67.6 | 63.1 | 42.3 | 62.6 |
| | DFT | 36.2 | **74.6** | 64.6 | 40.1 | 64.0 |
| | **SDFT** | **40.2** | 72.3 | 67.7 | 47.3 | **65.4** |

읽을 거리:

- **"덜 배우고 덜 잊는" 트레이드오프가 아니다.** 세 과제 모두 새 과제 점수가 **가장 높으면서** 기존 능력도 가장 잘 지킨다. 이 점이 이전 글에서 다룬 [Sparse Memory Finetuning](../memory-layers-sparse-finetuning-review/)(LoRA보다 덜 배움)과 결정적으로 다르다
- SFT의 망각은 **IFEval에서 가장 극적**이다. Science에서 74.3 → 35.3으로, 지시 따르기가 반토막 난다. SDFT는 66.8이다
- 그래도 **망각이 0은 아니다.** Science의 IFEval은 −7.5, 평균은 −1.0이다
- Medical에서는 DFT가 기존 능력 평균 64.0으로 꽤 가깝고, IFEval은 DFT가 더 높다. Tool Use의 SFT+re-invoke(63.7)도 멀지 않다

### 5.3 지식 습득 — Table 1

| 방법 | Strict | Lenient | OOD (간접 질문) |
|---|---|---|---|
| Base | 0 | 0 | 0 |
| CPT (원문 연속 사전학습) | 9 | 37 | 7 |
| SFT (QA 쌍) | 80 | 95 | 80 |
| **SDFT** | **89** | **100** | **98** |
| Oracle RAG | 91 | 100 | 100 |

- SDFT가 **정답 문서를 항상 찾아주는 RAG에 근접**한다
- 차이가 가장 큰 곳은 **OOD**(예: "2025년에 국제 인도적 지원이 필요했던 나라는?")다. SFT 80 vs SDFT 98. 사실을 "암송"이 아니라 **다른 질문에도 쓸 수 있는 형태**로 넣었다는 뜻이다
- 교사 문맥 ablation(App. A.2): 답만 주면 37%, 원문만 주면 75%, **원문+답을 주면 89%**
- **이 설정의 망각 수치는 보고되지 않았다.** 문서 9개에 GPT-5가 만들고 GPT-5-mini가 채점한 작은 실험이라는 점도 감안해야 한다

### 5.4 순차 학습 — 여러 스킬 누적 (Fig. 3)

Tool Use → Science → Medical 순으로 한 모델에 연속으로 학습한다(총 약 700~800 스텝).

- **SFT**: 다음 과제가 시작되자마자 이전 과제가 떨어진다("진동")
- **SDFT**: 세 스킬을 **누적**한다

다만 그림은 정규화된 곡선(0 = base, 1 = 최고치)뿐이다. **절대 수치와 순차 학습 후의 일반 능력은 보고되지 않았다.**

### 5.5 규모 — ICL이 약하면 안 된다 (Fig. 5 좌)

Science Q&A에서 SDFT − SFT 격차:

| 모델 | 격차 |
|---|---|
| Qwen2.5-3B | **−3.3** (SDFT가 더 나쁨) |
| Qwen2.5-7B | +4.0 |
| Qwen2.5-14B | **+6.9** |

3B에서는 in-context 학습이 "너무 약해서" 교사가 좋은 신호를 못 준다. 반대로 **모델이 클수록 이득이 커진다.** 방법 자체가 모델의 ICL 능력에 기생하는 구조이므로 당연한 결과다. 이 점은 실무 적용 범위를 정한다.

pass@k(최대 128)에서도 SDFT가 모든 k에서 SFT와 base보다 높다고 보고하며, "엔트로피 붕괴가 없다"는 근거로 제시한다(곡선 수치는 판독 불가).

### 5.6 추론 모델 — 답만 있는 데이터 (Table 2)

Olmo-3-7B-Think, Medical, **추론 과정 없이 최종 답만** 감독으로 줄 때:

| 방법 | 정확도 | 평균 토큰 |
|---|---|---|
| Base | 31.2% | 4,612 |
| + SFT | 23.5% | 3,273 |
| + **SDFT** | **43.7%** | 4,180 |

SFT는 답만 있는 데이터를 외우면서 **추론 길이가 줄고 성능이 base 아래로** 떨어진다. SDFT는 교사가 답을 본 채로 **자기 방식의 추론을 생성**하므로 추론 습관이 보존된다. 실무에서 "정답은 있는데 풀이가 없는" 데이터가 흔하다는 점에서 인상적인 결과다.

### 5.7 온폴리시가 꼭 필요한가 (Fig. 6)

Tool Use에서 다음을 비교했다.

- 교사 샘플로 SFT
- 고정된 교사 출력에 오프라인 KL 증류
- SDFT

오프라인 두 변형은 일반 SFT보다는 낫지만 **일관되게 SDFT보다 못하다.** 좋은 교사만으로는 부족하고 **학생 자신의 분포에서 배우는 것**이 핵심이라는 주장의 근거다(곡선 수치는 판독 불가).

### 5.8 비용

- SFT 대비 **FLOPs 약 2.5배, wall-clock 약 4배**
- GRPO와 달리 프롬프트당 생성 1회, 토큰·로짓 수준 신용 할당
- "SFT → re-invoke 같은 다단계 파이프라인보다는 전체적으로 쌀 수 있다"고 주장하지만 측정값은 없다

---

## 6. 저자가 인정한 한계

1. **말버릇 상속**: "Based on the text…" 같은 교사 특유의 표현을 배운다. 앞 토큰 마스킹으로 막지만 휴리스틱이다
2. **능력 요구**: ICL이 강해야 한다. 3B에서 실패
3. **큰 행동 변화는 어렵다**: 예컨대 비추론 모델을 명시적 CoT를 쓰는 모델로 바꾸는 것은 잘 안 됐다. 교사가 base에서 멀리 갈 수 없으니 당연하다. **최소 이탈이 장점이자 한계**다
4. **일부 망각은 남는다**
5. 향후 과제: RL과의 결합(초기화 또는 혼합 신호), 잡음 섞인 데모, 사용자 대화 같은 비정형 데이터

---

## 7. 비판적으로 읽기

| # | 지점 | 내용 |
|---|---|---|
| 1 | **이론과 구현의 불일치** | IRL 유도, Fig. 2 캡션, App. A.1 추정기 분석이 모두 reverse KL 기준인데, 결과는 전부 forward KL이다. v1에는 이 사실이 없고, v2에 한 문장과 README 정정이 추가됐다 |
| 2 | **baseline 공백** | GRPO, 리플레이, LoRA, "SFT + KL-to-base 페널티"가 없다. 특히 마지막 것은 "온폴리시가 핵심인가, KL 제약이 핵심인가"를 가르는 대조군인데 빠졌다 |
| 3 | **체크포인트 선택** | 새 과제 검증 성능으로 고른 뒤 망각을 쟀다. SFT를 더 일찍 멈추면 덜 잊었을 수 있다. 정확도를 맞춘 비교는 Fig. 4 산점도뿐이다 |
| 4 | **에폭 불균형** | SDFT 2~4에폭 vs SFT 1에폭. 2.5배/4배 비용 수치는 그들 설정 기준이다 |
| 5 | **규모·통계** | 주 결과는 Qwen2.5 한 계열, 14B 이하다. "3 시드, 95% CI"라고 적었지만 표에 CI가 없다. 순차 실험은 정규화 곡선뿐이다 |
| 6 | **데이터의 강한 모델 의존** | Science 데모는 GPT-4o로 만들었다. 지식 실험의 QA 생성과 채점은 GPT-5 계열이다 |
| 7 | **작은 오류** | 정의 전에 "Eq. (6)" 참조, RB 항의 $x$ 중복, "Figure 4a"(실제로는 Fig. 5 좌), ToolAlpaca base 42.9 vs 본문 "42%" |

그럼에도 핵심 결과(새 과제에서 **더 배우면서** 덜 잊음, 답만 있는 데이터로 추론 모델 개선)는 방법의 단순함에 비해 강하다.

---

## 8. 형제 논문 — SDPO (Reinforcement Learning via Self-Distillation)

[arXiv 2601.20802](https://arxiv.org/abs/2601.20802)는 SDFT 다음 날(2026-01-28) 올라왔다. Hübotter(제1저자)와 Shenfeld가 양쪽 모두에 참여한 **자매 논문**이다. SDPO는 SDFT를 "같은 아이디어의 오프라인·데모 버전"이라고 부른다.

| | **SDFT** | **SDPO** |
|---|---|---|
| 교사의 문맥 | 외부 전문가 **데모** | **환경 피드백**(런타임 에러, 실패한 테스트, 심사 텍스트) 또는 같은 GRPO 그룹의 **성공한 형제 롤아웃** |
| 환경 | 없음 (오프라인 데이터셋) | 있음 (온라인 RL) |
| 설정 이름 | 데모 기반 지속학습 | RL with Rich Feedback (RLRF) |
| 손실 | 온폴리시, 실제로는 forward KL | 온폴리시 reverse KL / JSD, top-100 로짓 |
| RL과의 관계 | 암묵적 보상 | GRPO 코드에서 **advantage만 교체**: $A_t = \log\frac{\pi(\hat y_t\mid x,f,\cdot)}{\pi(\hat y_t\mid x,\cdot)}$. $\lambda A_{\text{GRPO}} + (1-\lambda)A_{\text{SDPO}}$ 혼합도 가능 |
| 교사 안정화 | EMA | EMA 또는 초기 교사와 보간 |

**주요 결과**
- Science QA + ToolAlpaca(Qwen3-8B, Olmo3-7B): SDPO 70.2% vs 튜닝된 GRPO 66.6%, 응답 길이는 최대 **11배 짧음**
- LiveCodeBench v6(Qwen3-8B): 48.8% vs GRPO 41.2%. GRPO의 최종 정확도에 **4배 적은 생성**으로 도달
- 망각(LCBv6 학습 후, holdout 평균): base 43.5 / GRPO 41.8 / **SDPO 42.4** / 자기 교사 성공 샘플 SFT 41.4. 역시 **0은 아니고** "최선의 트레이드오프"다
- 이득은 모델이 클수록 커진다("회고 능력은 규모에서 창발"). 약한 모델(Qwen3-0.6B)에서는 GRPO 혼합이 필요하다
- GRPO 대비 스텝당 오버헤드 +5.8%(코드 환경 포함 시 +17.1%)

**두 논문을 합쳐 보면**: **"in-context로 똑똑해진 자기 자신이 온폴리시 교사가 된다"는 하나의 원리**를, 데모가 있을 때(SDFT)와 환경 피드백이 있을 때(SDPO) 각각 적용한 것이다.

---

## 9. 반년 뒤의 지형 — 후속 연구가 밝힌 것

Semantic Scholar 기준으로 SDFT를 인용한 논문은 약 195편이다(2026-09-30). 대부분은 수학 추론용 "온폴리시 자기 증류(OPSD)"를 다루고, SDFT의 지속학습 설정보다는 SDPO·OPSD 쪽을 잇는다. **SDFT 자체의 벤치마크를 독립적으로 충실히 재현한 연구는 찾지 못했다.**

### 9.1 지지하는 쪽

| 논문 | 내용 |
|---|---|
| **A Quantitative Characterization of Forgetting in Post-Training** ([2603.12163](https://arxiv.org/abs/2603.12163)) | 두 모드 가우시안 혼합 모델에서 forward KL(SFT)은 옛 모드 질량을 0으로 붕괴시키고, reverse KL은 질량을 보존한다. SDFT를 "데모 앵커에 끌리는 EMA 교사로의 reverse KL 추적"으로 모델링해, **앵커 강도 $\lambda>0$이면 질량 망각이 없고 드리프트가 유한**함을 증명한다. 단 데모가 정확해야 하고, $\lambda=0$(순수 EMA 자기 교사)이면 보장이 사라진다. 역시 reverse KL 기준 |
| **When Does Continual Learning Require Learning** ([2607.07847](https://arxiv.org/abs/2607.07847)) | Qwen3-8B 순차 벤치마크에서 GEPA, ACE, SFT, SDFT, GRPO, SDPO, Cartridges, TTT를 비교. SDFT가 **세 최종 단계 점수가 모두 zero-shot 이상인 유일한 방법**이었다. 시간 드리프트에서도 과거 정확도를 유지하며 미래 정확도를 올렸다. 단 SDFT를 변형해 썼다(교사 = 이전 단계 모델) |
| **Thinking Machines "On-Policy Distillation"** (블로그, 2025.10) | 내부 문서로 mid-training하면 Qwen3-8B의 IF-eval이 85% → 45%(문서 100%) / 79%(70%)로 떨어진다. 이전 모델을 교사로 온폴리시 증류하면 **83%로 회복**되고 내부 QA는 41%를 유지한다. 자기 샘플로 하는 SFT조차 유한 배치 잡음 때문에 결국 드리프트한다는 관찰이 SDFT 동기와 맞닿는다 |

### 9.2 비판하는 쪽

| 약점 | 근거 | SDFT에 직접 해당? |
|---|---|---|
| **낡은 사실 갱신에 약함** | Harrington et al.(2607.07847) TempWiki: 바뀌지 않은 사실의 F1이 슬라이스를 거듭하며 ~0.30 → ~0.15로 떨어진다. *"새 슬라이스를 자신의 이전 믿음에 맞춰 보간하려다 사실을 오염시킨다."* 지식 갱신은 GRPO가 최선이었다 | ○ (변형 SDFT) |
| **이동 교사로 반복하면 누적 드리프트** | O'Neill(2607.11020): 문맥에 사실을 넣은 원래 모델을 교사로 한 온라인 증류는 1회 기록에서 최고(strict 77~78%)였다. 하지만 **자기 누적 모델을 교사로 20회 반복하면 최악**이 된다(능력 −28점, 보존 11%). **원래 모델을 고정 교사로** 쓰면 능력 +2점, 보존 54%. 능력 손상과 원래 모델로부터의 KL의 상관은 ρ=0.946 | △ (SDFT 유사 변형) |
| **밀도 높은 토큰 매칭 ≠ RL의 보수성** | Denser ≠ Better(2607.01763): 순차 Math→Science→Tool→Code에서 SDPO는 빨리 특화되지만 **GRPO보다 더 잊고 붕괴**하기도 한다. 교사-학생 자기강화 루프가 포맷 아티팩트를 증폭한다. *"온폴리시 데이터만으로는 지속학습에 불충분하다."* | × (SDPO) |
| **다양성 붕괴** | Nicolicioiu et al.(2606.26091): 최적 자기 증류 정책은 base를 롤아웃-데모 간 조건부 상호정보량으로 기울여 **기존 확률 격차를 증폭**한다. pass@1은 RL 수준이지만 pass@k가 평탄해진다 | △ (SDFT도 범위로 명시) |
| **콜드 스타트** | ReDraft(2609.16639): 정책이 과제를 거의 못 풀 때 자기 증류는 신호가 약하다. OPSD +19.3 vs SFT +52.9 vs ReDraft +56.9(이전 과제 손실 6.2 / 16.6 / 1.5) | △ (OPSD로 대표, SDFT 미평가) |
| **특권 정보 편향·불확실성 억제** | 2608.04794, 2603.24472: 교사가 특정 참조 궤적으로 끌리고, "잘 모르겠다" 같은 인식적 표현을 억제해 OOD에서 최대 40% 하락 | × (OPSD/SDPO, 수학) |
| **조건 문맥이 정말 중요한가** | 다른 문제의 풀이를 줘도 비슷하다(2608.09228), 참조 없는 증류가 이득의 상당 부분을 설명한다(2609.20612) | × (수학 추론) |

**요약**: 비판 대부분은 SDPO와 OPSD(수학)에서 나왔다. SDFT의 데모 기반 스킬 학습 설정에 그대로 옮기기는 어렵다. 다만 **"낡은 사실 갱신"과 "이동 교사 반복"** 두 가지는 SDFT 계열에 직접 해당하고, 지속학습 용도에서 가장 중요한 약점이다.

> 참고: 2604.15794(Liu et al.)는 자기 방법을 "SDFT (Shenfeld et al., 2026)"라 부르지만, 실제로는 과거 체크포인트를 교사로 한 성능 회복용 KL이다. 데모 조건 온폴리시 자기 증류가 아니므로 재현 사례로 인용하면 안 된다.

---

## 10. 실무 가이드 — 언제 SDFT를 쓸까

| 쓰기 좋은 경우 | 피해야 할 경우 |
|---|---|
| 모델이 **7B 이상**이고 ICL이 강함 | 3B 이하 소형 모델 |
| **데모가 있고 보상이 없음** (전문가 풀이, 도구 호출 예시) | 검증 가능한 보상이 있음 → SDPO나 GRPO 검토 |
| 새 스킬·포맷·도메인 절차를 **기존 능력 손실 없이** 추가 | base와 **크게 다른 행동**이 필요함 (비추론 → 추론 전환) |
| **답만 있고 풀이가 없는** 데이터로 추론 모델 개선 | 모델이 과제를 **거의 못 푸는** 콜드 스타트 |
| 새 사실을 **다른 질문에도 쓸 수 있게** 넣기 (OOD 98) | **이미 아는 사실을 바꾸는** 갱신 (TempWiki 실패) |
| 여러 스킬을 순차 누적 | 출력 **다양성**이 중요한 탐색형 과제 |

**권장 설정**(논문과 저장소 기본값)
- 손실은 forward KL(`alpha=0.0`), 롤아웃은 프롬프트당 1개, 해석적 토큰별 KL
- EMA 교사 $\alpha=0.01$, 매 스텝 동기화
- 앞 3토큰 손실 마스킹
- 반복 적용할 때는 O'Neill의 결과를 따라 **원래 모델 쪽으로 교사를 고정하거나 앵커링**
- 예산: SFT의 FLOPs 2.5배, 시간 4배

---

## 11. 메모리 레이어와 엮으면

[이전 리뷰](../memory-layers-sparse-finetuning-review/)의 SMF 계열과 SDFT는 **약점이 정확히 엇갈린다.**

| | Sparse Memory Finetuning | SDFT |
|---|---|---|
| 강점 | 사실을 **격리된 슬롯**에 기록. 감사, 롤백, 삭제 가능 | 스킬·포맷을 **더 배우면서** 덜 잊음 |
| 약점 | 덜 배움. 사실에서만 검증됨 | 낡은 **사실 갱신**에 약함. 반복하면 드리프트. 변화가 가중치 전체에 흩어짐 |

그래서 분업이 자연스럽다.

```
  바뀌는 사실 · 민감한 지식   ──▶  메모리 슬롯 (SMF)        : 갱신·롤백·삭제 가능
  스킬 · 포맷 · 도메인 절차   ──▶  SDFT (온폴리시 증류)     : 기존 능력 보존
  안정된 지식의 공고화        ──▶  SDFT, 교사 = 슬롯을 켠 원래 모델, 학생 = 슬롯을 끈 모델
```

공고화 단계에서 O'Neill의 경고가 설계 규칙이 된다. **교사를 누적 모델로 두면 반복할수록 무너진다.** 교사는 원래 모델에 문맥이나 슬롯을 켠 것으로 고정한다. 또 Harrington의 결과에 따르면 **자주 바뀌는 사실은 공고화하지 말고 슬롯에 남겨야** 한다. SDFT가 가장 약한 "사실 갱신"을 슬롯이 맡는 구조다.

아직 이 조합을 실험한 논문은 없다(SDFT 인용 논문 중 SDFT를 SMF나 LoRA 부분공간 방법과 **결합**한 사례를 찾지 못했다).

---

## 12. 한 장 요약

```
  문제:   SFT는 오프폴리시 → base에서 멀리 감 → 잊는다 (RL's Razor: 망각 ∝ KL)
          RL은 온폴리시라 덜 잊지만 보상이 필요하다
              │
  방법:   교사 = 같은 모델 + 데모(in-context), EMA 가중치
          학생 샘플 위에서 토큰별로 교사를 따라감
          ★ 이론은 reverse KL, 실제 결과는 전부 forward KL (v2·README 정정)
              │
  이론:   신뢰 영역 RL의 역산 → 암묵적 보상 r_t = log π(y_t|c)/π_k(y_t)
          교사 100% 정답 · base와 KL 0.68 (SFT 1.26)
              │
  결과:   Qwen2.5-7B, 세 과제 모두 새 과제 ↑ + 기존 능력 평균 64.5~65.4 (base 65.5, SFT 53.4~60.2)
          지식 주입 OOD 98 (SFT 80, Oracle RAG 100)
          답만 있는 데이터로 Olmo-Think 31.2 → 43.7 (SFT는 23.5로 하락)
          규모: 3B −3.3 / 7B +4.0 / 14B +6.9
              │
  비용:   FLOPs 2.5배, wall-clock 4배
              │
  약점:   ICL 필요 · 큰 행동 변화 불가 · RL/리플레이/KL-SFT baseline 없음
          후속 연구: 낡은 사실 갱신 실패, 이동 교사 반복 시 붕괴,
                   (SDPO/OPSD에서) 다양성 붕괴 · 콜드 스타트 약함
              │
  형제:   SDPO — 데모 대신 환경 피드백, GRPO의 advantage만 교체
  결합:   사실은 슬롯(SMF), 스킬은 SDFT, 공고화는 고정 교사 SDFT
```

---

## 참고

- 논문: [arXiv:2601.19897](https://arxiv.org/abs/2601.19897) · [코드](https://github.com/idanshen/Self-Distillation) · [프로젝트 페이지](https://self-distillation.github.io/SDFT.html)
- 선행: [RL's Razor (2509.04259)](https://arxiv.org/abs/2509.04259) · [GKD (2306.13649)](https://arxiv.org/abs/2306.13649) · [Learning by Distilling Context (2209.15189)](https://arxiv.org/abs/2209.15189) · Thinking Machines, *On-Policy Distillation* (2025.10)
- 형제: [SDPO — Reinforcement Learning via Self-Distillation (2601.20802)](https://arxiv.org/abs/2601.20802) · [코드](https://github.com/lasgroup/SDPO)
- 후속·비판: [2603.12163](https://arxiv.org/abs/2603.12163) · [2607.07847](https://arxiv.org/abs/2607.07847) · [2607.11020](https://arxiv.org/abs/2607.11020) · [2607.01763](https://arxiv.org/abs/2607.01763) · [2606.26091](https://arxiv.org/abs/2606.26091) · [2609.16639](https://arxiv.org/abs/2609.16639) · [2608.04794](https://arxiv.org/abs/2608.04794) · [2603.24472](https://arxiv.org/abs/2603.24472)

## 관련 블로그 포스트

- [저망각 파인튜닝 연구 동향 서베이 (2026)](../low-forgetting-finetuning-survey-2026/) — SDFT를 "온폴리시 목적" 계열로 놓은 지형도. 그 글의 "reverse-KL로 맞춘다" 서술은 §3.3의 정정을 참고
- [메모리 레이어로 잊지 않고 배우기](../memory-layers-sparse-finetuning-review/) — §11 결합의 다른 쪽 절반
- [DPO 리뷰](../dpo-review/) — reverse KL 목적과 암묵적 보상의 배경
