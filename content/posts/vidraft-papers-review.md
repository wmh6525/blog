---
title: "[논문 리뷰] VIDraft 논문 6편 묶어 읽기 — 인과 누설 감사·Aether·Darwin 병합·FINAL Bench"
date: 2026-10-08
tags: ["논문리뷰", "VIDraft", "모델병합", "하이브리드아키텍처", "Mamba", "인과성", "벤치마크", "메타인지", "LLM"]
categories: ["ML/AI"]
summary: "국내 AI 스타트업 VIDraft(비드래프트)가 2026년에 공개한 논문 6편을 묶어 읽었다. 하이브리드 모델의 미래 정보 누설을 층 단위로 찾아내는 감사 기법(The Mask Is Not the Model)이 가장 탄탄하다. HF transformers의 Zamba2·Nemotron-H 순수 PyTorch 경로에서 실제 축 오류를 찾아냈다. Latin square로 7가지 믹서를 배치한 Aether-7B는 가중치·코드·로그를 모두 공개했지만, 논문이 'Mamba식 선형 재귀'라고 설명한 층이 공개 코드에서는 일반 softmax 어텐션이다. 학습 없는 진화적 병합 Darwin은 GPQA Diamond 86.9%를 보고하지만, 테스트셋으로 후보를 고르고 '공식 순위'라는 리더보드 항목이 자체 보고(verified:false)다. 메타인지 벤치마크 FINAL Bench는 LLM 판정만으로 채점하며 과제 수 표기가 본문 안에서 엇갈린다. 6편 모두 동료 심사 전이다."
math: true
toc: true
draft: false
---

> **이 글의 범위와 확인 방법**
> - VIDraft 소속 저자가 쓴 논문 6편(arXiv 5편, SSRN 1편)을 다룬다. VIDraft 웹사이트가 스스로 나열한 논문 목록과 같다.
> - 6편 모두 PDF 본문을 읽었다. 비판한 부분은 **2026-10-08 기준으로 원자료를 직접 확인**했다. 확인한 원자료는 공개 코드(Hugging Face), GitHub API(이슈·PR 상태), Hugging Face 리더보드 데이터다. 확인 방법은 각 항목에 적었다.
> - 6편 모두 **동료 심사 전 프리프린트**다. Darwin 논문은 "NeurIPS 2026 투고"라고만 적혀 있다.

## 0. VIDraft는 어떤 회사인가

VIDraft(비드래프트, VIDRAFT Inc.)는 서울의 AI 스타트업이다. 대표는 김민식이다. 스스로를 "Pre-AGI 모델 개발사", "AI 파운드리"로 소개한다. 주요 제품은 다음과 같다([vidraft.net](https://vidraft.net/index.html?lang=en), [전자신문](https://www.etnews.com/20260904000247)).
- **Darwin**: 학습 없는 모델 병합
- **AETHER**: 자체 오픈 파운데이션 모델
- **AX-RAY**: AI 안전성·인과 누설 감사
- **POCKET**: 온디바이스 모델
- **MARL**: 환각을 줄인다고 소개하는 추론 미들웨어
- **FINAL Bench**: 메타인지 벤치마크

Hugging Face에서는 "VIDraft"와 "FINAL-Bench" 두 조직으로 모델을 공개한다. 논문 5편의 arXiv 저자 목록은 거의 같다(김태봉, 홍영식, 김민식, 최선영, 장재원, 김민서 등). 소수의 같은 팀이 모든 논문을 썼다.

---

## 1. 한눈에

| 논문 | 공개 | 주제 | 한 줄 | 이 글의 평가 |
|---|---|---|---|---|
| [The Mask Is Not the Model](https://arxiv.org/abs/2608.22876) | 2026-08-24 | 인과 누설 감사 | 미래 토큰 정보가 새는 **층**을 정확히 찾는다. 실제 버그 발견 | **탄탄함** — 대조군, 정직한 한계 |
| [Placement Is Free, Composition Is Not](https://arxiv.org/abs/2609.20269) | 2026-07-29 (v1) | Aether 아키텍처 | 7가지 믹서를 Latin square로 배치한 6.59B MoE | **혼재** — 공개는 충실, 설명과 코드 불일치 |
| [Darwin Family](https://arxiv.org/abs/2605.14386) | 2026-05-14 | 학습 없는 병합 | 진화 탐색으로 층별 병합 비율을 정해 GPQA 86.9% | **약함** — 테스트셋 선택, 자체 보고 순위 |
| [FINAL Bench](https://doi.org/10.2139/ssrn.6280258) | 2026-04-13 (SSRN) | 메타인지 벤치마크 | 오류를 스스로 찾고 고치는 능력을 잰다 | **약함** — LLM 판정만, 과제 수 불일치 |
| [Quantum Cryptanalysis on IBM Hardware](https://arxiv.org/abs/2607.18340) | 2026-07-20 | 양자 암호 해독 시연 | 장난감 암호에 Simon·Grover 알고리즘 | ML 밖. 핵심 기법 비공개 |
| [Suzuki's Weil-Quadratic-Form Operator](https://arxiv.org/abs/2607.24830) | 2026-07-23 | 수론 수치 실험 | 리만 가설 관련 연산자의 수치 이산화 | ML 밖. 증명이 아니라고 명시 |

이 글은 ML 논문 네 편을 자세히 보고, 과학 논문 두 편은 짧게 다룬다.

---

## 2. The Mask Is Not the Model — 미래 정보가 새는 층을 찾는다

**The Mask Is Not the Model: Auditing Prefix Invariance in Attention, State-Space, and Hybrid Sequence Models** ([2608.22876](https://arxiv.org/abs/2608.22876), v3 2026-09-28)

### 문제: 마스크가 맞아도 미래가 샐 수 있다

언어 모델은 왼쪽에서 오른쪽으로 한 토큰씩 생성한다. 그래서 위치 $t$의 출력은 **$t$ 이후 토큰에 영향을 받으면 안 된다**(인과성). 트랜스포머에서는 어텐션 마스크가 이를 보장한다.

그런데 요즘 모델은 어텐션만 쓰지 않는다. Mamba 같은 상태공간모델(SSM), 순환 구조, 합성곱을 섞은 **하이브리드 모델**이 많다. 이런 층은 어텐션 마스크와 무관하게 계산된다. 특히 긴 시퀀스를 **청크(chunk)**로 잘라 병렬 계산하는 구현에서는, 청크 사이를 잇는 코드에 실수가 있으면 미래 정보가 섞여 들어갈 수 있다. 마스크만 검사해서는 이를 잡을 수 없다는 것이 제목의 뜻이다.

### 방법: 마지막 토큰만 다른 두 입력을 비교한다

방법은 단순하다.
1. 마지막 토큰 하나만 다른 무작위 시퀀스 두 개를 만든다.
2. 캐시를 끄고 float32로 두 번 순전파한다. 모든 층에 훅을 건다.
3. 마지막 이전 모든 위치에서, 층마다 두 출력의 차이를 잰다.
4. 차이가 임계값(1e-6)을 처음 넘는 층이 **누설의 출발점**이다.

인과적인 모델이라면 이전 위치의 출력은 **비트 단위로 정확히 같아야** 한다. 그래서 정상 모델에서는 차이가 정확히 0이고, 임계값을 어떻게 잡든 결과가 거의 바뀌지 않는다.

### 결과

- **주입한 결함 찾기**: 공개 체크포인트 8개에 결함 패턴 8종을 깊이 3곳에 심었다(192회). 제안 방법은 **192/192**를 정확한 층까지 찾았다. 정적 마스크 검사는 **0/192**였다.
- **다른 방법과 비교**(96회):

| 방법 | 탐지 | 위치 특정 |
|---|---|---|
| 최종 출력(logits)만 비교 | 96/96 | 0/96 |
| 뒷부분을 섞은 perplexity | 71/96 | 0/96 |
| 그래디언트 기반 | 96/96 | **96/96** |
| 제안 방법 | 96/96 | **96/96** |

- **실제 버그**: 시퀀스가 청크 크기보다 길면 Zamba2-1.2B는 위치 256부터, Nemotron-H-8B는 128부터 미래 정보가 샜다. 저자들은 HF transformers의 **순수 PyTorch 청크 스캔 코드**에서 원인을 찾았다. 청크 사이 상태를 이어 붙일 때 입력 청크 축이 아니라 **출력 청크 축으로 합산**하고 있었다. 두 파일에 같은 3줄이 있었고, 2줄 수정으로 누설이 정확히 0이 됐다.

### 업스트림에서는 어떻게 됐나 (GitHub API로 확인)

- 저자들은 2026-07-22 transformers에 이슈 [#47475](https://github.com/huggingface/transformers/issues/47475)와 PR [#47476](https://github.com/huggingface/transformers/pull/47476)을 올렸다.
- 메인테이너는 약 90분 뒤 둘 다 **병합 없이 닫았다.** 기술적 코멘트 없이 "Code agent slop" 라벨이 붙었다.
- 다만 하루 전(07-21)에 다른 메인테이너가 연 리팩터링 PR [#47452](https://github.com/huggingface/transformers/pull/47452)("Refactor all linear attention models…")가 07-24에 병합됐다. 이 PR로 해당 코드가 참조 구현과 같은 축 처리로 바뀌었다. 이 PR은 인과성 문제를 언급하지 않는다.
- 정리하면 **문제의 코드 패턴은 실제로 있었고 지금은 사라졌다.** 다만 VIDraft의 보고가 "메인테이너가 확인한 버그"였다고 말하기는 어렵다. 논문 v3(09-28)에는 이슈가 닫혔다는 사실이나 업스트림 코드가 바뀌었다는 사실이 적혀 있지 않다.

### 평가

**강점.**
- 한계를 솔직하게 적었다. "탐지 자체는 logits만 비교하는 방법보다 나을 게 없다. 기여는 위치 특정뿐이다", "그래디언트 기반 방법도 위치를 똑같이 찾는다"고 직접 썼다.
- 스스로 저지른 실수에서 교훈을 뽑았다. Falcon-H1을 세 번 불러왔더니 어떤 입력에도 차이가 0이었는데, 모델이 제대로 동작하지 않는 상태였다. 그래서 "깨끗하다"는 판정에는 **같은 체크포인트에 대한 양성 대조군**이 필요하다고 적었다. 또 테스트 시퀀스는 **청크·윈도 크기보다 길어야** 한다.
- 실제 코드에서 검증 가능한 결함을 찾았다.

**한계.**
- 방법 자체는 표준적인 섭동 테스트다.
- 누설한 두 모델이 같은 코드 계보라서, 이 결과는 "이런 버그가 존재한다"는 증명이지 빈도 추정이 아니다.
- 퓨즈드 CUDA 커널 경로는 검사하지 않았다.

**하이브리드 모델을 직접 구현하거나 고치는 사람에게 실용적인 체크리스트**로 읽을 만하다.

---

## 3. Aether — Latin square로 7가지 믹서를 배치하다

**Placement Is Free, Composition Is Not: The Latin Square as a Provably-Balanced Construction for Heterogeneous Sequence-Mixer Stacks** ([2609.20269](https://arxiv.org/abs/2609.20269), v3 2026-10-01)

### 모델

Aether-7B-5Attn([FINAL-Bench/Aether-7B-5Attn](https://huggingface.co/FINAL-Bench/Aether-7B-5Attn))의 기술 보고서다.
- 6.59B 파라미터 MoE(활성 약 2.98B), 49층. 전문가 25개 중 7개를 고르고 공유 전문가 1개를 둔다.
- 층마다 시퀀스 믹서가 다르다. full, sliding, differential, linear, NSA, compress, hybrid 7종이다.
- 배치는 **7×7 Latin square**를 따른다. Latin square는 각 기호가 모든 행과 열에 정확히 한 번씩 나오는 표다(스도쿠의 규칙 일부와 같다). 층을 7개씩 묶어 행으로 보면, 모든 믹서가 각 묶음에 한 번씩, 그리고 묶음 안의 각 위치에 한 번씩 나온다.
- 학습: Qwen3 토크나이저, 고유 토큰 42.1B(반복 포함 약 144B), B200 16장으로 약 46일.
- **가중치, 코드, 데이터 레시피, 로그를 모두 공개했다.** 국내 팀의 파운데이션 모델 공개로서 의미가 있다.

### 소거 실험

700.9M 프록시 모델에서 믹서 4종(full F, sliding S, differential G, Mamba-2 M)을 16층 4×4 Latin square로 배치하고, 1,500스텝을 시드 8개로 학습했다.

| 배치 | 검증 손실 변화 (latin 5.286 기준) |
|---|---|
| 균형 잡힌 주기 반복 (periodic) | +0.16% (잡음 수준) |
| 같은 믹서끼리 연속 블록 | +0.59% |
| 전부 full attention (동질) | +1.68% |
| Mamba-2만 빼기 | **+2.14%** |
| 어텐션 변형 하나 빼기 | 잡음 수준 |

1.514B(시드 3개)에서는 동질 스택 +2.63%, Mamba-2 제거 +3.20%였다. 결론은 "정확한 배치는 거의 상관없고, **여러 종류가 깊이 전체에 고르게 섞여 있는지**가 중요하다. 그중 결정적인 것은 SSM 계열이다"이다.

### 확인한 문제: "Mamba식" 층이 코드에서는 softmax 어텐션이다

논문은 플래그십의 `linear` 층을 **"Mamba식 선형 재귀 믹서, 상태공간 계열"**이라고 설명한다(§3.1). 그리고 소거 실험과 플래그십을 이렇게 연결한다.

> "플래그십은 Mamba식 선형 믹서로 하중을 받는(load-bearing) 계열을 제공한다. 따라서 플래그십은 이 소거 실험이 필수라고 밝힌 메커니즘 계열을 빠뜨리지 않고 포함한다."

그런데 Hugging Face에 공개된 모델 코드(`modeling_aether_v2_7way.py`, 루트와 `aether_pkg/` 두 사본, 2026-10-08 내려받음)의 `LinearAttention` 클래스는 다음과 같다.

```python
class LinearAttention(nn.Module):
    """Linear attention (Mamba/RWKV-inspired) for long-context efficiency."""
    ...
    def forward(self, hidden_states, ...):
        q = self.q_proj(hidden_states).view(...).transpose(1, 2)
        k = self.k_proj(hidden_states).view(...).transpose(1, 2)
        v = self.v_proj(hidden_states).view(...).transpose(1, 2)
        g = self.gate(hidden_states).view(...).sigmoid()
        ...
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=(q_len > 1))
        out = out * g
        out = self.norm(out).reshape(bsz, q_len, -1)
        return self.o_proj(out), past_key_value
```

`F.scaled_dot_product_attention(..., is_causal=True)`는 **일반 causal softmax 어텐션**이다. 여기에 시그모이드 게이트와 RMSNorm을 붙였을 뿐, 재귀나 상태공간 계산, 선형 어텐션(커널 트릭)이 없다. 이름과 주석만 "Mamba-style"이다.

**왜 중요한가.** 논문의 소거 실험은 "어텐션 변형끼리는 서로 바꿔도 되고, SSM 계열이 빠지면 손해가 크다"고 말한다. 그 논리를 따르면, 공개 코드 기준의 플래그십은 **결정적이라고 밝힌 바로 그 계열이 없는** 모델이다. 논문도 "플래그십의 linear 구현이 프록시의 Mamba-2만큼 기여하는지는 다음에 확인할 일"이라고 적어 두었다. 하지만 구현이 softmax 어텐션이라는 사실은 적혀 있지 않다.

같은 팀의 Mask 논문도 Aether를 "7개 연속 층마다 각 타입이 한 번씩 나온다"고 설명한다. 이는 Aether 논문 자신이 "Latin square로는 불가능하다"고 증명한 성질이라 두 논문의 서술이 서로 맞지 않는다.

### 그 밖의 한계

- **소거 실험이 매우 짧다.** 1,500스텝에 검증 손실 약 5.3 nats(perplexity 약 200)이다. 수렴한 아키텍처의 품질이 아니라 **학습 초반의 속도**를 잰 것에 가깝다.
- **제목의 Latin square가 이기지 않는다.** latin과 단순 주기 반복의 차이(+0.16%)는 잡음 수준이다. 실험이 보여주는 것은 "고르게 섞어라"이지 "Latin square여야 한다"가 아니다. 이 점은 저자들도 인정한다("배치는 공짜, 구성은 아니다"). "SSM이 중요하다"는 결과는 기존 하이브리드 모델 연구들과 같은 방향이다.
- **플래그십 성능이 낮다.** lm-eval 0-shot(과제당 600문항)에서 ARC-Challenge 22.2(acc_norm 25.8), WinoGrande 51.8, BoolQ 54.3, KoBEST WiC 48.8, BoolQ 47.8로 여러 과제가 **찍기 수준**이다. 저자들도 순위 경쟁이 목적이 아니라고 밝힌다. 그래도 같은 규모의 동질 모델이나 비슷한 크기의 공개 모델과 비교하지 않아서, 이 아키텍처가 실제 규모에서 이득인지는 알 수 없다.
- **공개된 학습 코드는 마지막 단계뿐이다.** 런처는 공개되지 않은 455,000스텝 체크포인트에서 재개해 학습률 5e-5 → 5e-6으로 어닐링한다.

**좋은 점.** 시드 수를 보고했고, 판단 규칙을 미리 정해 두었고, 시드 2개 실험에서 나온 거짓 양성을 스스로 철회했다.

---

## 4. Darwin — 학습 없이 체크포인트를 섞어 추론을 올린다

**Darwin Family: MRI-Trust-Weighted Evolutionary Merging for Training-Free Scaling of Language-Model Reasoning** ([2605.14386](https://arxiv.org/abs/2605.14386))

### 모델 병합이란

같은 base 모델에서 출발해 서로 다르게 파인튜닝한 모델 두 개가 있다고 하자. **모델 병합**은 추가 학습 없이 두 모델의 가중치를 섞어 양쪽 능력을 모두 갖게 하려는 방법이다. 가장 단순한 방법은 평균이고, 많이 쓰는 방법으로 TIES, DARE(변화량 일부를 무작위로 버리고 나머지를 키움), SLERP 등이 있다. Sakana AI는 병합 비율을 **진화 알고리즘으로 탐색**하는 방법을 제안했다.

### Darwin의 방법

- "아버지"는 base 모델(예: Qwen3.5-27B), "어머니"는 같은 base의 파인튜닝 모델이다(예: Claude Opus 4.6 추론 흔적으로 증류한 커뮤니티 모델).
- 텐서마다 $T = \text{base} + (1-r)\,\Delta_A + r\,\Delta_B$로 섞는다. $\Delta$는 각 부모가 base에서 달라진 양이다.
- 텐서별 비율 $r$은 두 가지를 섞어 정한다.
  - **MRI 점수**: 텐서의 정적 통계(엔트로피, 분산, 노름)와, 추론 프롬프트와 일반 프롬프트에서 그 층이 얼마나 다르게 활성화되는지를 합친 진단 점수. 프롬프트 123개를 쓴다(약 절반은 한국어).
  - **게놈**: 전역 비율, 어텐션/FFN/임베딩 비율, 블록별 비율 등 14개 숫자. CMA-ES 진화 알고리즘(개체 50)으로 탐색한다.
  - $r = \tau\, r_{\text{MRI}} + (1-\tau)\, r_{\text{genome}}$. $\tau$도 게놈의 일부로 함께 진화한다.
- 실제 병합 연산은 DARE-TIES다.
- 모델당 H100 1~2장으로 약 1~5시간이 든다.

### 결과

| Darwin-27B-Opus | 아버지 | 어머니 | 단순 평균/SLERP | Darwin |
|---|---|---|---|---|
| GPQA Diamond | 0.855 | 0.862 | 0.861 | **0.869** |
| ARC-Challenge | 0.710 | 0.740 | 0.750 | 0.779 |
| MMLU | 0.754 | **0.782** | 0.768 | 0.776 |

- $\tau$ 소거(GPQA): 게놈만 84.4, MRI만 85.6, $\tau$=0.7 고정 86.0, 진화한 $\tau$ 86.9.
- 계열 표: Darwin-4B-David 85.0, Darwin-31B-Opus 85.9, Darwin-35B-A3B-Opus 90.0.

### 확인한 문제

**① 테스트셋으로 고르고 테스트셋으로 보고한다.** 탐색의 2단계는 후보를 "추론 벤치마크에서 직접 평가해" 고른다. 그리고 결과의 핵심 수치도 같은 GPQA다. 별도의 선택용 데이터셋이 없다. GPQA Diamond는 198문항이라 어머니 대비 +0.7%p는 **약 1.4문항**이다. 테스트셋에서 여러 후보 중 최고를 고르면 이 정도 차이는 선택 효과만으로도 생길 수 있다.

**② 약속한 기준선이 표에 없다.** 본문은 "TIES식 병합, 진단 없는 진화적 병합(Sakana식)과 비교한다"고 쓴다. 하지만 표에 있는 기준선은 "단순 평균/SLERP" 한 열뿐이다. 가장 직접적인 비교 대상인 Sakana식 진화 병합의 수치가 없다.

**③ "공식 6위"는 자체 보고다** (Hugging Face 리더보드 데이터로 확인).
- 논문은 "GPQA Diamond 리더보드 **공식** 6위(1,252개 모델 중)", "독립적으로 검증됐다"고 쓴다.
- Hugging Face의 GPQA 데이터셋 리더보드는 모델 카드에 올린 평가 결과를 모아 보여준다. Darwin-27B-Opus 항목의 출처는 **"Model Card"**(모델 저장소의 `.eval_results/gpqa_diamond.yaml`)이고, `verified: false`다. 2026-10-08 기준 이 리더보드의 111개 항목 모두 `verified: false`였고, Darwin-27B-Opus는 **31위**(86.9)였다.
- 즉 이 순위는 **스스로 보고한 점수의 순위**다. "공식", "독립 검증"이라는 표현은 근거가 없다. 국내 언론의 "공식 리더보드 1위" 보도들도 같은 성격의 자체 보고 점수에 기반한다.

**④ 4B 모델의 85.0은 다수결 점수다.** Darwin-4B-David의 GPQA 85.0은 부록에 "maj@8"로 표기돼 있다. 8번 생성해 다수결한 점수라 다른 모델의 단일 생성 점수와 바로 비교할 수 없다. 4B 모델이 GPQA Diamond 85%라면 매우 이례적인 결과인데, 독립 재현이 없다.

**⑤ 코드를 찾지 못했다.** 논문은 "V6 코드베이스(약 13,771줄)를 Apache 2.0으로 공개한다"고 쓴다. 조사 시점에 공개 저장소를 찾지 못했고, 논문이 가리키는 Space(`VIDraft/DARWIN-Evolution`)는 401(접근 불가)을 반환했다. 가중치는 공개돼 있지만 일부는 접근 신청이 필요하다.

**⑥ 작은 불일치.** 표 1의 "전체 평균"이 각 열의 아홉 개 값 평균과 맞지 않는다. Darwin 열 평균은 0.796인데 0.786으로 적혀 있다. 아버지·어머니 열도 약 0.01씩 낮게 적혀 있어 순위는 바뀌지 않는다. 본문에는 존재하지 않는 절(§3.3.2)과 표(Table B.1)에 대한 참조도 있다.

**정리.** "MRI 진단 + 진화 탐색"이라는 아이디어 자체는 시도해 볼 만하다. 하지만 지금 근거로는 **Darwin이 단순 병합이나 Sakana식 진화 병합보다 낫다는 주장이 성립하지 않는다.** 독립된 선택용 데이터, 빠진 기준선, 다중 시드 신뢰구간이 필요하다.

---

## 5. FINAL Bench — 모델은 자기 오류를 고칠 수 있나

**FINAL Bench: Measuring Functional Metacognitive Reasoning in Large Language Models** (SSRN [10.2139/ssrn.6280258](https://doi.org/10.2139/ssrn.6280258), 2026-04-13)

SSRN 페이지는 자동 접근이 막혀 있어서, VIDraft가 [Hugging Face 데이터셋](https://huggingface.co/datasets/FINAL-Bench/Metacognitive)에 올린 PDF(2026-02-21 판)를 읽었다. SSRN 판과 저자 순서나 내용이 다를 수 있다.

### 설계

- 15개 분야, 3개 난이도, 8개 유형의 **함정이 숨어 있는 과제**를 직접 만들었다.
- 다섯 축으로 채점한다. 과정 품질 15%, 메타인지 정확도(MA) 20%, **오류 복구(ER) 25%**, 통합 깊이 20%, 최종 정답 20%.
- 두 조건을 비교한다. 기본(API 한 번 호출)과 MetaCog(추론 → 자기 검토 → 수정 3단계 스캐폴드).
- 채점은 **전부 LLM 판정자 앙상블**(GPT-5.2, Claude Opus 4.6, Gemini 3 Pro)이 한다.
- 모델 9개: Claude Opus 4.6, GPT-5.2, Gemini 3 Pro, DeepSeek-V3.2, GPT-OSS-120B, GLM-5 등.

### 결과

- 기본 조건 평균 61.12(Kimi K2.5 68.71로 1위, Claude Opus 4.6 56.04로 최하위). MetaCog 평균 75.17(+14.05).
- 향상분의 94.8%가 오류 복구(ER) 축에서 나왔다.
- 기본 조건에서 메타인지 정확도 0.694 대 오류 복구 0.302. "오류가 있다는 건 알지만 고치지는 못한다"는 해석이다.
- 기본 점수가 낮을수록 향상이 컸다(r = −0.777).

### 확인한 문제

- **결론이 설계에 들어 있다.** "자기 검토 후 수정하라"고 강제하는 스캐폴드를 쓰면, "오류를 찾아 고쳤는가"로 정의한 채점 축이 오르는 것은 자연스럽다. 기본 점수가 낮은 모델이 더 많이 오르는 것도 올라갈 여지가 크기 때문일 수 있다.
- **과제 수가 본문 안에서 엇갈린다**(PDF 원문으로 확인). 본문은 "100개 과제 × 9개 모델 × 2개 조건 = 1,800회 평가"라고 쓴다. 그런데 핵심 결과 그림의 제목은 "9 Models × 30 Tasks"이고, 유형별 표의 n을 더해도 30이다.
- **LLM 판정만 쓴다.** 사람과의 일치도 κ = 0.87을 보고하지만 표본 크기와 평가자 정보가 없다. 기본 조건 평가의 79.6%에서 ER이 0.25였다는 점은 채점 기준이 매우 거칠다는 신호다.
- 자체 제작 벤치마크이고, 신뢰구간이 없고, 한 번만 실행했고, 부록이 생략됐다.

"자기 검토 스캐폴드가 오류 수정을 돕는다"는 방향 자체는 그럴듯하다. 하지만 이 논문의 수치를 **모델 간 메타인지 능력 순위**로 읽기는 어렵다.

---

## 6. 과학 논문 두 편 (짧게)

**Quantum Cryptanalysis on IBM Quantum Hardware** ([2607.18340](https://arxiv.org/abs/2607.18340))
- IBM 양자 하드웨어(Heron)에서 장난감 대칭 암호(Even–Mansour, 3라운드 Feistel, CBC-MAC)에 Simon 알고리즘을, 장난감 SPN에 Grover를 돌렸다. 이전 실기 결과는 Even–Mansour N=4였다.
- 제목은 "N=4에서 N=10으로 확장"이다. 하지만 **정답 키를 1순위로 깔끔하게 찾은 것은 n=5까지**다. n=6~10은 양자 출력으로 후보를 줄인 뒤 고전 계산으로 마무리한 "하이브리드"이고, n=10에서 정답 키는 1023개 중 63위였다. 이는 고전적인 생일 경계($2^{n/2}$)와 비슷한 수준이다.
- 저자들은 깨끗한 키 복구가 "하드웨어 맞춤 회로 조정·판독 후선택 기법"에 기대고 있다고 밝혔다. 그런데 이 기법을 **IP 결정 대기를 이유로 공개하지 않았다.** 저자들은 알고리즘 자체의 재현성에는 영향이 없다고 하지만, 보고한 하드웨어 수치를 재현하려면 이 기법이 필요하다.
- 저자들도 양자 우위가 없고 AES/RSA와 무관하다고 명시한다.

**A Numerical Realization of Suzuki's Weil-Quadratic-Form Operator** ([2607.24830](https://arxiv.org/abs/2607.24830), math.GM)
- M. Suzuki가 제안한, 리만 가설과 관련된 연산자를 유한요소법으로 수치 이산화하고 고유값의 근사 법칙을 보고한다.
- 논문은 **리만 가설을 증명하거나 증명에 다가가지 않는다**고 반복해서 밝히고, 핵심 유도가 휴리스틱이라고 적는다. 이 글에서는 수학적 가치를 판단하지 않는다.

---

## 7. 묶어서 보면

**공통점.**
- 같은 6~7명이 같은 순서로 모든 논문을 썼다. 논문마다 회사 제품과 바로 연결된다. Darwin은 병합 모델 계열, Aether는 자체 파운데이션 모델, Mask 논문은 AX-RAY, FINAL Bench는 MARL·메타인지 제품이다.
- Aether와 Mask 논문은 서로를 동반 논문으로 인용한다. Mask 감사를 Aether에 적용했고, Aether 논문의 "인과 안전성" 절은 이 감사에 기댄다.
- 논문마다 "정직한 범위(honest scope)" 절과 자기 철회 기록이 있다.

**근거의 단단함은 논문마다 크게 다르다.**

| | 공개 | 통제 실험 | 주장과 근거의 일치 |
|---|---|---|---|
| Mask | 방법 재현 가능 | 양성·음성 대조군 | 높음 (업스트림 상태 서술은 빠짐) |
| Aether | 가중치·코드·로그 | 시드 8개, 사전 규칙 | 핵심 서술이 코드와 불일치 |
| Darwin | 가중치 (코드 미확인) | 테스트셋 선택 | "공식 순위" 근거 없음 |
| FINAL Bench | 데이터셋 | LLM 판정만 | 과제 수 불일치 |

**읽는 사람을 위한 정리.**
- **Mask 논문**은 하이브리드 모델을 다루는 사람이라면 체크리스트로 쓸 만하다. 시퀀스를 청크 크기보다 길게, 양성 대조군을 두고, 층마다 비교한다.
- **Aether**는 공개 범위가 넓어 직접 확인해 볼 수 있다는 것이 가장 큰 장점이다. 다만 "SSM을 포함한 하이브리드"로 쓰려면 `linear` 층이 실제로 무엇인지부터 확인해야 한다.
- **Darwin과 FINAL Bench**의 수치, 그리고 이를 인용한 "리더보드 1위" 류의 보도는 독립 검증 전까지 **자체 보고**로 읽는 것이 맞다.

국내 팀이 파운데이션 모델부터 감사 도구까지 빠르게 공개하는 것은 반가운 일이다. 그만큼 논문 서술과 공개물이 서로 맞는지, 순위 주장이 무엇에 근거하는지를 함께 확인하며 읽어야 한다.

---

## 참고

**논문**
- The Mask Is Not the Model — [arXiv:2608.22876](https://arxiv.org/abs/2608.22876)
- Placement Is Free, Composition Is Not (Aether) — [arXiv:2609.20269](https://arxiv.org/abs/2609.20269) · 모델 [FINAL-Bench/Aether-7B-5Attn](https://huggingface.co/FINAL-Bench/Aether-7B-5Attn)
- Darwin Family — [arXiv:2605.14386](https://arxiv.org/abs/2605.14386)
- FINAL Bench — [SSRN 10.2139/ssrn.6280258](https://doi.org/10.2139/ssrn.6280258) · [HF 데이터셋](https://huggingface.co/datasets/FINAL-Bench/Metacognitive)
- Quantum Cryptanalysis on IBM Quantum Hardware — [arXiv:2607.18340](https://arxiv.org/abs/2607.18340)
- Suzuki's Weil-Quadratic-Form Operator — [arXiv:2607.24830](https://arxiv.org/abs/2607.24830)

**확인에 쓴 원자료 (2026-10-08)**
- transformers 이슈 [#47475](https://github.com/huggingface/transformers/issues/47475), PR [#47476](https://github.com/huggingface/transformers/pull/47476), 리팩터링 PR [#47452](https://github.com/huggingface/transformers/pull/47452)
- Aether 모델 코드 `modeling_aether_v2_7way.py` (HF 저장소 루트와 `aether_pkg/`)
- GPQA 데이터셋 리더보드 — [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa)

## 관련 블로그 포스트

- [Mamba 계열 완전 정리](../mamba-family-complete-survey/) — 3장의 SSM·하이브리드 배경
- [Hymba 리뷰](../hymba-review/) — 어텐션과 SSM을 한 층에 섞는 하이브리드
- [순수 SSM만으로는 왜 LLM이 어려운가](../why-pure-ssm-cannot-be-llm/) — "SSM 계열이 하중을 받는다"는 3장 결과의 배경
