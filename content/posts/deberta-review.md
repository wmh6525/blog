---
title: "[논문 리뷰] DeBERTa v1/v2/v3 — 분리된 어텐션과 향상된 마스크 디코더로 BERT를 넘어서다"
date: 2026-04-01
tags: ["논문리뷰", "DeBERTa", "Transformer", "NLU", "BERT"]
categories: ["ML/AI"]
summary: "DeBERTa 모델 패밀리(v1/v2/v3)를 상세 리뷰한다. Disentangled Attention의 수식, Enhanced Mask Decoder, ELECTRA 스타일 학습(RTD), Gradient-Disentangled Embedding Sharing까지 — BERT 계열 인코더 모델의 정점을 추적한다."
math: true
toc: true
draft: false
---

## 논문 정보

| 버전 | 제목 | 학회 | 날짜 |
|------|------|------|------|
| **V1** | DeBERTa: Decoding-enhanced BERT with Disentangled Attention | ICLR 2021 | 2020.06 |
| **V2** | (V1 확장, 별도 논문 없음) | - | 2021 |
| **V3** | DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing | ICLR 2023 | 2021.11 |

- **저자**: Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen
- **소속**: Microsoft Research
- **코드**: [github.com/microsoft/DeBERTa](https://github.com/microsoft/DeBERTa)

---

## 1. DeBERTa의 핵심 아이디어

DeBERTa = **D**ecoding-**e**nhanced **BERT** with **D**isentangled **A**ttention

두 가지 핵심 혁신:
1. **Disentangled Attention**: 콘텐츠와 위치를 분리하여 어텐션 계산
2. **Enhanced Mask Decoder (EMD)**: 절대 위치 정보를 디코더 단에서 주입

---

## 2. 혁신 1: Disentangled Attention (분리된 어텐션)

### 표준 Transformer Attention

$$Q = HW_q, \quad K = HW_k, \quad V = HW_v$$

$$A = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

여기서 $H$는 토큰의 **콘텐츠**와 **위치** 정보가 혼합된 단일 벡터이다 (BERT는 토큰 임베딩 + 절대 위치 임베딩을 입력 단에서 합산).

### DeBERTa의 분리: 콘텐츠 벡터 + 위치 벡터

DeBERTa는 각 토큰을 **두 개의 분리된 벡터**로 표현한다:
- $H_i$: **콘텐츠 벡터** (이 토큰이 "무엇"인지)
- $P_{i|j}$: **상대 위치 벡터** (토큰 $i$와 $j$의 상대적 거리)

어텐션 스코어는 **네 가지 항**으로 분해된다:

$$A_{i,j} = \underbrace{H_i H_j^\top}_{\text{c2c: 콘텐츠↔콘텐츠}} + \underbrace{H_i P_{j|i}^\top}_{\text{c2p: 콘텐츠↔위치}} + \underbrace{P_{i|j} H_j^\top}_{\text{p2c: 위치↔콘텐츠}} + \underbrace{P_{i|j} P_{j|i}^\top}_{\text{p2p: 위치↔위치 (제거)}}$$

**p2p 항은 제거**한다 — 상대 위치 임베딩에서 위치↔위치 항은 추가 정보를 거의 제공하지 않기 때문.

### 최종 분리 어텐션 수식

$$\tilde{A}_{i,j} = Q_i^c {K_j^c}^\top + Q_i^c {K_{\delta(i,j)}^r}^\top + K_j^c {Q_{\delta(j,i)}^r}^\top$$

- $Q^c = HW_{q,c}$, $K^c = HW_{k,c}$, $V^c = HW_{v,c}$: **콘텐츠** 프로젝션
- $Q^r = PW_{q,r}$, $K^r = PW_{k,r}$: **상대 위치** 프로젝션
- $P$: 공유 상대 위치 임베딩 행렬 ($\mathbb{R}^{2k \times d}$, $k$=최대 상대 거리)

### 스케일링 팩터

$$H_o = \text{softmax}\left(\frac{\tilde{A}}{\sqrt{3d}}\right) V^c$$

표준 어텐션의 $\frac{1}{\sqrt{d}}$ 대신 **$\frac{1}{\sqrt{3d}}$**를 사용한다 — 3개 항의 합이므로 크기가 3배 커지기 때문.

### 상대 위치 함수

$$\delta(i,j) = \begin{cases} 0 & \text{if } i-j \leq -k \\ 2k-1 & \text{if } i-j \geq k \\ i-j+k & \text{otherwise} \end{cases}$$

$k$는 최대 상대 거리 (사전학습 시 $k=512$).

### 기존 방법과의 비교

| 방법 | 어텐션 항 | 위치 인코딩 |
|------|---------|----------|
| BERT | 합산된 단일 벡터 | 절대 위치 (입력 단) |
| Shaw et al. (2018) | c2c + c2p | 상대 위치 |
| Transformer-XL | c2c + c2p (단일 바이어스) | 상대 위치 |
| **DeBERTa** | **c2c + c2p + p2c** | **상대 위치 (분리 프로젝션)** |
| RoPE | 회전으로 인코딩 | 상대 위치 (회전 기반) |

**p2c 항의 중요성**: "단어 쌍의 어텐션 가중치는 콘텐츠뿐 아니라 상대 위치에도 의존한다."

### Ablation 결과 (Base 모델)

| 구성 | RACE | SQuAD v2.0 | MNLI-m |
|------|------|-----------|--------|
| Full DeBERTa | 71.7 | 82.5 | 86.3 |
| -c2p | 69.3 (-2.4) | 81.3 (-1.2) | 85.9 (-0.4) |
| -p2c | 69.6 (-2.1) | 80.8 (-1.7) | 86.0 (-0.3) |
| -EMD | 70.3 (-1.4) | 81.3 (-1.2) | 86.1 (-0.2) |

c2p와 p2c 모두 성능에 유의미하게 기여한다.

---

## 3. 혁신 2: Enhanced Mask Decoder (EMD)

### 왜 절대 위치가 필요한가?

예문: "a new **store** opened beside the new **mall**"

`store`와 `mall`이 마스킹되면, 둘 다 지역 문맥이 "new ___"로 동일하다. **상대 위치만으로는 둘을 구별할 수 없다** — 절대 위치 정보가 있어야 "3번째 위치"와 "8번째 위치"를 구분한다.

### EMD의 작동

BERT: 절대 위치를 **입력 단(첫 레이어)**에서 추가
DeBERTa: 절대 위치를 **디코더 단(마지막 레이어)**에서 추가

```
인코더 출력 H (상대 위치만 사용하여 학습)
    ↓ + 절대 위치 임베딩 I
[추가 Transformer 레이어 × 2 (가중치 공유)]
    ↓
MLM 예측 헤드
```

**왜 마지막에 넣는가?** 초기 레이어에서 절대 위치를 주입하면 모델이 위치 정보에 과도하게 의존할 수 있다. 대부분의 처리는 상대 위치만으로 충분하고, 최종 예측에서만 절대 위치가 필요하다.

---

## 4. DeBERTa V2: 스케일 업

V1에서의 주요 변경:

| 항목 | V1 | V2 |
|------|-----|-----|
| 어휘 | 50K GPT-2 BPE | **128K SentencePiece** |
| 상대 위치 | 선형 | **로그 버킷** (T5 스타일) |
| nGiE | 없음 | **n-gram 컨볼루션** 추가 |
| 프로젝션 공유 | 없음 | 위치/콘텐츠 프로젝션 공유 |
| 최대 규모 | 700M (XLarge) | **1.5B (XXLarge)** |

### 로그 버킷 상대 위치

가까운 거리는 고유 임베딩, 먼 거리는 로그 함수로 그룹화:

```python
log_pos = ceil(log(abs_pos / mid) / log((max_pos-1) / mid) * (mid-1)) + mid
```

### SuperGLUE 결과 (V2-XXLarge, 1.5B)

$$\text{DeBERTa 1.5B + SiFT: } \textbf{89.9} \quad \text{(인간 기준선: 89.8)}$$

**최초로 SuperGLUE에서 인간 기준선을 초과한 모델.**

| 과제 | DeBERTa | 인간 |
|------|---------|------|
| BoolQ | 90.4 | 89.0 |
| COPA | 96.8 | 100.0 |
| MultiRC (F1a) | 88.2 | 81.8 |
| ReCoRD (F1) | 94.5 | 91.7 |
| **평균** | **89.9** | **89.8** |

참고: T5-11B는 89.3 — DeBERTa 1.5B가 **7배 작은 모델로** T5를 능가.

---

## 5. DeBERTa V3: ELECTRA 스타일 학습

### MLM의 한계

BERT의 MLM은 입력의 ~15%만 마스킹하여 예측 → 토큰의 **85%는 학습 신호를 제공하지 않는다**.

### Replaced Token Detection (RTD)

ELECTRA 방식: **100%의 토큰**에서 학습 신호를 얻는다 ("6.7배 효율 향상").

```
Generator (소형 MLM):  마스킹된 토큰의 대체어를 생성
Discriminator (DeBERTa): 모든 토큰이 원본인지 대체된 것인지 판별
```

$$\mathcal{L} = \mathcal{L}_{MLM} + \lambda \cdot \mathcal{L}_{RTD} \quad (\lambda = 50)$$

### GDES: Gradient-Disentangled Embedding Sharing

ELECTRA의 문제: Generator와 Discriminator가 임베딩을 공유하면, MLM 손실과 RTD 손실이 임베딩을 **반대 방향으로 끌어당긴다** (tug-of-war).

- MLM: 유사한 토큰의 임베딩이 가까워야 함 (대체어 생성에 유리)
- RTD: 유사한 토큰의 임베딩이 구별되어야 함 (원본/대체 판별에 유리)

**GDES 해법**:

$$E_D = \text{sg}(E_G) + E_{\delta}$$

$\text{sg}()$는 stop-gradient 연산자. Discriminator는 Generator 임베딩을 사용하되, RTD의 그래디언트는 잔차 $E_{\delta}$만 업데이트하고 $E_G$는 건드리지 않는다.

| 공유 방식 | MNLI-m | SQuAD F1 |
|---------|--------|---------|
| ES (바닐라 공유) | 88.8 | 86.3 |
| NES (공유 안 함) | 88.3 | 85.3 |
| **GDES** | **89.3** | **87.2** |

---

## 6. 모델 크기 비교

| 모델 | 레이어 | 히든 | 헤드 | 파라미터 | 어휘 |
|------|-------|------|------|---------|------|
| V1-Base | 12 | 768 | 12 | ~134M | 50K |
| V1-Large | 24 | 1024 | 16 | ~385M | 50K |
| V2-XXLarge | 48 | 1536 | 24 | ~1.5B | 128K |
| **V3-XSmall** | **12** | **384** | **6** | **22M** | **128K** |
| V3-Small | 6 | 768 | 12 | 44M | 128K |
| V3-Base | 12 | 768 | 12 | 86M | 128K |
| V3-Large | 24 | 1024 | 12 | 304M | 128K |
| mDeBERTa-V3 | 12 | 768 | 12 | 86M | 250K |

**주목**: V3-XSmall (22M)이 RoBERTa-Base (86M)보다 MNLI에서 +1.2%p 높다.

---

## 7. 벤치마크 결과 종합

### GLUE (V3-Large)

| 과제 | V3-Large | RoBERTa-L | ELECTRA-L |
|------|---------|----------|----------|
| MNLI-m | **91.8** | 90.2 | 90.9 |
| SST-2 | **96.9** | 96.4 | 96.9 |
| QNLI | **96.0** | 93.9 | 94.9 |
| CoLA | **75.3** | 68.0 | 69.1 |
| RTE | **92.7** | 86.6 | 88.0 |
| GLUE 평균 | **91.37** | - | - |

### Base 모델 비교 (SQuAD v2.0)

| 모델 | 파라미터 | SQuAD v2.0 F1/EM | MNLI-m |
|------|---------|-----------------|--------|
| BERT-base | 110M | -/- | 84.5 |
| RoBERTa-base | 86M | 83.7/80.5 | 87.6 |
| ELECTRA-base | 86M | -/80.5 | 88.8 |
| DeBERTa-v1-base | 100M | 86.2/83.1 | 88.8 |
| **DeBERTa-v3-base** | **86M** | **88.4/85.4** | **90.6** |

### 다국어 (mDeBERTa-V3-Base, XNLI zero-shot)

| 모델 | XNLI 평균 |
|------|---------|
| XLM-R Base | 76.2% |
| **mDeBERTa-V3-Base** | **79.8%** (+3.6%p) |

---

## 8. SiFT: Scale-invariant Fine-Tuning

대규모 모델의 파인튜닝 안정성을 위한 가상 적대적 학습:

표준 적대적 학습: 원시 임베딩에 섭동 추가
SiFT: **정규화된** 임베딩 벡터에 섭동 추가

$$\tilde{e} = e + \epsilon \cdot \frac{\delta}{\lVert \delta \rVert}, \quad \delta \sim \mathcal{N}(0, I)$$

정규화된 공간에서의 섭동이 스케일에 무관하게 안정적인 학습을 보장한다.

---

## 9. BERT 계열 모델 계보에서의 위치

```
BERT (2018)
  ├── RoBERTa (2019): 더 많은 데이터 + 더 긴 학습
  ├── ALBERT (2019): 파라미터 공유로 경량화
  ├── XLNet (2019): 순열 언어 모델
  ├── ELECTRA (2020): RTD 사전학습
  └── DeBERTa (2020): 분리된 어텐션 + EMD
        ├── V2 (2021): 1.5B 스케일, SuperGLUE 인간 초과
        └── V3 (2021): RTD + GDES → 최고 효율
```

**DeBERTa V3는 현재 가장 널리 사용되는 인코더 모델 중 하나**이다. HuggingFace에서 `deberta-v3-base`의 월간 다운로드 수가 **235만+ 회**이며, 텍스트 분류, NLI, QA, NER, 감성 분석 등 다양한 다운스트림 과제에서 표준 백본으로 사용된다.

---

## 10. 핵심 통찰 요약

1. **콘텐츠와 위치를 분리**하면 어텐션이 더 정밀해진다 — c2p + p2c 항이 각각 1-2%p 개선을 제공
2. **절대 위치는 마지막에 주입**하는 것이 좋다 — 대부분의 처리는 상대 위치로 충분
3. **RTD는 MLM보다 6.7배 효율적** — 100% 토큰에서 학습 신호
4. **GDES로 tug-of-war 해결** — Generator/Discriminator 임베딩의 그래디언트 분리
5. **22M 파라미터로 86M 모델을 능가** — 아키텍처 혁신이 스케일보다 중요할 수 있다
