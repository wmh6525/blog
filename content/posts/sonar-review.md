---
title: "[논문 리뷰] SONAR — 200개 언어 + 음성을 하나의 1024차원 공간에 통합한 다국어 임베딩"
date: 2026-05-06
tags: ["논문리뷰", "SONAR", "임베딩", "다국어", "음성", "Meta"]
categories: ["ML/AI"]
summary: "SONAR 논문 상세 리뷰. NLLB-200을 1024-dim 병목으로 증류하여 200개 언어 텍스트 + 37개 언어 음성을 같은 임베딩 공간에 매핑한다. LCM, BLASER 2.0, MuTox의 기반이 된 핵심 인프라."
math: true
toc: true
draft: false
---

## 논문 정보

- **제목**: SONAR: Sentence-Level Multimodal and Language-Agnostic Representations
- **저자**: Paul-Ambroise Duquenne, Holger Schwenk, Benoît Sagot
- **소속**: Meta AI (FAIR) + Inria
- **arXiv**: 2308.11466 (2023.08)
- **코드**: [github.com/facebookresearch/SONAR](https://github.com/facebookresearch/SONAR)
- **약어**: Sentence-level multim**O**dal and la**N**guage-**A**gnostic **R**epresentations

---

## 1. 한 줄 요약

> **NLLB-200(200개 언어 번역 모델)을 1024-dim 단일 벡터 병목으로 증류하여 200개 언어 텍스트 + 37개 언어 음성을 같은 의미 공간에 매핑한 시스템.**

CLIP이 이미지-텍스트를 같은 공간에 둔 것처럼, SONAR는 **모든 언어와 음성을 같은 공간에** 둔다.

---

## 2. 계보 — 어떻게 여기까지 왔는가?

```
LASER (BiLSTM, 93언어, 텍스트만)
    ↓
LASER3 (Teacher-Student, 200언어)
    ↓
NLLB-200 (54B 번역 모델)
    ↓
★ SONAR (Transformer 인코더-디코더, 200언어 + 37 음성)
    ↓
Omnilingual SONAR (1,560+ 언어, 2026.03)
```

**흥미로운 양방향 관계**:
- LASER → NLLB의 학습 데이터 마이닝에 기여
- NLLB → SONAR의 초기화 가중치 제공
- SONAR → BLASER 2.0, MuTox, LCM의 기반

---

## 3. 핵심 아이디어

### 문제

- BERT/SBERT는 단일 언어 위주
- LASER3은 BiLSTM 기반, 음성 미지원, 디코더 없음
- 다국어 임베딩이 있어도 **벡터에서 텍스트로 복원** 불가

### SONAR의 해결

```
[입력]
  영어: "I love machine learning"      ─┐
  한국어: "기계 학습을 사랑합니다"        ─┤
  스페인어: "Me encanta el ML"          ─┼─→ 같은 1024-d 벡터
  음성: 어떤 언어의 "I love ML"          ─┘    (의미가 같으면)

[출력]
  같은 벡터로:
  - 영어 디코더 → "I love machine learning"
  - 중국어 디코더 → "我爱机器学习"
  - 일본어 디코더 → "機械学習が大好きです"
  - ... 200개 언어 어느 쪽으로든
```

---

## 4. 아키텍처

### 전체 구조

```
                ┌─────────────────────┐
   텍스트 ─────→│ Text Encoder        │─┐
                │ (NLLB-200 24-layer) │ │
                └─────────────────────┘ │
                                        ↓
                          [1024-dim SONAR 벡터]
                                        │
                ┌─────────────────────┐ │
   음성 ──────→│ Speech Encoder      │─┘
                │ (w2v-BERT 2.0 +     │
                │  3-layer adapter)   │
                └─────────────────────┘
                          
                          ↓ 같은 벡터 ↓
                          
                ┌─────────────────────┐
                │ Text Decoder        │→ 200개 언어 중 어느 텍스트
                │ (NLLB 24-layer)     │
                └─────────────────────┘
```

### Text Encoder

| 항목 | 값 |
|------|-----|
| 베이스 | NLLB-200 1B 인코더 |
| 레이어 | 24-layer Transformer |
| 출력 차원 | **1024-dim 단일 벡터** |
| 풀링 | **Mean-pooling** (max/EOS와 비교 후 채택) |
| 토크나이저 | SentencePiece, vocab 256K |
| 파라미터 | ~0.8B |

### Text Decoder

| 항목 | 값 |
|------|-----|
| 레이어 | 24-layer Transformer 디코더 (NLLB와 동일) |
| **핵심 변형** | **단일 1024-dim 벡터에만 cross-attention** |

기존 NLLB 디코더는 인코더의 가변 길이 시퀀스에 cross-attention. SONAR는 **단일 벡터(병목)**에만 어텐드 — 모든 정보가 그 벡터에 압축되도록 강제.

### Speech Encoder

| 항목 | 값 |
|------|-----|
| 베이스 | **w2v-BERT 2.0** (600M) |
| 풀링 | 3-layer Transformer 디코더가 attention-pooling |
| 출력 | 1024-dim (텍스트와 같은 공간) |
| 입력 샘플링 | 16kHz |
| 언어별 | 영어 인코더 1개 + 비영어 어댑터 다수 |

### Speech Decoder?

**없음**. 음성→텍스트는 텍스트 디코더 재사용 (zero-shot 작동).

---

## 5. 학습 방법 — 4가지 손실

총 손실:

$$\mathcal{L} = \mathcal{L}_{\text{MT}} + \alpha \cdot \mathcal{L}_{\text{MSE}} + \beta \cdot \mathcal{L}_{\text{DAE}}$$

- $\alpha = 0.1$ (MSE 가중치)
- $\beta = 0.01$ (DAE 가중치 — **매우 약하게**)

### 5.1 Translation Loss ($\mathcal{L}_{\text{MT}}$)

표준 sequence-to-sequence cross-entropy:

$$\mathcal{L}_{\text{MT}} = -\sum_{t} \log P(y_t \mid y_{\lt t}, E(x))$$

- $E(x)$: 인코더 출력 (1024-d 벡터)
- $y$: 다른 언어로의 번역
- 한 입력을 **두 개의 다른 타겟 언어**로 번역하도록 학습 → 언어 무관성 강제

### 5.2 MSE Cross-Lingual Similarity Loss ($\mathcal{L}_{\text{MSE}}$)

같은 의미의 다른 언어 문장 쌍이 같은 벡터를 갖도록:

$$\mathcal{L}_{\text{MSE}} = \| E(s_{\text{ko}}) - E(s_{\text{en}}) \|_2^2$$

예: $E(\text{"사랑해요"}) \approx E(\text{"I love you"})$

### 5.3 Auto-Encoding Loss ($\mathcal{L}_{\text{AE}}$)

같은 언어로 복원:

$$\mathcal{L}_{\text{AE}} = -\log P_{\text{decoder-ko}}(s_{\text{ko}} \mid E(s_{\text{ko}}))$$

→ 1024-d 벡터가 **충분한 정보**를 담아야 함을 보장.

### 5.4 Denoising Auto-Encoding Loss ($\mathcal{L}_{\text{DAE}}$)

노이즈 추가된 입력에서 원본 복원:

$$\mathcal{L}_{\text{DAE}} = -\log P_{\text{decoder}}(s \mid E(\tilde{s}))$$

여기서 $\tilde{s}$는 노이즈가 추가된 $s$.

**왜 가중치를 작게?** 큰 값을 쓰면 xsim++(언어 무관 평가) 악화 — DAE가 영어/특정 언어 편향을 강화.

### 5.5 추가 트릭: Random Interpolation Decoding

평행 문장 쌍 임베딩을 무작위 비율로 보간한 벡터로 디코더 fine-tuning:

$$z = \lambda \cdot E(s_{\text{en}}) + (1 - \lambda) \cdot E(s_{\text{ko}}), \quad \lambda \sim \text{Uniform}(0, 1)$$

→ 데이터 증강 + 임베딩 공간의 매끄러움 강화.

---

## 6. 음성 인코더 학습 — Teacher-Student

### 핵심: ASR(전사) 데이터만 사용

```
[Teacher] Text Encoder (frozen)
                ↓
   "사랑해" 텍스트 → 1024-d 벡터 v_text

[Student] Speech Encoder
                ↓
   "사랑해" 음성 → 1024-d 벡터 v_speech

[학습]
   MSE(v_text, v_speech) 최소화
```

### 결과: Zero-Shot Speech Translation

음성 학습 데이터에 **번역**이 없었는데도:

```python
korean_audio.wav → Speech Encoder → 1024-d vec
                → Text Decoder (영어) → "I love you"
```

ASR만 학습했지만 음성→다른 언어 번역이 **zero-shot으로 작동**.

### 사용 코퍼스

- Common Voice 12, MuST-C, VoxPopuli, LibriSpeech 등

---

## 7. 학습 데이터

### 텍스트
- **소스**: NLLB가 사용한 모든 bitext
  - 인간 라벨링 평행 코퍼스
  - Back-translation
  - LASER로 마이닝된 데이터
- **언어**: 200개 (FLORES-200 기준)
- **학습 스텝**: 인코더-디코더 100K, 디코더 fine-tuning 50K

### 두 단계 학습 전략

```
Stage 1: 다국어 텍스트 임베딩 공간 구축
  ├── Text encoder + decoder 학습
  └── 4가지 손실 동시 최소화

Stage 2: 음성을 같은 공간에 매핑
  ├── Text encoder = teacher (frozen)
  └── Speech encoder = student
```

---

## 8. 실험 결과

### 8.1 Cross-Lingual Similarity Search

**xsim**: 다른 언어 문장 중 평행 쌍을 찾는 정확도
**xsim++**: entity 변경/causality 변환 같은 어려운 negatives 포함

| 모델 | xsim ↓ | xsim++ ↓ |
|------|--------|----------|
| LASER3 | 5.1 | 36.4 |
| LaBSE | 10.7 | 36.1 |
| **SONAR** | **1.4** | **15.2** |

→ **xsim++ 에러율 45% 감소** (vs LaBSE)

### 8.2 FLORES-200 BLEU (기계 번역)

| 방향 | SONAR | NLLB-1B | Δ |
|------|-------|---------|---|
| X → eng | 32.9 | 35.2 | -2.3 |
| eng → X | 20.7 | 24.9 | -4.2 |

**1024-d 단일 벡터 병목**인데도 NLLB와 경쟁력 있음 (정보 압축의 위력).

### 8.3 Speech-to-Text Translation (FLEURS, X→eng, **zero-shot**)

| 언어 | SONAR | Whisper-v2 (supervised) |
|------|-------|----------------------|
| 프랑스어 | 33.7 | 34.9 |
| 스페인어 | **28.0** | 27.2 |
| **스와힐리어** | **23.5** | 7.6 |
| 러시아어 | 28.4 | 31.1 |
| **37개 평균** | **25.3** | 24.5 |

**ASR만 학습한 zero-shot SONAR가 supervised Whisper를 평균적으로 능가.** 특히 저자원 언어(스와힐리)에서 압도적.

### 8.4 LASER3 + T-modules 비교

프랑스어 → 영어 음성 번역:
- **SONAR: 46.1 spBLEU**
- LASER3 + T-modules: 40.4 spBLEU

→ 통합 학습이 모듈 결합보다 우수.

---

## 9. 다른 임베딩 모델과 비교

| 모델 | 차원 | 텍스트 언어 | 음성 | 디코더 | 핵심 차별점 |
|------|------|----------|------|------|-----------|
| **SONAR** | **1024** | **200** | **37** | **있음** | 디코더 + 음성 통합 |
| LASER | 1024 | 93 | 0 | 없음 | BiLSTM, 텍스트만 |
| LaBSE | 768 | 109 | 0 | 없음 | BERT 기반 |
| multilingual-E5 | 768 | 100+ | 0 | 없음 | 일반 임베딩 |
| OpenAI ada-002 | 1536 | 다국어 | 0 | 없음 | 폐쇄 API |
| Cohere multilingual | 1024 | 100+ | 0 | 없음 | 폐쇄 API |

**SONAR만의 독보적 특징**:
1. **디코더 보유** — 임베딩에서 텍스트 복원 가능
2. **음성 통합** — 같은 공간에서 텍스트와 음성
3. **200개 언어** — 가장 많은 언어 커버
4. **오픈소스** — 모든 모델 공개

---

## 10. 활용 사례

### 10.1 대규모 Bitext Mining

Common Crawl 같은 단일언어 텍스트에서 **자동으로 평행 문장 발굴** → NLLB 학습 데이터 자동 생성.

### 10.2 Cross-Lingual Transfer

```python
# 영어로 학습한 분류기가 한국어에도 작동
ko_text → SONAR → 1024-d vec → (영어로 학습된) 분류기 → 결과
```

### 10.3 Zero-shot Speech Translation

훈련에 없던 언어 쌍도:
```python
한국어 음성 → SONAR → 영어 텍스트 (zero-shot)
```

### 10.4 LCM의 기반 (가장 중요)

LCM은 **SONAR 공간에서 다음 문장 임베딩을 자회귀로 예측**:

```
[문장 1 SONAR] [문장 2 SONAR] [문장 3 SONAR]
                    ↓ LCM 예측 (Diffusion)
              [문장 4 SONAR]
                    ↓ SONAR 디코더
              "다음 문장의 자연어 출력"
```

---

## 11. 후속 연구 생태계

### BLASER 2.0 (Findings of EMNLP 2024)

- David Dale, Marta R. Costa-jussà
- **SONAR 임베딩 기반 번역 품질 평가 지표**
- 텍스트 202개 + 음성 57개 언어
- Reference-based + Reference-free 모두 지원
- Reference-free → 문장 단위 채점 → **할루시네이션 탐지, 학습 데이터 필터링**에 활용

```python
from sonar.models.blaser.loader import load_blaser_model
blaser = load_blaser_model("blaser_2_0_ref").eval()
# src_emb, mt_emb, ref_emb로 번역 품질 점수
```

### MuTox — 다국어 독성 분류기

- SONAR 임베딩 위에 분류기 학습
- 음성/텍스트 21개 언어 toxicity 검출

### LCM (Large Concept Model, 2024.12)

- arXiv:2412.08821
- **SONAR가 concept embedding space**
- 자세한 내용: [연속 의미 공간 LM 서베이](continuous-vector-language-models.md)

### SONAR-LLM (2025.08)

- arXiv:2508.05305
- LCM의 diffusion sampler 제거 → frozen SONAR 디코더로 token-level CE 학습
- 39M ~ 1.3B 파라미터 범위

### Omnilingual SONAR (2026.03)

- **수천 개 언어로 확장** (1,560개 언어 평가)
- 텍스트/음성/코드/수식 통합
- Split-softmax contrastive loss + synthetic hard negatives
- **결과**:
  - FLORES 200 언어 cross-lingual similarity 에러율 **반감**
  - 1,560 언어에서 에러율 **15배 감소**
  - 1,560→영어 번역 +15 chrF++

---

## 12. 사용법

### 설치

```bash
pip install sonar-space
pip install fairseq2 --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.6.0/cu124
```

### 텍스트 임베딩

```python
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

t2vec = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder"
)

embeddings = t2vec.predict(
    ["안녕하세요. 제 이름은 SONAR입니다."],
    source_lang="kor_Hang"
)
print(embeddings.shape)  # torch.Size([1, 1024])
```

### 임베딩 → 텍스트 디코딩

```python
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

vec2text = EmbeddingToTextModelPipeline(
    decoder="text_sonar_basic_decoder",
    tokenizer="text_sonar_basic_encoder"
)

# 한국어 임베딩을 영어로 디코딩
texts = vec2text.predict(embeddings, target_lang="eng_Latn", max_seq_len=512)
print(texts)  # ["Hello. My name is SONAR."]

# 같은 임베딩을 프랑스어로
texts = vec2text.predict(embeddings, target_lang="fra_Latn", max_seq_len=512)
# ["Bonjour. Je m'appelle SONAR."]
```

### Text-to-Text Translation (200×200 = 40,000 방향)

```python
from sonar.inference_pipelines.text import TextToTextModelPipeline

t2t = TextToTextModelPipeline(
    encoder="text_sonar_basic_encoder",
    decoder="text_sonar_basic_decoder",
    tokenizer="text_sonar_basic_encoder"
)

result = t2t.predict(
    ["저는 SONAR입니다."],
    source_lang="kor_Hang",
    target_lang="fra_Latn"
)
```

### Speech-to-Text Translation (Zero-shot)

```python
from sonar.inference_pipelines.speech import SpeechToTextModelPipeline

s2t = SpeechToTextModelPipeline(
    encoder="sonar_speech_encoder_kor",  # 한국어 음성 인코더
    decoder="text_sonar_basic_decoder",
    tokenizer="text_sonar_basic_decoder"
)

# 한국어 음성 → 영어 텍스트
en_text = s2t.predict(["./korean_audio.wav"], target_lang="eng_Latn")
```

### GPU + FP16 가속

```python
import torch

embedder = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder",
    device=torch.device("cuda"),
    dtype=torch.float16,
)
```

### 다양한 디코딩 전략

```python
from fairseq2.generation import TopPSampler

# Top-p 샘플링으로 더 다양한 출력
results = vec2text.predict(
    embeddings,
    target_lang="eng_Latn",
    sampler=TopPSampler(0.99),
    max_seq_len=128
)
```

---

## 13. 한계

1. **1024-d 병목의 정보 손실**: 긴 문장이나 정보 밀도 높은 문장에서 손실
2. **NLLB 대비 BLEU 격차**: X→eng -2.3, eng→X -4.2 spBLEU
3. **언어별 음성 인코더 필요** (37개) — 단일 모델 통합 X (Omnilingual SONAR가 해결)
4. **음성 인코더는 ASR만 학습** — 진정한 speech-to-speech는 어려움
5. **디코더는 한 번에 한 언어**: target_lang 명시 필요
6. **저자원 언어 디코더 품질 편차**

---

## 14. SONAR가 가능하게 한 것들

```
SONAR
  ├── BLASER 2.0 — 200언어 번역 평가
  ├── MuTox — 21언어 독성 분류
  ├── LCM — 문장 단위 자회귀 LLM
  ├── SONAR-LLM — token-level decoder fine-tuning
  ├── Omnilingual SONAR — 1,560+ 언어 확장
  └── 대규모 다국어 데이터 마이닝
```

**SONAR는 단순한 임베딩 모델이 아니라, 언어 무관 의미 표현의 핵심 인프라**가 되었다.

---

## 15. 핵심 정리

| 질문 | 답 |
|------|------|
| **무엇인가?** | 200개 언어 텍스트 + 37개 언어 음성을 1024-d 벡터로 매핑 |
| **왜 만들었나?** | 언어 무관 의미 표현 → 다국어 작업의 공통 기반 |
| **어떻게 만들었나?** | NLLB-200 teacher + 4가지 손실 (MT CE, MSE, AE, DAE) |
| **어떤 결과?** | xsim++ 에러 45% 감소, zero-shot 음성 번역이 supervised Whisper 평균 능가 |
| **무엇에 쓰나?** | Bitext mining, Zero-shot 번역, BLASER 평가, LCM 기반 |
| **한계는?** | 1024-d 병목, 언어별 음성 인코더, NLLB 대비 약간의 BLEU 격차 |

### 핵심 한 줄

> **"SONAR는 '의미는 언어와 모달리티에 무관하다'는 가설을 1024차원 벡터로 구현한 시스템이다. CLIP이 vision-text를 통합했다면, SONAR는 다국어 + 음성을 통합했다."**

---

## 16. 관련 블로그 포스트

- [연속 의미 공간 언어 모델 서베이](continuous-vector-language-models.md) — LCM 등 활용 사례
- [RAG 동향 총정리](rag-survey-2026.md)
- [검색기 학습 논문 총정리](retriever-training-survey.md) — 다른 임베딩 모델들

---

## 참고 자료

### 핵심
- [SONAR 논문 (arXiv:2308.11466)](https://arxiv.org/abs/2308.11466)
- [공식 코드 (github.com/facebookresearch/SONAR)](https://github.com/facebookresearch/SONAR)
- [Hugging Face](https://huggingface.co/facebook/SONAR)

### 후속 연구
- [BLASER 2.0 (Findings EMNLP 2024)](https://aclanthology.org/2024.findings-emnlp.943/)
- [LCM (arXiv:2412.08821)](https://arxiv.org/abs/2412.08821) | [코드](https://github.com/facebookresearch/large_concept_model)
- [SONAR-LLM (arXiv:2508.05305)](https://arxiv.org/abs/2508.05305)

### 선행
- [LASER (Schwenk 2017)](https://github.com/facebookresearch/LASER)
- [NLLB-200 (arXiv:2207.04672)](https://arxiv.org/abs/2207.04672)
