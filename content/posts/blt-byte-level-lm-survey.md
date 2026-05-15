---
title: "[서베이] BLT와 바이트 수준 언어 모델의 계보 — 토크나이저 없는 LLM을 향해"
date: 2026-03-27
tags: ["논문리뷰", "BLT", "바이트", "토크나이저프리"]
categories: ["ML/AI"]
summary: "Byte Latent Transformer(BLT)와 바이트/문자 수준 언어 모델 28개의 계보를 정리한다. 고정 패칭(MEGABYTE)에서 동적 엔트로피 패칭(BLT), 완전 E2E 학습(H-Net)까지, '토크나이저 없는 LLM'의 발전을 추적한다."
math: true
toc: true
draft: false
---

## 1. 왜 토크나이저를 없애려 하는가?

BPE 같은 고정 토크나이저의 근본적 문제:

| 문제 | 설명 |
|------|------|
| **고정 어휘** | 학습 데이터의 통계에 종속. 새 도메인/언어에 취약 |
| **형태소 불일치** | 한국어 "공부했다" → "공부", "했", "다"로 잘리지 않을 수 있음 |
| **노이즈 취약** | "Hello" vs "H3llo" → 완전히 다른 토큰 시퀀스 |
| **다국어 불공정** | 영어 1토큰 ≈ 한국어 3-5토큰 → 같은 의미에 더 많은 연산 |
| **문자 수준 이해 불가** | 철자, 운율, 아나그램 등을 처리 못함 |

**바이트 수준 모델의 약속**: 이 모든 문제를 원천적으로 해결한다. 하지만 시퀀스가 3-6배 길어지는 비용을 어떻게 감당할 것인가?

---

## 2. BLT: Byte Latent Transformer — 핵심 모델

### 논문 정보

- **제목**: Byte Latent Transformer: Patches Scale Better Than Tokens
- **저자**: Artidoro Pagnoni et al. (FAIR at Meta)
- **학회**: ACL 2025 (arXiv: 2412.09871)

### 아키텍처: 3단계 파이프라인

```
바이트 시퀀스 → [Local Encoder] → 패치 표현 → [Latent Transformer] → [Local Decoder] → 바이트 출력
```

1. **Local Encoder**: 경량 Transformer로 바이트를 **패치 표현**으로 압축. Cross-attention pooling + byte n-gram 해시 임베딩
2. **Latent Global Transformer**: 계산 비용이 큰 핵심 모델. **패치 수준**에서 작동 (바이트 수가 아닌)
3. **Local Decoder**: Cross-attention으로 패치 표현을 바이트로 디코딩

### 핵심 혁신: 엔트로피 기반 동적 패칭

사전 학습된 소형 바이트 LM이 각 위치에서 **다음 바이트 엔트로피**를 계산한다:

$$H(t) = -\sum_b p(b_t \mid b_{\lt t}) \log p(b_t \mid b_{\lt t})$$

엔트로피가 높은 위치(예측하기 어려운 곳)에 패치 경계를 배치한다:

- **엔트로피 높음** → 복잡한 영역 → **작은 패치** → 더 많은 연산 할당
- **엔트로피 낮음** → 예측 가능한 영역 → **큰 패치** → 연산 절약

이것은 **Bayesian Surprise와 직접 연결**된다 — 놀라움이 큰 위치에 더 많은 "주의"를 기울이는 것이다.

### Byte N-gram 해시 임베딩

3-gram부터 8-gram까지의 바이트 n-gram을 해시하여 임베딩:

$$e(b_t) = \sum_{n=3}^{8} E_n[\text{hash}(b_{t-n+1:t})]$$

이것만으로 ~0.04 BPB 개선. 바이트 수준에서 서브워드 정보를 복원하는 핵심 트릭.

### 주요 결과

| 비교 | 결과 |
|------|------|
| vs Llama 3 (BPE) | 동등 FLOP에서 **동등 이상** 성능 (1B-8B) |
| CUTE (철자 벤치마크) | Llama 3보다 **+27점** (99.9% 정확도) |
| 노이즈 강건성 | HellaSwag 변형에서 **+8점** |
| 추론 FLOP | 패치 크기 8에서 BPE 대비 **~50% 절감** |

---

## 3. 계보: 바이트 수준 모델의 발전

### 3.1 고정 패칭 → 동적 패칭 → 완전 E2E

```
MEGABYTE (2023)      BLT (2024)           H-Net (2025)
고정 크기 패치    →   엔트로피 기반 동적  →   완전 E2E 학습 경계
간단하지만 비효율      외부 모델 필요          경계도 모델이 학습
```

### 3.2 MEGABYTE (NeurIPS 2023, Meta)

BLT의 직접적 전신. 바이트 시퀀스를 **고정 크기 패치**로 나누고, 글로벌 모델(패치 간)과 로컬 모델(패치 내)을 분리.

- **한계**: 고정 크기 패치가 의미 단위와 무관하게 잘림
- **BLT로의 발전**: 엔트로피 기반 동적 패칭으로 의미 있는 경계에서 분할

### 3.3 Dynamic Token Pooling (ACL 2023, Nawrot et al.)

BLT 엔트로피 패칭의 직접적 영감. 여러 경계 추론 방법을 비교:
- Gumbel-sigmoid 학습
- 서브워드 토크나이저에서 지도 학습
- **엔트로피 스파이크 기반** ← BLT가 이것을 채택

### 3.4 SpaceByte (NeurIPS 2024)

공백(space) 위치에만 글로벌 블록을 삽입하는 간단한 휴리스틱. BLT 논문에서 "가장 가까운 경쟁자"로 언급됨. 규칙 기반이라 추가 모델 불필요.

### 3.5 H-Net (arXiv 2025, CMU Goombalab)

**완전 E2E**로 경계를 학습하는 계층적 아키텍처. Mamba-2를 바이트 인코더/디코더로, Transformer를 청크 백본으로 사용. 외부 엔트로피 모델 없이 ~4.5-5 bytes/chunk로 자연 수렴.

---

## 4. 플랫 바이트 수준 모델 (압축 없이)

계층적 압축 없이 모든 바이트를 직접 처리하는 접근.

### ByT5 (TACL 2022, Google)

T5를 UTF-8 바이트에 직접 적용. 노이즈 강건성과 다국어 성능에서 mT5를 능가. 하지만 3-6배 긴 시퀀스로 **추론 속도가 심각하게 느림**.

### CANINE (TACL 2022, Google)

유니코드 문자열에 직접 작동하는 인코더. **다운샘플링-Transformer-업샘플링** 패턴의 선구자. BLT의 3단계 구조에 직접 영감.

### MambaByte (COLM 2024, Cornell)

Mamba SSM을 바이트 수준 자기회귀 모델링에 적용. SSM의 선형 복잡도가 긴 바이트 시퀀스에 자연스럽게 적합. BLT와 다른 접근 — 압축 대신 효율적 아키텍처 사용.

### EvaByte (2025, SambaNova/HKU)

6.5B 바이트 LM. EVA(efficient attention) + multibyte prediction으로 5-10배 빠른 디코딩. 5배 적은 학습 데이터로 토큰 기반 모델과 동등.

---

## 5. 학습된 다운샘플링

### Charformer / GBST (ICLR 2022, Google)

**Gradient-Based Subword Tokenization** — 후보 서브워드 블록을 열거하고, 미분 가능한 스코어링으로 소프트 혼합. BLT의 학습된 세그멘테이션 개념의 선구자.

### MrT5 (ICLR 2025, Stanford/Google)

ByT5 인코더의 초기 레이어에 **삭제 게이트**를 추가하여 불필요한 바이트 토큰 제거. 80% 시퀀스 축소, 40% 런타임 개선.

### Funnel-Transformer (NeurIPS 2020, CMU/Google)

인코더가 깊어질수록 시퀀스 길이를 점진적으로 축소. BLT/Hourglass의 점진적 압축 개념의 원조.

---

## 6. Hourglass / 계층적 아키텍처

### Hourglass Transformer (NAACL 2022, Warsaw/OpenAI)

모래시계 형태 — 전반부 다운샘플링, 후반부 업샘플링, 스킵 연결. BLT의 **Local Encoder → Latent Transformer → Local Decoder** 구조에 직접 영감.

### Perceiver / Perceiver IO / Perceiver AR (ICML 2021-22, DeepMind)

Cross-attention으로 긴 입력을 고정 크기 잠재 공간으로 압축. BLT의 cross-attention pooling 메커니즘의 원조. Perceiver AR은 자기회귀 생성으로 확장.

---

## 7. 증류 / 변환

### Bolmo (2025, Ai2)

기존 OLMo(서브워드)를 BLT 아키텍처로 **증류**하여 바이트 모델 생성. 전체 사전학습의 <1% 비용. BLT 아키텍처의 실용성 검증.

### 토큰→바이트 증류 (2026)

Llama, Qwen 등 기존 토큰 LLM을 바이트 모델로 변환. ~125B 바이트로 선생 능력의 92%+ 유지.

---

## 8. 전체 비교 표

| 모델 | 연도 | 접근 | 패칭 | 바이트/패치 비율 | 규모 |
|------|------|------|------|----------------|------|
| **ByT5** | 2021 | 플랫 | 없음 (1:1) | 1 | ~13B |
| **CANINE** | 2021 | 다운샘플 | 고정 stride | ~4 | ~130M |
| **Charformer** | 2021 | 소프트 다운샘플 | 학습된 혼합 | 가변 | ~110M |
| **Hourglass** | 2021 | 계층적 | 고정 풀링 | 2-4 | ~100M |
| **MEGABYTE** | 2023 | 계층적 | **고정 크기** | 고정 (4-8) | ~1.5B |
| **SpaceByte** | 2024 | 규칙 기반 | 공백 위치 | ~5 | ~800M |
| **MambaByte** | 2024 | 플랫 SSM | 없음 | 1 | ~350M |
| **BLT** | 2024 | 계층적 | **엔트로피 동적** | ~6 (가변) | **8B** |
| **MrT5** | 2024 | 삭제 게이트 | 학습된 삭제 | 5 (80% 삭제) | ~580M |
| **EvaByte** | 2025 | 플랫 효율 | 없음 | 1 (multibyte 예측) | **6.5B** |
| **Bolmo** | 2025 | 증류 | BLT식 | ~5 | **7B** |
| **H-Net** | 2025 | 계층적 | **E2E 학습** | ~4.5 (자연 수렴) | **1B+** |
| **T-Free** | 2025 | 계층적 | 문자→단어 | ~5 | **7B** |

---

## 9. 핵심 통찰

### 패칭 전략의 스펙트럼

```
고정 패칭 ←→ 규칙 기반 ←→ 외부 모델 기반 ←→ 완전 E2E
MEGABYTE    SpaceByte    BLT              H-Net
(간단)      (빠름)       (성능 최고)       (가장 순수)
```

### BLT의 위치

BLT는 현재 **가장 성공적인 바이트 수준 LLM**이다:
- 8B 규모에서 BPE 기반 Llama 3과 동등 이상
- 엔트로피 기반 동적 패칭이 고정 패칭을 압도
- N-gram 해시 임베딩이 서브워드 정보를 효과적으로 복원

### 남은 과제

1. **E2E 학습**: BLT는 외부 엔트로피 모델에 의존. H-Net이 E2E를 시도했지만 아직 초기
2. **추론 속도**: 바이트 디코딩이 토큰 디코딩보다 느림. Multibyte prediction(EvaByte)이 유망
3. **대규모 확장**: BLT는 8B까지만 검증. 70B+ 규모 미확인
4. **멀티모달**: bGPT, MBLM이 시도했지만 아직 초기

### SancMamba와의 연결

SancMamba 프로젝트는 BLT 계보의 연장선에 있다:
- BLT의 엔트로피 패칭 → SancMamba의 SSM 기반 경계 감지
- MEGABYTE의 글로벌/로컬 분리 → SancMamba의 인코더/개념 추론/디코더
- H-Net의 E2E 학습 → SancMamba의 목표 (경계를 SANC 에너지로 학습)

---

## 참고 문헌

| 모델 | arXiv |
|------|-------|
| BLT | 2412.09871 |
| MEGABYTE | 2305.07185 |
| SpaceByte | 2404.14408 |
| ByT5 | 2105.13626 |
| CANINE | 2103.06874 |
| Charformer/GBST | 2106.12672 |
| Hourglass | 2110.13711 |
| Dynamic Token Pooling | 2211.09761 |
| MambaByte | 2401.13660 |
| Perceiver | 2103.03206 |
| Perceiver AR | 2202.07765 |
| Funnel-Transformer | 2006.03236 |
| EvaByte | (HKU NLP blog) |
| Bolmo | 2512.15586 |
| H-Net | 2507.07955 |
| MrT5 | 2410.20771 |
| T-Free | 2501.10322 |
| MBLM | 2502.14553 |
| bGPT | 2402.19155 |
| Sparse Transformer | 1904.10509 |
