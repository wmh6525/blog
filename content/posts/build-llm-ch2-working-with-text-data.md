---
title: "[Book] Build a Large Language Model (Ch.2) — 텍스트 데이터 다루기"
date: 2026-03-30
tags: ["연구노트", "LLM", "토크나이저", "임베딩", "BPE"]
categories: ["ML/AI"]
summary: "Sebastian Raschka의 'Build a Large Language Model (From Scratch)' 챕터 2를 한국어로 정리한다. 텍스트 토크나이징, 토큰 ID 변환, 특수 토큰, BPE, 슬라이딩 윈도우 샘플링, 토큰 임베딩, 위치 인코딩까지 LLM 데이터 전처리의 전체 파이프라인을 코드와 함께 다룬다."
math: true
toc: true
draft: false
---

> **원서**: Sebastian Raschka, *Build a Large Language Model (From Scratch)*, Manning, 2025
>
> 이 글은 원서 Chapter 2 "Working with text data"의 핵심 내용을 한국어로 요약·정리한 것입니다.

---

## 이 챕터에서 다루는 내용

- LLM 학습을 위한 텍스트 전처리
- 텍스트를 단어/서브워드 토큰으로 분할
- BPE(Byte Pair Encoding)를 이용한 고급 토크나이징
- 슬라이딩 윈도우 방식의 학습 샘플 생성
- 토큰을 벡터로 변환하여 LLM에 입력

---

## 2.1 워드 임베딩 이해하기

딥러닝 모델은 원시 텍스트를 직접 처리할 수 없다. 텍스트는 범주형(categorical) 데이터이므로, 신경망의 수학 연산과 호환되는 **연속 벡터 표현**으로 변환해야 한다.

이 변환 과정을 **임베딩(embedding)**이라 한다. Word2Vec 같은 사전 학습된 임베딩 모델도 있지만, LLM은 **학습 과정에서 자체 임베딩을 최적화**한다. 이렇게 하면 임베딩이 당면한 과제와 데이터에 맞게 조정된다.

임베딩 차원은 1에서 수천까지 다양하다:
- GPT-2 (125M 파라미터): 768차원
- GPT-3 (175B 파라미터): 12,288차원

차원이 높을수록 더 미세한 관계를 포착하지만, 계산 비용이 증가한다.

---

## 2.2 텍스트 토크나이징

LLM에 입력하기 위한 첫 단계는 텍스트를 **개별 토큰**으로 분할하는 것이다. 토큰은 단어, 특수 문자, 구두점 등이 될 수 있다.

### 간단한 정규표현식 토크나이저

```python
import re

text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
# ['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']
```

이 방식으로 Edith Wharton의 단편소설 "The Verdict"(20,479자)를 토크나이징하면 **4,690개 토큰**이 생성된다.

### 핵심 설계 결정

- **대소문자**: 소문자로 통일하지 않는다 — 고유명사, 문장 구조, 대문자 패턴을 LLM이 학습하도록
- **공백**: 제거하여 토큰 수를 줄이거나, 보존하여 구조 정보를 유지 (코드 생성 등에 유용)

---

## 2.3 토큰을 토큰 ID로 변환

토큰 문자열을 **정수 ID**로 매핑하기 위해 **어휘(vocabulary)**를 구축한다.

```python
# 어휘 구축
all_words = sorted(set(preprocessed))  # 고유 토큰 정렬
vocab = {token: integer for integer, token in enumerate(all_words)}
# {'!': 0, '"': 1, ...., 'younger': 1127, 'your': 1128}
```

어휘 크기: **1,130개** (The Verdict 기준)

### SimpleTokenizerV1 구현

```python
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
```

**문제**: 학습 데이터에 없는 단어(예: "Hello")를 만나면 `KeyError` 발생 → 특수 토큰이 필요하다.

---

## 2.4 특수 컨텍스트 토큰 추가

미지의 단어와 문서 경계를 처리하기 위해 특수 토큰을 추가한다:

- `<|unk|>`: 어휘에 없는 미지의 단어를 대체
- `<|endoftext|>`: 서로 관련 없는 텍스트 소스 사이의 경계 표시

```python
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
# 어휘 크기: 1,130 → 1,132
```

### 다른 특수 토큰들

| 토큰 | 용도 |
|------|------|
| `[BOS]` | 시퀀스의 시작 표시 |
| `[EOS]` | 시퀀스의 끝 표시 (= `<\|endoftext\|>`) |
| `[PAD]` | 배치 내 짧은 텍스트를 패딩 |

GPT 모델은 `<|endoftext|>` 하나로 `[EOS]`와 `[PAD]` 역할을 모두 수행한다. 또한 BPE 토크나이저 덕분에 `<|unk|>` 토큰도 불필요하다.

---

## 2.5 BPE (Byte Pair Encoding)

GPT-2, GPT-3, ChatGPT에서 사용된 토크나이저. BPE의 핵심 아이디어:

1. 모든 개별 문자를 어휘에 추가 ("a", "b", "c", ...)
2. 가장 자주 함께 나타나는 문자 쌍을 반복적으로 병합 ("d" + "e" → "de")
3. 빈도 컷오프까지 반복 → 서브워드 어휘 구축

### tiktoken 라이브러리 사용

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
integers = tokenizer.encode("Hello, do you like tea?")
# [15496, 11, 466, 345, 588, 8887, 30]

strings = tokenizer.decode(integers)
# 'Hello, do you like tea?'
```

**BPE의 장점**: 미지의 단어도 서브워드 또는 개별 문자로 분해하여 처리 가능. `<|unk|>` 토큰이 불필요하다.

예: "Akwirw ier" → "Ak" + "w" + "ir" + "w" + " " + "ier" (6개 서브워드 토큰)

GPT-2의 BPE 어휘 크기: **50,257개** (`<|endoftext|>`가 ID 50,256으로 가장 마지막)

---

## 2.6 슬라이딩 윈도우로 데이터 샘플링

LLM 학습의 입력-타겟 쌍은 **다음 단어 예측** 과제로 생성한다:

```
입력: "LLMs learn to predict one word"  → 타겟: "at"
입력: "LLMs learn to predict one word at" → 타겟: "a"
입력: "LLMs learn to predict one word at a" → 타겟: "time"
```

### 입력-타겟 쌍 구성

```python
context_size = 4
x = enc_sample[:context_size]   # 입력 토큰들
y = enc_sample[1:context_size+1] # 타겟 = 입력을 1칸 시프트

# x: [290, 4920, 2241, 287]     → "and established himself in"
# y: [4920, 2241, 287, 257]     → "established himself in a"
```

### PyTorch Dataset 구현

```python
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
```

### stride의 의미

- `stride=1`: 입력 윈도우가 한 위치씩 이동 → 최대한 많은 학습 샘플, 하지만 중복 많음
- `stride=max_length`: 입력 윈도우가 겹치지 않음 → 오버피팅 방지

---

## 2.7 토큰 임베딩 생성

토큰 ID를 LLM이 처리할 수 있는 **연속 벡터**로 변환하는 마지막 단계.

### 임베딩 레이어 = 룩업 테이블

```python
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
```

임베딩 레이어의 가중치 행렬: `(50257, 256)` — 어휘의 각 토큰에 대해 256차원 벡터가 하나씩 있다.

토큰 ID를 넣으면 해당 행의 벡터를 **조회(lookup)**하여 반환한다:

```python
# 토큰 ID 3의 임베딩 = 가중치 행렬의 4번째 행 (0-indexed)
embedding_layer(torch.tensor([3]))
# tensor([-0.4015, 0.9666, -1.1481])
```

배치 `(8, 4)` → 임베딩 후 `(8, 4, 256)` — 배치 8개, 각 4토큰, 각 256차원 벡터.

**핵심**: 임베딩 가중치는 학습 과정에서 최적화된다. 랜덤 초기화에서 시작하여 LLM 전체 학습과 함께 의미 있는 표현으로 발전한다.

---

## 2.8 단어 위치 인코딩

Self-attention 메커니즘은 **위치 정보가 없다** — 같은 토큰 ID는 시퀀스의 어디에 있든 같은 임베딩 벡터를 생성한다.

이를 해결하기 위해 **위치 임베딩(positional embedding)**을 토큰 임베딩에 더한다.

### 두 가지 위치 임베딩 방식

| 방식 | 설명 | 사용 |
|------|------|------|
| **절대 위치 임베딩** | 각 위치에 고유한 벡터 할당 | GPT |
| **상대 위치 임베딩** | 토큰 간 거리/관계를 인코딩 | RoPE (Llama 등) |

### GPT 스타일 절대 위치 임베딩

```python
context_length = max_length  # 4
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
# shape: (4, 256)
```

### 최종 입력 = 토큰 임베딩 + 위치 임베딩

```python
input_embeddings = token_embeddings + pos_embeddings
# shape: (8, 4, 256)
```

이 `input_embeddings`가 LLM의 핵심 모듈(attention, FFN 등)에 입력되는 최종 형태이다.

---

## 요약: 전체 데이터 파이프라인

```
원시 텍스트
  ↓ 토크나이징 (BPE)
토큰 시퀀스: ["This", "is", "an", "example", "."]
  ↓ 어휘 매핑
토큰 ID: [40134, 2052, 133, 389, 12]
  ↓ 슬라이딩 윈도우 (입력/타겟 쌍 생성)
입력: [40134, 2052, 133, 389]  →  타겟: [2052, 133, 389, 12]
  ↓ 임베딩 레이어 (룩업)
토큰 임베딩: (batch, seq_len, 256)
  ↓ + 위치 임베딩
최종 입력: (batch, seq_len, 256) → LLM에 입력
```

이 파이프라인이 **Stage 1: 데이터 준비와 샘플링**의 전체 과정이며, 다음 챕터에서는 이 입력을 처리하는 **Attention 메커니즘**을 구현한다.
