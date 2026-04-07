---
title: "[연구분석] 도메인 특화 RAG용 LLM 구축 완전 가이드 — 경량 기법부터 풀 파이프라인까지"
date: 2026-04-07
tags: ["연구노트", "RAG", "LLM", "파인튜닝", "도메인특화", "LoRA", "RAFT", "Self-RAG"]
categories: ["ML/AI"]
summary: "RAG 시스템을 위한 도메인 최적화 LLM 구축의 모든 접근법을 총정리한다. Continued Pre-training, SFT(LoRA/QLoRA), RAFT, Self-RAG, RA-DIT, 임베딩 파인튜닝, DPO/RLHF, Knowledge Distillation까지 — 논문 근거, 데이터 요구량, 컴퓨트 비용, 기대 성능 향상, 적용 시점을 모두 포함한다."
math: true
toc: true
draft: false
---

## 0. 전체 조감도: 경량 → 중량 순서

도메인 특화 RAG LLM을 구축하는 접근법은 **비용과 복잡도** 순서로 정리하면 다음과 같다.

| 단계 | 접근법 | 비용 | 기대 향상 | 주요 논문 |
|------|--------|------|----------|----------|
| 0 | **프롬프트 엔지니어링** | $0 | +10-30% | Few-shot, CoT |
| 1 | **임베딩 모델 파인튜닝** | $ | +5-15% 검색 정확도 | BGE, E5-Mistral |
| 2 | **SFT (LoRA/QLoRA)** | $$ | +15-40% 생성 품질 | LoRA, QLoRA, LIMA |
| 3 | **RAFT** | $$ | +20-35% RAG 정확도 | Zhang et al. 2024 |
| 4 | **Self-RAG / RA-DIT** | $$$ | +25-40% 전체 품질 | Asai et al. 2023, Lin et al. 2023 |
| 5 | **DPO/RLHF** | $$$ | +10-20% 충실도 | Rafailov et al. 2023, SimPO |
| 6 | **Continued Pre-training** | $$$$ | +10-30% 도메인 이해 | Don't Stop Pretraining, BloombergGPT |
| 7 | **Knowledge Distillation** | $$$ | 90-99% 교사 성능 유지 | InstructRetro |

---

## 1. Continued Pre-training (CPT) / Domain-Adaptive Pre-training

### 핵심 논문: "Don't Stop Pretraining" (Gururangan et al., ACL 2020)

- **저자**: Suchin Gururangan, Ana Marasovic, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, Noah A. Smith
- **arXiv**: 2004.10964

#### 핵심 발견

두 가지 추가 사전학습 전략이 모두 효과적:

1. **Domain-Adaptive Pre-Training (DAPT)**: 도메인 비라벨 코퍼스로 추가 사전학습
2. **Task-Adaptive Pre-Training (TAPT)**: 태스크 데이터(비라벨)로 추가 사전학습

$$\text{최적 성능} = \text{DAPT} + \text{TAPT} \text{ (순차 적용)}$$

| 전략 | 데이터 규모 | 효과 |
|------|-----------|------|
| DAPT 단독 | 수십~수백GB | 고/저자원 모두에서 성능 향상 |
| TAPT 단독 | 수MB~수GB | DAPT보다 작은 데이터로도 효과적 |
| DAPT + TAPT | 결합 | **최고 성능** |
| Curated TAPT | 선별된 태스크 유사 데이터 | 자원 제한 시 DAPT 대안 |

4개 도메인(바이오메디컬, CS, 뉴스, 리뷰) x 8개 분류 태스크에서 검증.

> **핵심 인사이트**: "multi-phase adaptive pretraining offers large gains in task performance." 도메인 데이터로의 TAPT는 DAPT 후에도 추가 성능 향상을 가져온다.

### 주요 사례: BloombergGPT

- **저자**: Shijie Wu et al. (Bloomberg)
- **arXiv**: 2303.17564
- **모델**: 50B 파라미터, 금융 특화

#### 데이터 전략

| 데이터 | 토큰 수 | 비율 |
|--------|---------|------|
| **FinPile** (금융 도메인) | 363B | ~51% |
| **일반 데이터** (Pile 등) | 345B | ~49% |
| **합계** | 708B | 100% |

> **핵심 교훈**: 도메인 데이터와 일반 데이터를 **약 1:1 비율**로 혼합하면 금융 태스크에서 큰 폭의 향상을 얻으면서도 일반 벤치마크 성능을 유지할 수 있다. Training Chronicles(Appendix C)에서 학습 과정의 실무 경험을 상세히 공개.

### 주요 사례: Code Llama

- **저자**: Baptiste Roziere et al. (Meta AI, 26인)
- **arXiv**: 2308.12950
- **모델**: 7B, 13B, 34B, 70B 변종

Llama 2 기반으로 **코드 데이터에 대한 continued pre-training** 수행:
- 16K 시퀀스 길이로 학습, 100K 토큰 입력에서도 개선
- Code Llama - Python 7B가 Llama 2 70B를 HumanEval/MBPP에서 능가
- HumanEval 67%, MBPP 65% 달성
- 인필링, 긴 컨텍스트, 제로샷 명령 수행 지원

### 주요 사례: Med-PaLM 2

- **저자**: Karan Singhal, Tao Tu et al. (Google, 30+인)
- **arXiv**: 2305.09617

PaLM 2 기반, 의료 도메인 파인튜닝:
- **MedQA 86.5%** (전작 대비 19%+ 향상)
- MedMCQA, PubMedQA, MMLU 임상 토픽에서 SOTA
- 의사 평가: 9개 축 중 8개에서 Med-PaLM 2 응답을 **의사 작성 답변보다 선호**
- **Ensemble Refinement** 프롬프팅 기법 도입

### CPT 실무 가이드라인

| 항목 | 권장 |
|------|------|
| **데이터 규모** | 최소 수십GB, 이상적으로 수백GB |
| **도메인:일반 비율** | 1:1 ~ 7:3 (도메인 비중 높이되 일반 유지) |
| **학습률** | 원래 사전학습의 1/10 ~ 1/5 |
| **토큰 수** | 수십B ~ 수백B |
| **GPU 비용** | 7B 모델 기준 수백~수천 GPU-hours |
| **적용 시점** | 도메인 용어/개념이 일반 LLM에 없을 때 |
| **주의사항** | Catastrophic forgetting 방지를 위해 일반 데이터 혼합 필수 |

### 중요한 발견: Fine-tuning으로는 새 지식 주입이 어렵다

Gekhman et al. (2024, arXiv 2405.05904)의 연구:
- LLM은 파인튜닝으로 새로운 사실 지식을 습득하는 데 어려움을 겪음
- 새로운 지식이 포함된 예시는 기존 지식 강화 예시보다 **훨씬 느리게 학습**됨
- 새 지식 예시가 학습되면서 **할루시네이션이 점진적으로 증가**
- 결론: LLM은 사전학습 단계에서 사실을 습득하며, 파인튜닝은 지식 활용 최적화가 주된 역할

> **RAG에 대한 시사점**: 이것이 RAG가 필수적인 근본적 이유다. 파인튜닝만으로는 도메인 지식을 안정적으로 주입할 수 없으므로, 외부 검색으로 지식을 공급하고 모델은 그 지식을 **활용하는 능력**을 최적화해야 한다.

Gekhman et al.의 별도 연구 (2023, arXiv 2312.05934)도 동일 결론:
> "RAG consistently outperforms [unsupervised fine-tuning], both for existing knowledge encountered during training and entirely new knowledge."

---

## 2. Supervised Fine-Tuning (SFT) for RAG

### 2.1 LoRA (Low-Rank Adaptation)

- **저자**: Edward J. Hu, Yelong Shen et al. (Microsoft)
- **arXiv**: 2106.09685
- **학회**: ICLR 2022

#### 핵심 메커니즘

사전학습된 가중치를 동결하고, 각 Transformer 레이어에 **저랭크 분해 행렬**을 주입:

$$W' = W + BA$$

여기서 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$.

| 지표 | LoRA vs Full Fine-tuning |
|------|------------------------|
| 학습 파라미터 | **1/10,000** (GPT-3 175B 기준) |
| GPU 메모리 | **1/3** |
| 추론 지연 | **추가 없음** (행렬 병합 가능) |
| 성능 | RoBERTa, DeBERTa, GPT-2, GPT-3에서 동등 또는 우수 |

#### RAG 파인튜닝용 실무 설정

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,              # 랭크: 8-64 범위, 도메인 복잡도에 따라 조정
    lora_alpha=32,     # 스케일링: 보통 r의 2배
    lora_dropout=0.05, # 오버피팅 방지
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 어텐션 레이어
    bias="none",
    task_type="CAUSAL_LM",
)
```

### 2.2 QLoRA (Quantized LoRA)

- **저자**: Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer
- **arXiv**: 2305.14314
- **학회**: NeurIPS 2023

#### 3대 혁신

1. **4-bit NormalFloat (NF4)**: 정규분포 가중치에 정보이론적으로 최적인 4비트 양자화
2. **Double Quantization**: 양자화 상수 자체를 양자화 → 추가 메모리 절감
3. **Paged Optimizers**: 메모리 스파이크를 페이징으로 관리

#### 성능

- 65B 모델을 **단일 48GB GPU**에서 파인튜닝 가능
- 16비트 풀 파인튜닝과 **동등한 태스크 성능** 유지
- Guanaco 모델: Vicuna 벤치마크에서 **ChatGPT 성능의 99.3%** 달성
- **단일 GPU, 24시간** 학습

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,       # Double Quantization
)
```

### 2.3 LIMA: "Less Is More for Alignment"

- **저자**: Chunting Zhou et al. (Meta AI, 15인)
- **arXiv**: 2305.11206
- **학회**: NeurIPS 2023

#### 핵심 발견

> "Almost all knowledge in large language models is learned during pretraining, and only limited instruction tuning data is necessary."

- 65B 모델 + **1,000개 정교하게 선별된 예시** (RLHF 없음)
- GPT-4와 비교: **43%**에서 동등 또는 우수
- Bard 대비 **58%**, DaVinci003 대비 **65%** 선호

> **RAG SFT에 대한 시사점**: 데이터 **양보다 질**이 중요하다. 1,000개의 고품질 (query, context, response) 트리플렛이 10,000개의 저품질 데이터보다 효과적일 수 있다.

### 2.4 RAG용 SFT 데이터 포맷

RAG 특화 SFT의 핵심은 **검색된 컨텍스트를 활용하여 답변하는 능력**을 가르치는 것이다.

#### 표준 데이터 포맷

```json
{
  "instruction": "다음 문맥을 참고하여 질문에 답하시오.",
  "context": "[검색된 문서 1]\n[검색된 문서 2]\n...",
  "question": "사용자 질문",
  "response": "문맥에 근거한 답변. 출처: [문서 1]"
}
```

#### 데이터 생성 전략

1. **실제 도메인 문서**에서 질문 자동 생성 (GPT-4 활용)
2. **검색 시스템**을 사용하여 실제 검색 결과를 context로 포함
3. **오답 문서(distractor)** 혼합으로 선별 능력 학습
4. **답변 불가 케이스** 포함: "제공된 문맥에서 답을 찾을 수 없습니다"

#### 중요 발견: 1,000개 예시로 충분

Yoran et al. (2023, arXiv 2310.01558)의 연구:
> "Even 1,000 examples suffice to train the model to be robust to irrelevant contexts while maintaining high performance on examples with relevant ones."

관련/무관 문맥을 혼합한 **1,000개 학습 예시**만으로 모델이 무관한 검색 결과를 무시하는 법을 학습.

---

## 3. RAFT (Retrieval Augmented Fine-Tuning)

- **저자**: Tianjun Zhang, Shishir G. Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion Stoica, Joseph E. Gonzalez (UC Berkeley)
- **arXiv**: 2403.10131 (2024.03)
- **코드**: [gorilla.cs.berkeley.edu/blogs/9_raft.html](https://gorilla.cs.berkeley.edu/blogs/9_raft.html)

### 핵심 아이디어: "오픈 북 시험 공부법"

RAFT는 도메인 특화 RAG를 **오픈 북 시험 준비**에 비유한다.

| 시나리오 | 비유 | 방법 |
|---------|------|------|
| **Closed-book** | 교과서 없이 시험 | 사전학습 지식만 사용 |
| **Open-book (일반 RAG)** | 아무 참고서나 허용된 시험 | 범용 RAG |
| **RAFT** | 특정 교과서로 오픈 북 시험 준비 | 도메인 문서로 파인튜닝 |

### 학습 데이터 구성

핵심 혁신: **Oracle 문서와 Distractor 문서를 혼합**하여 학습.

```
P%의 데이터:    Q + D* + D₁ + D₂ + ... + Dₖ → A*
(1-P)%의 데이터: Q + D₁ + D₂ + ...           → A*
```

- **D***: Oracle 문서 (정답이 포함된 문서)
- **D₁...Dₖ**: Distractor 문서 (무관한 문서)
- **P%**: Oracle 문서가 포함되는 비율
- **(1-P)%**: Distractor만 있는 경우 → 모델이 도메인 지식을 **암기**하도록 유도

### Chain-of-Thought 답변 형식

```
##Reason: [추론 과정 설명]
##begin_quote## 관련 문서에서 직접 인용한 텍스트 ##end_quote##
##Answer: [최종 답변]
```

- **Verbatim citation**: 관련 문서에서 정답 근거를 **그대로 인용**
- **CoT reasoning**: 논리적 추론 과정을 명시
- 이 두 가지의 조합이 할루시네이션을 크게 줄임

### 벤치마크 결과 (Llama2-7B 기준)

| 데이터셋 | 도메인 | RAFT 효과 |
|---------|--------|----------|
| **PubMed QA** | 바이오메디컬 | 유의미한 향상 |
| **HotpotQA** | 위키피디아 일반 지식 | 유의미한 향상 |
| **Gorilla** | API 문서/코드 생성 | 유의미한 향상 |

### 핵심 Ablation 결과

| 컴포넌트 | 제거 시 영향 |
|---------|------------|
| Distractor 문서 제거 | 성능 하락 — 모델이 무관 문서 무시 능력 상실 |
| CoT 제거 | 성능 하락 — 단순 답변은 근거가 약해짐 |
| Verbatim quote 제거 | 성능 하락 — 문서 근거 없는 답변 증가 |
| Oracle 비율(P) 조정 | P가 너무 높으면 암기 불충분, 너무 낮으면 검색 활용 불충분 |

### RAFT 실무 적용

```python
# RAFT 학습 데이터 생성 파이프라인 (개념적)
def create_raft_dataset(documents, num_distractors=4, oracle_ratio=0.8):
    dataset = []
    for doc in documents:
        # GPT-4로 질문-답변 생성
        qa_pairs = generate_qa_with_cot(doc)  
        
        for q, a in qa_pairs:
            if random.random() < oracle_ratio:
                # P%: Oracle + Distractors
                distractors = sample_distractors(documents, exclude=doc, k=num_distractors)
                context = shuffle([doc] + distractors)
            else:
                # (1-P)%: Distractors만
                distractors = sample_distractors(documents, exclude=doc, k=num_distractors+1)
                context = distractors
            
            dataset.append({
                "question": q,
                "context": context,
                "answer": format_cot_answer(a, doc)  # ##Reason...##Answer 형식
            })
    return dataset
```

---

## 4. RAG-Specific Fine-Tuning 접근법

### 4.1 Self-RAG (Self-Reflective Retrieval-Augmented Generation)

- **저자**: Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi
- **arXiv**: 2310.11511 (2023.10)
- **학회**: ICLR 2024

#### 핵심 혁신: 반성 토큰(Reflection Tokens)

일반 RAG는 항상 검색하지만, Self-RAG는 **언제 검색할지 스스로 결정**한다.

| 반성 토큰 | 역할 | 값 |
|-----------|------|-----|
| **[Retrieve]** | 검색이 필요한가? | Yes / No / Continue |
| **[ISREL]** | 검색 문서가 관련 있는가? | Relevant / Irrelevant |
| **[ISSUP]** | 생성이 문서에 의해 지지되는가? | Fully / Partially / No Support |
| **[ISUSE]** | 생성이 유용한가? | 1-5 등급 |

#### 3단계 학습 파이프라인

**1단계: 비평가(Critic) 모델 학습**
- GPT-4로 반성 토큰 주석 데이터 생성
- 소규모 모델을 비평가로 학습

**2단계: 생성기(Generator) 학습**
- 비평가가 주석한 반성 토큰을 포함하여 표준 next-token prediction
- 입력-출력 쌍에 검색 문서 + 반성 토큰을 증강

**3단계: 추론 — Segment-level Beam Search**
- 각 세그먼트에서 [Retrieve] 토큰 확률로 검색 여부 결정
- 검색 시: 여러 문서로 병렬 생성 → 반성 토큰 점수로 최선 선택
- **Tree decoding**: 바람직한 비평 토큰 확률의 선형 보간으로 K개 최적 연속 식별

#### Self-RAG vs 일반 RAG

| 특성 | 일반 RAG | Self-RAG |
|------|---------|---------|
| 검색 시점 | 항상 (고정) | 적응적 (필요시만) |
| 문서 평가 | 없음 | [ISREL] 토큰으로 자동 평가 |
| 생성 검증 | 없음 | [ISSUP] 토큰으로 근거 검증 |
| 검색 빈도 제어 | 불가 | 추론 시 소프트 제약으로 조절 가능 |
| RLHF 필요 | - | **불필요** (메모리 효율적, 안정적) |

#### 결과

- Self-RAG 7B/13B: ChatGPT 및 retrieval-augmented Llama2-Chat를 6개 태스크에서 능가
- **사실성(factuality)과 인용 정확도(citation accuracy)**에서 유의미한 향상
- 특히 long-form generation에서 큰 개선

### 4.2 RA-DIT (Retrieval-Augmented Dual Instruction Tuning)

- **저자**: Xi Victoria Lin, Xilun Chen et al. (Meta)
- **arXiv**: 2310.01352 (2023.10)

#### 핵심: LLM과 Retriever를 **동시에** 업데이트

기존 접근법은 LLM 또는 Retriever 중 하나만 업데이트하지만, RA-DIT는 **두 단계 순차 파인튜닝**을 수행:

**Step 1 — LM Update**: 사전학습된 LM이 검색된 정보를 **더 잘 활용**하도록 파인튜닝
**Step 2 — Retriever Update**: LM이 **선호하는** 검색 결과를 반환하도록 Retriever를 업데이트 (LM-Supervised Retrieval, LSR)

$$\text{RA-DIT} = \underbrace{\text{LM Fine-tune}}_{\text{검색 활용 능력 ↑}} + \underbrace{\text{Retriever Fine-tune}}_{\text{LM 선호도 기반 검색 ↑}}$$

#### 결과: RA-DIT 65B

| 설정 | 향상 |
|------|------|
| Zero-shot | 기존 대비 **+8.9%** (평균) |
| Five-shot | 기존 대비 **+1.4%** (평균) |

> 지식 집약적(knowledge-intensive) 벤치마크에서 SOTA 달성.

### 4.3 ChatQA (NVIDIA)

- **저자**: NVIDIA Research
- **arXiv**: 2401.10225 (2024.01)

#### 2단계 Instruction Tuning

1. **Stage 1 — SFT**: 일반적인 instruction following 학습
2. **Stage 2 — Context-Enhanced Fine-tuning**: 검색된 문맥과 함께 QA를 수행하는 학습, 답변 불가 케이스 포함

#### 핵심 특징

- **ChatQA-1.0-70B** (Llama2 기반): GPT-4-0613을 **능가** (54.14 vs 53.90)
- **Llama3-ChatQA-1.5-70B**: GPT-4-Turbo를 **4.4% 능가**
- OpenAI GPT 합성 데이터 **미사용**
- 답변 불가(unanswerable) 질문 처리 포함
- 대화형 QA 최적화 Dense Retriever 별도 학습

### 4.4 RankRAG (NVIDIA)

NVIDIA의 후속 연구로, 단일 LLM이 **문맥 랭킹**과 **답변 생성**을 동시에 수행하도록 instruction-tuning. 별도 리랭커 없이 LLM 자체가 검색 결과를 평가하고 답변을 생성한다.

### 4.5 InstructRetro: 검색 증강 사전학습의 스케일링

- **저자**: NVIDIA Research
- **arXiv**: 2310.07713
- **학회**: ICML 2024

43B GPT 모델에 Retro 증강 방식으로 **1.2조 토큰에서 검색하며** 100B 토큰 추가 학습:

| 태스크 유형 | GPT 대비 향상 |
|------------|-------------|
| 단문 QA / 독해 (8개) | **+7%** |
| 장문 QA (4개) | **+10%** |
| 요약 (3개) | **+16%** |

- 추가 GPU 비용: 기존의 **2.58%**만 추가
- 인코더를 제거하고 디코더 백본만 사용해도 **동등한 성능** 유지

### 4.6 Making RALMs Robust to Irrelevant Context

- **저자**: Ori Yoran et al.
- **arXiv**: 2310.01558

검색 결과에 무관한 문서가 포함될 때의 성능 저하 문제 해결:

**접근 1 — NLI 필터링**: NLI 모델로 검색 문서-답변 쌍의 수반(entailment) 여부 확인 → 무관 문서 제거 (단, 관련 문서도 일부 제거됨)

**접근 2 — Mixed Context Fine-tuning**: 관련/무관 문맥을 혼합하여 파인튜닝 → 모델이 자체적으로 무관 정보를 무시하는 법을 학습

> **핵심 발견**: "Even 1,000 examples suffice to train the model to be robust to irrelevant contexts."

### 4.7 LLM-Embedder: 통합 검색 모델

- **저자**: Peitian Zhang, Shitao Xiao, Zheng Liu et al.
- **arXiv**: 2310.07554

LLM의 RAG를 위한 **통합 임베딩 모델**:
- **Rank-aware Reward**: LLM 출력에서 정답의 랭킹 위치를 보상으로 활용
- **Graded Distillation**: 보상 값과 상대 순서를 모두 반영한 증류 목적함수
- **Multi-task Optimization**: QA, 대화, 코드 등 다양한 검색 기능을 단일 모델로 통합

---

## 5. 임베딩 모델 도메인 파인튜닝

### 5.1 왜 임베딩 파인튜닝이 중요한가?

범용 임베딩 모델은 도메인 특화 용어, 약어, 관계를 제대로 포착하지 못한다. 예를 들어:
- 법률: "선의취득"과 "부당이득"의 관계
- 의료: "HbA1c"와 "당뇨 관리"의 관계
- 금융: "EBITDA"와 "수익성 지표"의 관계

### 5.2 BGE / C-Pack 다단계 학습 파이프라인

- **저자**: BAAI (Beijing Academy of AI)
- **arXiv**: 2309.07597 (C-Pack), 학회: SIGIR 2024

#### 3단계 학습

```
Stage 1: RetroMAE 사전학습
  ↓  (비지도, 마스크 오토인코더)
Stage 2: 대규모 약지도(Weak Supervision) 대조학습
  ↓  (수억 쌍, 크로스 인코더 증류)
Stage 3: 소규모 고품질 태스크별 파인튜닝
  ↓  (명령어 기반 임베딩)
최종 모델
```

- C-MTEB에서 기존 중국어 임베딩 대비 **+10%**
- 영어 MTEB에서도 SOTA

### 5.3 E5-Mistral: LLM으로 합성 학습 데이터 생성

- **저자**: Liang Wang, Nan Yang et al. (Microsoft)
- **arXiv**: 2401.00368
- **학회**: ACL 2024

#### 핵심 혁신: 합성 데이터만으로 SOTA

1. **대규모 합성 데이터 생성**: 프로프라이어터리 LLM으로 수십만 개의 임베딩 태스크 생성 (93개 언어)
2. **최소 학습**: 1,000 스텝 미만의 학습으로 경쟁력 있는 결과
3. **Decoder-only LLM 파인튜닝**: 오픈소스 decoder-only LLM에 표준 대조 손실로 파인튜닝
4. **합성 + 라벨 데이터 혼합** 시 BEIR, MTEB에서 **새로운 SOTA**

### 5.4 GritLM: 생성 + 임베딩 통합 모델

- **저자**: Niklas Muennighoff, Hongjin Su et al.
- **arXiv**: 2402.09906

#### Generative Representational Instruction Tuning (GRIT)

**단일 모델**이 생성과 임베딩을 동시에 수행:
- 명령어(instruction)로 생성 vs 임베딩 태스크를 구분
- 생성 전용 학습과 임베딩 전용 학습 **각각의 성능 손실 없이** 통합
- RAG에서 **별도 임베딩 모델 불필요** → 긴 문서에서 **60% 이상 속도 향상**

| 변종 | MTEB | 생성 |
|------|------|------|
| GritLM 7B | 오픈 모델 SOTA | 동급 최고 |
| GritLM 8x7B | 최상위 | 오픈 생성 모델 능가 |

### 5.5 실무: 도메인 임베딩 파인튜닝 파이프라인

#### Step 1: 합성 학습 데이터 생성

```python
# GPT-4를 사용한 (query, positive, negative) 트리플렛 생성
PROMPT = """
다음 문서를 읽고, 이 문서를 검색하기 위한 자연스러운 질문 3개를 생성하시오.
각 질문에 대해:
1. 이 문서가 정답인 질문 (positive)
2. 유사하지만 이 문서로는 답할 수 없는 질문 (hard negative)

문서: {document}
"""
```

#### Step 2: Hard Negative Mining

```python
from sentence_transformers.util import mine_hard_negatives

# BM25 + 임베딩 크로스 검색으로 하드 네거티브 채굴
# 의미적으로 유사하지만 정답이 아닌 문서를 선별
hard_negatives = mine_hard_negatives(
    dataset=train_dataset,
    model=base_embedding_model,
    num_negatives=5,
)
```

#### Step 3: 대조 학습

```python
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# MultipleNegativesRankingLoss: in-batch 네거티브 활용
loss = losses.MultipleNegativesRankingLoss(model)
# 또는 CachedMultipleNegativesRankingLoss: 더 큰 유효 배치 크기

args = SentenceTransformerTrainingArguments(
    output_dir="domain-embeddings",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
    evaluator=InformationRetrievalEvaluator(...),
)
trainer.train()
```

#### 주요 손실 함수 선택 가이드

| 손실 함수 | 데이터 형태 | 용도 |
|----------|-----------|------|
| **MultipleNegativesRankingLoss** | (query, positive) | 검색/랭킹, in-batch 네거티브 |
| **TripletLoss** | (anchor, positive, negative) | 명시적 네거티브 제공 시 |
| **CosineSimilarityLoss** | (text1, text2, score) | 유사도 점수 있을 때 |
| **BatchHardTripletLoss** | (text, label) | 배치 내 하드 네거티브 자동 선택 |
| **ContrastiveLoss** | (text1, text2, 0/1) | 이진 유사도 판단 |

---

## 6. DPO/RLHF for RAG Faithfulness

### 6.1 배경: 왜 RAG에 선호도 최적화가 필요한가?

RAG 시스템에서 LLM은 검색된 문맥에 **충실한(faithful)** 답변을 생성해야 하지만, 종종:
- 검색 문맥을 무시하고 사전학습 지식으로 답변 (over-reliance on parametric knowledge)
- 검색 문맥에 없는 내용을 **날조** (hallucination)
- 문맥의 잘못된 정보를 무비판적으로 수용 (over-reliance on context)

ClashEval (Wu et al., 2024, arXiv 2404.10198) 연구에 따르면:
> "LLMs override their own correct prior knowledge over 60% of the time" when presented with incorrect retrieved content.
> 모델의 초기 응답 **신뢰도(token probability)**가 낮을수록 검색 컨텍스트를 더 쉽게 수용.

### 6.2 DPO (Direct Preference Optimization)

- **원 논문**: Rafailov et al. (2023)
- **SimPO 개선**: Yu Meng, Mengzhou Xia, Danqi Chen (NeurIPS 2024, arXiv 2405.14734)

#### DPO의 핵심: 보상 모델 없는 선호도 학습

기존 RLHF 파이프라인:
```
SFT → 선호도 주석 → 보상 모델 학습 → RL 최적화 (PPO)
```

DPO 파이프라인:
```
SFT → 선호도 주석 → 직접 선호도 최적화 (DPO 손실)
```

보상 함수에서 최적 RL 정책으로의 **해석적 매핑**을 이용하여, 보상 모델과 RL을 **단일 지도 손실**로 대체.

#### RAG용 DPO 데이터 생성

```python
# RAG 충실도 선호도 쌍 생성
def create_rag_dpo_data(query, context, model):
    # Chosen: 문맥에 충실한 응답
    chosen = generate_faithful_response(query, context)
    
    # Rejected: 할루시네이션 포함 응답
    rejected = generate_with_hallucination(query, context)
    # 또는: 문맥 무시 응답, 사실 왜곡 응답 등
    
    return {
        "prompt": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:",
        "chosen": chosen,
        "rejected": rejected,
    }
```

#### DPO 학습 (QLoRA 호환)

```python
from trl import DPOTrainer
from transformers import AutoModelForCausalLM, AutoPeftModelForCausalLM

# SFT 모델을 base로 사용
model = AutoPeftModelForCausalLM.from_pretrained(
    sft_model_path,
    load_in_4bit=True,
    is_trainable=True,
)
model_ref = AutoPeftModelForCausalLM.from_pretrained(
    sft_model_path,
    load_in_4bit=True,
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=model_ref,
    beta=0.1,  # 온도: 0.1-0.5, 레퍼런스 모델 의존도 조절
    train_dataset=rag_preference_dataset,
    tokenizer=tokenizer,
    args=training_args,
)
dpo_trainer.train()
```

#### 핵심 학습 지표

| 지표 | 설명 | 목표 |
|------|------|------|
| rewards/chosen | 선호 응답의 로그 확률 차이 | 높을수록 좋음 |
| rewards/rejected | 비선호 응답의 로그 확률 차이 | 낮을수록 좋음 |
| rewards/accuracies | chosen > rejected 빈도 | → 1.0 |
| rewards/margins | chosen과 rejected 보상 차이 | 양수, 증가 추세 |

### 6.3 SimPO: 참조 모델 없는 개선된 DPO

- **저자**: Yu Meng, Mengzhou Xia, Danqi Chen
- **학회**: NeurIPS 2024

시퀀스의 **평균 로그 확률**을 암묵적 보상으로 사용:
- 참조 모델 **불필요** → 계산 효율 향상
- 목표 보상 마진으로 선호/비선호 구분 강화
- DPO 대비 AlpacaEval 2에서 **+6.4점**, Arena-Hard에서 **+7.5점**

### 6.4 RAG 충실도를 위한 DPO 선호도 데이터 유형

| 유형 | Chosen (선호) | Rejected (비선호) |
|------|-------------|-----------------|
| **문맥 충실도** | 검색 문맥에 근거한 답변 | 문맥에 없는 정보 포함 답변 |
| **인용 정확도** | 정확한 출처 인용 | 잘못된 출처 또는 인용 없음 |
| **답변 불가 인식** | "제공된 문맥에서 답을 찾을 수 없습니다" | 근거 없이 답변 생성 |
| **사실 정확도** | 문맥의 사실을 정확히 반영 | 문맥 내용을 왜곡/과장 |

---

## 7. Knowledge Distillation for Domain RAG

### 7.1 왜 지식 증류가 필요한가?

프로덕션 RAG에서의 현실:
- **대형 모델** (70B+): 최고 성능이지만 추론 비용이 높음
- **소형 모델** (7B-13B): 비용 효율적이지만 RAG 활용 능력이 부족

> 목표: 대형 모델의 RAG 행동을 소형 모델에 증류하여, **비용은 소형 모델 수준, 성능은 대형 모델 근접**을 달성.

### 7.2 Teacher-Student 접근법

#### 기본 파이프라인

```
[교사 모델 (70B)] + [검색 시스템] → 고품질 (query, context, response) 생성
                                          ↓
                               교사의 응답으로 학생 모델 SFT
                                          ↓
                              [학생 모델 (7B)] — 교사 수준 RAG 능력
```

#### 데이터 증류 전략

1. **Response Distillation**: 교사 모델의 RAG 응답을 학생 모델의 SFT 데이터로 사용
2. **Reasoning Distillation**: 교사의 CoT 추론 과정을 포함하여 증류
3. **Reflection Distillation**: Self-RAG의 반성 토큰 판단까지 증류

### 7.3 InstructRetro 접근법

NVIDIA의 InstructRetro (arXiv 2310.07713)는 다른 방향의 증류를 보여준다:

1. 대형 모델(43B)에 **검색 증강 사전학습** 수행 (Retro 방식)
2. Instruction tuning 후 **인코더를 제거**해도 성능 유지
3. 검색 증강 사전학습의 **지식이 디코더 가중치에 내재화**됨을 시사

> 이는 검색 증강 학습이 일종의 **자체 증류** 역할을 한다는 것을 보여준다.

### 7.4 실무 증류 파이프라인

```python
# Step 1: 교사 모델로 RAG 응답 생성
teacher_model = load_model("meta-llama/Llama-3-70B-Instruct")
retriever = load_retriever("domain_index")

distillation_data = []
for query in domain_queries:
    contexts = retriever.search(query, top_k=5)
    response = teacher_model.generate(
        prompt=format_rag_prompt(query, contexts),
        temperature=0.7,
    )
    distillation_data.append({
        "query": query,
        "context": contexts,
        "response": response,
    })

# Step 2: 학생 모델 SFT (QLoRA)
student_model = load_model("meta-llama/Llama-3-8B-Instruct", quantize=True)
train_with_qlora(student_model, distillation_data)
```

---

## 8. 실무 파이프라인: 도메인 RAG LLM 구축 단계별 가이드

### 8.1 의사결정 트리

```
도메인 RAG LLM이 필요한가?
  │
  ├─ 일반 LLM + RAG로 충분한가?
  │    ├─ YES → 프롬프트 엔지니어링으로 시작 (비용: $0)
  │    └─ NO ↓
  │
  ├─ 검색 품질이 문제인가?
  │    ├─ YES → 임베딩 모델 파인튜닝 (§5)
  │    └─ NO ↓
  │
  ├─ 모델이 검색 결과를 잘 활용 못하는가?
  │    ├─ YES → RAFT (§3) 또는 SFT (§2)
  │    └─ NO ↓
  │
  ├─ 할루시네이션이 문제인가?
  │    ├─ YES → DPO (§6) 또는 Self-RAG (§4.1)
  │    └─ NO ↓
  │
  ├─ 도메인 용어/개념 이해가 부족한가?
  │    ├─ YES → Continued Pre-training (§1)
  │    └─ NO ↓
  │
  └─ 추론 비용이 문제인가?
       └─ YES → Knowledge Distillation (§7)
```

### 8.2 권장 순서 (투자 대비 효과 순)

| 순서 | 단계 | 예상 기간 | 비용 | ROI |
|------|------|----------|------|-----|
| 1 | 프롬프트 엔지니어링 + 고급 RAG 기법 | 1-2주 | $0 | ★★★★★ |
| 2 | 임베딩 모델 도메인 파인튜닝 | 1-2주 | GPU 수시간 | ★★★★☆ |
| 3 | RAFT (검색 문맥 활용 SFT) | 2-4주 | GPU 수일 | ★★★★☆ |
| 4 | DPO (충실도 선호도 최적화) | 1-2주 | GPU 수일 | ★★★☆☆ |
| 5 | Self-RAG (적응적 검색 + 반성) | 4-6주 | GPU 수주 | ★★★☆☆ |
| 6 | Knowledge Distillation | 2-4주 | GPU 수일 | ★★★☆☆ |
| 7 | Continued Pre-training | 4-8주 | GPU 수백시간+ | ★★☆☆☆ |

### 8.3 흔한 실수

| 실수 | 올바른 접근 |
|------|-----------|
| CPT부터 시작 | 프롬프트/RAG 파이프라인 최적화가 선행 |
| SFT 데이터 양에 집중 | LIMA: 1,000개 고품질 > 10,000개 저품질 |
| 범용 임베딩 사용 고집 | 도메인 파인튜닝으로 검색 정확도 5-15% 향상 |
| 할루시네이션을 SFT로 해결 시도 | DPO/Self-RAG가 충실도 문제에 더 효과적 |
| Retriever 무시 | RA-DIT: LLM + Retriever 동시 최적화가 최선 |
| 무관 문서 처리 무시 | RAFT의 distractor 학습 / mixed context 학습 |
| 일반 데이터 혼합 생략 (CPT) | Catastrophic forgetting → 일반 능력 상실 |

### 8.4 비용 추정 (2026년 기준)

| 접근법 | 7B 모델 | 70B 모델 | 필요 GPU |
|--------|--------|---------|---------|
| **SFT (QLoRA)** | 4-8 GPU-hours | 24-48 GPU-hours | 1x A100 48GB |
| **RAFT** | 8-16 GPU-hours | 48-96 GPU-hours | 1x A100 48GB |
| **DPO** | 4-8 GPU-hours | 24-48 GPU-hours | 1x A100 48GB |
| **Self-RAG** | 16-32 GPU-hours | 96-192 GPU-hours | 4x A100 |
| **RA-DIT** | 24-48 GPU-hours | 192-384 GPU-hours | 8x A100 |
| **CPT** | 100-500 GPU-hours | 1000-5000 GPU-hours | 8-64x A100 |
| **임베딩 FT** | 1-4 GPU-hours | - | 1x A100 |

---

## 9. 평가: 도메인 RAG LLM 검증

### 9.1 RAGAS 프레임워크

- **논문**: Shahul Es et al., arXiv 2309.15217
- **특징**: **참조 답변 없이** RAG 파이프라인 평가 가능

| 메트릭 | 측정 대상 | 방법 |
|--------|---------|------|
| **Faithfulness** | 답변이 검색 문맥에 충실한가? | LLM이 답변의 각 주장이 문맥에서 지지되는지 판단 |
| **Answer Relevance** | 답변이 질문에 관련 있는가? | 답변에서 역 질문 생성 → 원 질문과 유사도 측정 |
| **Context Precision** | 검색된 문맥 중 관련 비율 | 상위 문맥의 관련성 순서 평가 |
| **Context Recall** | 필요한 정보가 검색되었는가? | Ground truth 답변의 각 문장이 문맥에서 지지되는지 |

### 9.2 ARES (Automated RAG Evaluation System)

- **저자**: Jon Saad-Falcon et al.
- **arXiv**: 2311.09476
- **학회**: NAACL 2024

3개 차원으로 평가:
1. **Context Relevance**: 검색 문맥의 관련성
2. **Answer Faithfulness**: 답변의 문맥 충실도
3. **Answer Relevance**: 답변의 질문 관련성

#### 핵심 혁신
- 합성 학습 데이터로 **경량 LLM 심판** 파인튜닝
- **Prediction-Powered Inference (PPI)**: 소량의 인간 주석(수백 개)으로 예측 오류 보정
- 도메인 전이에도 **강건한** 판단 성능

> KILT, SuperGLUE, AIS 등 8개 지식 집약 태스크에서 검증.

### 9.3 도메인 RAG 평가 체크리스트

```
□ 검색 품질
  ├─ Recall@K: 상위 K 문서에 정답 포함 비율
  ├─ MRR (Mean Reciprocal Rank): 정답 문서의 평균 역순위
  └─ NDCG: 순서 고려한 관련성 점수

□ 생성 품질
  ├─ Faithfulness: 문맥에 충실한 답변 비율
  ├─ Groundedness: 주장별 근거 존재 여부
  ├─ Answer Accuracy: 도메인 전문가 평가
  └─ Hallucination Rate: 근거 없는 주장 비율

□ 엔드투엔드
  ├─ Exact Match / F1: 정답과의 일치도
  ├─ BERT-Score: 의미적 유사도
  ├─ 도메인 전문가 블라인드 평가
  └─ A/B 테스트 (기존 시스템 vs 새 시스템)

□ 운영 지표
  ├─ 응답 지연 (TTFT + 생성 시간)
  ├─ 처리량 (QPS)
  └─ 비용 ($/1K queries)
```

---

## 10. 최신 동향 (2025-2026)

### 10.1 합성 데이터 생성의 부상

E5-Mistral (Wang et al., ACL 2024)이 보여준 패러다임:
- **프로프라이어터리 LLM**으로 수십만 개의 학습 태스크 합성
- **1,000 스텝 미만**의 학습으로 BEIR/MTEB SOTA
- 93개 언어에 걸친 다국어 임베딩

이 접근법은 모든 RAG 컴포넌트에 적용 가능:
- **임베딩 학습 데이터**: LLM으로 (query, document) 쌍 생성
- **SFT 데이터**: LLM으로 (instruction, context, response) 트리플렛 생성
- **DPO 데이터**: LLM으로 chosen/rejected 쌍 생성
- **평가 데이터**: LLM으로 도메인 QA 벤치마크 생성

### 10.2 통합 모델의 시대

GritLM이 보여준 방향:
- **하나의 모델**이 임베딩(검색)과 생성을 동시에 수행
- RAG 파이프라인에서 **별도 임베딩 모델 불필요**
- 긴 문서 처리 시 **60% 이상 속도 향상**
- 2026 트렌드: Jina Embeddings v4, Qwen3-Embedding 등도 유사 방향

### 10.3 멀티모달 RAG 임베딩

2026년 최신 임베딩 모델들은 **텍스트 + 이미지 + PDF**를 동시 처리:
- **Gemini Embedding 2**: 텍스트/이미지/영상/오디오/PDF 5개 모달리티
- **Jina Embeddings v4**: Qwen2.5-VL 기반, 3개 LoRA 어댑터로 태스크 특화
- **Cohere Embed v4**: 멀티모달, 128K 컨텍스트

### 10.4 Parameter-Efficient 접근법의 진화

| 기법 | 특징 | 적용 |
|------|------|------|
| **LoRA** | 저랭크 행렬 분해 | 기본 |
| **QLoRA** | 4비트 양자화 + LoRA | 메모리 제한 환경 |
| **DoRA** | 방향/크기 분리 LoRA | 더 나은 수렴 |
| **LoRA+** | 학습률 분리 (A/B 행렬) | 빠른 수렴 |

### 10.5 QuIM-RAG: 질문 역인덱싱

- **논문**: IEEE Access, 2024 (arXiv 2501.02702)

문서 청크에서 **잠재적 질문을 미리 생성**하고, 사용자 질의와 생성된 질문을 매칭:
- 기존 벡터 검색의 의미 갭 해소
- 전통적 RAG 대비 BERT-Score, RAGAS 모두 향상

### 10.6 Chain-of-Note (CoN)

- **저자**: Wenhao Yu et al.
- **arXiv**: 2311.09210
- **학회**: EMNLP 2024

검색된 문서에 대한 **순차적 읽기 노트**를 생성하여 관련성 평가:
- 노이즈 검색 시 **+7.9 EM score** 향상
- 답변 불가 인식: **+10.5%** rejection rate 향상
- LLaMa-2 7B + ChatGPT 생성 학습 데이터

---

## 11. 종합 비교 테이블

| 접근법 | 논문 | 데이터 요구 | 컴퓨트 | 주요 효과 | 적용 시점 |
|--------|------|-----------|--------|----------|----------|
| **CPT (DAPT)** | Gururangan 2020 | 수십GB+ 도메인 텍스트 | $$$$ | 도메인 이해력 ↑ | 도메인 용어가 범용 LLM에 없을 때 |
| **SFT (LoRA)** | Hu et al. 2021 | 1K-10K 예시 | $$ | 지시 따르기 ↑ | 기본적인 도메인 적응 |
| **SFT (QLoRA)** | Dettmers 2023 | 1K-10K 예시 | $ | LoRA와 동등, 메모리 절감 | GPU 제한 환경 |
| **RAFT** | Zhang et al. 2024 | 1K-5K QA + 문서 | $$ | 검색 문맥 활용 ↑ | 도메인 문서 집합이 고정된 RAG |
| **Self-RAG** | Asai et al. 2023 | 150K+ 반성 토큰 데이터 | $$$ | 적응적 검색, 자기 검증 ↑ | 검색 품질이 변동적인 환경 |
| **RA-DIT** | Lin et al. 2023 | 다중 태스크 데이터 | $$$ | LLM + Retriever 동시 최적화 | 최고 성능 추구 |
| **ChatQA** | NVIDIA 2024 | 2단계 SFT 데이터 | $$$ | 대화형 RAG QA ↑ | 대화형 도메인 QA |
| **DPO** | Rafailov 2023 | 1K-10K 선호도 쌍 | $$ | 충실도/할루시네이션 ↓ | 할루시네이션이 주요 문제일 때 |
| **임베딩 FT** | BGE/E5 | 10K-100K 쌍 | $ | 검색 정확도 ↑ | 도메인 용어 검색이 부정확할 때 |
| **Knowledge Distill** | InstructRetro | 교사 응답 데이터 | $$$ | 소형 모델로 성능 유지 | 추론 비용 최적화 |

---

## 12. 결론: 최적 전략 선택

### RAG 시스템에서 모델 최적화의 우선순위

1. **먼저 RAG 파이프라인 자체를 최적화**: 청킹, 검색, 리랭킹, 프롬프트 설계
2. **임베딩 모델 도메인 적응**: 가장 비용 대비 효과가 높은 모델 개선
3. **RAFT 또는 Context-aware SFT**: 모델이 검색 결과를 더 잘 활용하도록 교육
4. **DPO**: 충실도가 중요한 경우 추가 적용
5. **CPT**: 위 모든 것이 부족하고, 도메인 언어 자체의 이해가 필요할 때만

> **핵심 원칙**: RAG의 근본 목적은 **외부 지식 검색으로 LLM의 지식 한계를 보완**하는 것이다. 따라서 모델의 지식을 늘리는 것(CPT)보다, 모델이 **검색된 정보를 정확하게 활용하는 능력**을 향상시키는 것(RAFT, Self-RAG, DPO)이 더 효율적이다.

---

## 참고 문헌

1. Gururangan et al., "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks," ACL 2020. arXiv:2004.10964
2. Wu et al., "BloombergGPT: A Large Language Model for Finance," arXiv:2303.17564
3. Roziere et al., "Code Llama: Open Foundation Models for Code," arXiv:2308.12950
4. Singhal et al., "Towards Expert-Level Medical Question Answering with Large Language Models," arXiv:2305.09617
5. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022. arXiv:2106.09685
6. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," NeurIPS 2023. arXiv:2305.14314
7. Zhou et al., "LIMA: Less Is More for Alignment," NeurIPS 2023. arXiv:2305.11206
8. Zhang et al., "RAFT: Adapting Language Model to Domain Specific RAG," arXiv:2403.10131
9. Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection," ICLR 2024. arXiv:2310.11511
10. Lin et al., "RA-DIT: Retrieval-Augmented Dual Instruction Tuning," arXiv:2310.01352
11. NVIDIA, "ChatQA: Surpassing GPT-4 on Conversational QA and RAG," arXiv:2401.10225
12. Muennighoff et al., "GritLM: Generative Representational Instruction Tuning," arXiv:2402.09906
13. Wang et al., "Improving Text Embeddings with Large Language Models," ACL 2024. arXiv:2401.00368
14. BAAI, "C-Pack: Packed Resources For General Chinese Embeddings," SIGIR 2024. arXiv:2309.07597
15. Zhang et al., "Retrieve Anything To Augment Large Language Models," arXiv:2310.07554
16. Rafailov et al., "Direct Preference Optimization," 2023
17. Meng et al., "SimPO: Simple Preference Optimization with a Reference-Free Reward," NeurIPS 2024. arXiv:2405.14734
18. NVIDIA, "InstructRetro: Instruction Tuning for Retrieval-Augmented LLMs," ICML 2024. arXiv:2310.07713
19. Shahul Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation," arXiv:2309.15217
20. Saad-Falcon et al., "ARES: An Automated Evaluation Framework for RAG Systems," NAACL 2024. arXiv:2311.09476
21. Yoran et al., "Making Retrieval-Augmented Language Models Robust to Irrelevant Context," arXiv:2310.01558
22. Gekhman et al., "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?" arXiv:2405.05904
23. Yu et al., "Chain-of-Note: Enhancing Robustness in RALMs," EMNLP 2024. arXiv:2311.09210
24. Yan et al., "Corrective Retrieval Augmented Generation," arXiv:2401.15884
25. Wu et al., "ClashEval: Quantifying the tug-of-war between LLM's internal prior and external evidence," arXiv:2404.10198
