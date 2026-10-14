---
title: "[서베이] RAG 검색 속도를 획기적으로 줄이는 방법들 — KV 캐시, 서브밀리초 인덱스, 예측적 프리페칭"
date: 2026-04-03
tags: ["연구노트", "RAG", "LLM", "최적화", "검색"]
categories: ["ML/AI"]
summary: "RAG의 검색·추론 지연을 극적으로 줄이는 최신 기법 20+개를 정리한다. TurboRAG(9.4배), CacheBlend(EuroSys Best Paper), Seismic(105배), VoiceAgentRAG(316배)까지."
math: true
toc: true
draft: false
---

<!-- LEARNING-START -->
## 처음 읽는 분을 위한 안내

**학습 위치:** [자료를 찾아 답하는 AI]({{< relref "learn" >}}#rag) · 13/17

답변을 기다리는 시간에는 자료를 찾는 시간, 읽을 입력을 준비하는 시간, 글을 만드는 시간이 섞여 있다. 이 글은 각 단계의 시간을 줄이는 방법을 비교한다. 먼저 어느 부분이 느린지 나눠 봐야 무엇을 고칠지 정할 수 있다.

**먼저 알아둘 말**

- [지연·처리량]({{< relref "ai-glossary" >}}#latency): 지연은 요청 하나의 결과를 기다리는 시간이고, 처리량은 일정 시간에 처리하는 작업의 양이다. 둘은 같은 지표가 아니다.
- [KV 캐시·프리필]({{< relref "ai-glossary" >}}#kv): KV 캐시는 Attention에 쓸 중간 계산 결과를 저장한 것이다. 프리필은 주어진 입력 문맥을 먼저 처리하는 단계다.
- [검색기·리랭킹]({{< relref "ai-glossary" >}}#retriever): 검색기는 후보 문서를 찾는다. 리랭킹은 그 후보를 다시 평가해 질문에 더 관련 있는 순서로 정렬한다.

**이번 글의 목표:** 검색·입력 처리·생성 중 느린 단계 찾기.

**이 학습 경로의 앞 글:** [\[논문 리뷰\] OPQE — HyDE에 강화학습을 결합한 On-Policy Pseudo-document Query Expansion]({{< relref "posts/opqe-review" >}}).

**본문에서 먼저 볼 순서:** **6. 아키텍처 수준 접근**.

첫 독서에서는 문제와 방법의 흐름을 잡자. 수식·코드·세부 성능표는 두 번째에 읽어도 된다. 구현과 수학적 주장까지 확인하려면 그 부분을 다시 살펴봐야 한다.

<!-- PAPER-PREREQUISITES -->
### 선수 개념과 읽을 논문

**먼저 이해할 개념:** 벡터의 유사도, 관련/무관 예시의 비교, 검색과 생성의 구분.

아래 순서는 이 글을 이해하기 위한 공부 순서다. ‘개념 준비’는 읽기 안내이며, 직접적인 연구 계보는 연결 칸에 따로 표시했다. 이미 해당 개념을 설명할 수 있다면 아는 부분은 넘어가도 된다.

| 순서·논문 | 연결·익힐 개념 | 먼저 볼 부분 |
|---|---|---|
| 1. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) · [블로그 해설]({{< relref "posts/rag-survey-2026" >}}) | **개념 준비**: 검색기·생성기·외부 지식 | 질문에서 문서를 검색하고 문서와 함께 답을 만드는 구성 |
| 2. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | **개념 준비**: Attention·위치 정보·인코더/디코더 | 현재 토큰이 다른 위치의 정보를 읽는 구조 그림과 Attention 설명 |
| 3. [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832) · [블로그 해설]({{< relref "posts/retriever-training-survey" >}}) | **개념 준비**: 토큰별 표현·late interaction | 전체 문장 벡터 하나 대신 토큰 표현들을 비교하는 구조 |

**현재 글과 연결하기:** 위 개념을 익힌 뒤, 이 글에서 ‘검색·입력 처리·생성 중 느린 단계 찾기’에 해당하는 부분을 읽자. 기존 방법의 어느 단계가 바뀌었는지 그림이나 입력·출력으로 설명해 보자.

[선수 논문 전체 지도와 원문 읽는 법]({{< relref "paper-prerequisites" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- PAPER-PREREQUISITES-END -->

<!-- LEARNING-START-END -->

---

## 한눈에 보는 속도 향상 비교

| 카테고리 | 기법 | 속도 향상 | 학회 |
|---------|------|---------|------|
| KV 캐시 사전 계산 | **TurboRAG** | TTFT 9.4배↓ | EMNLP 2025 |
| KV 캐시 스티칭 | **CacheBlend** | TTFT 3.3배↓, 처리량 5배↑ | EuroSys 2025 **Best Paper** |
| 검색 자체 제거 | **CAG** | 검색 시간 = 0 | ACM Web 2025 |
| 예측적 프리페칭 | **VoiceAgentRAG** | 캐시 히트 시 **316배** | Salesforce, 2026.03 |
| 학습된 역인덱스 | **Seismic** | **84-105배** | SIGIR 2024 준최우수 |
| Late Interaction | **PLAID** | GPU 7배 / CPU 45배 | CIKM 2022 |
| 바이너리 양자화 | 1-bit + 리스코어 | **25-40배** | 다수 |
| GPU ANN | **CAGRA** | 인덱스 빌드 10-30배 | NVIDIA, 2023 |
| 점진적 차원 검색 | **Matryoshka** | **14배** | NeurIPS 2022 |
| 병렬 초안+검증 | **Speculative RAG** | 지연 51%↓ | Google, 2024 |
| 희소 컨텍스트 선택 | **Sparse RAG** | 디코딩 2-3배↑ | ICLR 2025 |
| 다중 레벨 캐싱 | **RAGCache** | TTFT 4배↓ | ACM ToCS, 2024 |

표의 배속은 각각 무엇을 측정했는지와 함께 읽어야 한다. 검색 시간이 줄어든 경우, 프리필이 빨라진 경우, 전체 응답이 빨라진 경우는 서로 다른 결과다. 아래 기법을 적용할 때도 먼저 요청의 시간을 검색·자료 준비·프리필·생성으로 나눠 보면 된다. 전체에서 작은 비중을 차지하는 단계를 크게 줄여도 사용자가 느끼는 시간 변화는 그보다 작을 수 있다.

---

## 1. KV 캐시 사전 계산 — 프리필을 없애다

### 1.1 TurboRAG (EMNLP 2025)

- **논문**: "TurboRAG: Accelerating RAG with Precomputed KV Caches for Chunked Text"
- **저자**: Songshuo Lu, Hua Wang et al.
- **핵심 결과**: TTFT(Time To First Token) **최대 9.4배** 감소 (4.13s → 0.65s)

**아이디어**: 오프라인에서 각 청크의 KV 캐시를 미리 계산해 저장. 추론 시에는 검색된 청크의 KV 캐시를 **로드만** 하면 된다 — 프리필 연산 자체를 건너뛴다.

```
=== 기존 RAG ===
질의 → 검색 → [청크들을 프롬프트에 삽입] → 프리필(느림!) → 생성

=== TurboRAG ===
[오프라인] 각 청크 → KV 캐시 계산 → 저장

[온라인]  질의 → 검색 → KV 캐시 로드 → 바로 생성!
```

**핵심 기술**:
- **Independent Attention**: 각 청크의 KV 캐시가 다른 청크에 독립적으로 계산
- **Reordered RoPE**: 청크 순서가 바뀌어도 위치 인코딩이 올바르게 작동하도록 재정렬
- 모델 아키텍처 수정 **불필요**

**트레이드오프**: KV 캐시 저장 공간 추가 필요 (공간↔시간 트레이드오프).

### 1.2 CacheBlend (EuroSys 2025 Best Paper)

- **논문**: "CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion"
- **저자**: Jiayi Yao, Hanchen Li et al.
- **핵심 결과**: TTFT **2.2-3.3배**↓, 처리량 **2.8-5배**↑

**기존 문제**: 프롬프트 캐싱은 **접두사(prefix) 일치**에서만 작동. RAG에서는 검색 결과가 프롬프트 중간에 삽입되므로 캐시 히트가 불가능.

**CacheBlend의 해결**: 위치와 무관하게 사전 계산된 KV 캐시를 재사용하되, **소수의 토큰만 선택적으로 재계산**하여 청크 간 어텐션을 복원.

$$\text{재계산 비율} \ll \text{전체 토큰 수} \implies \text{거의 프리필 없이 KV 캐시 재사용}$$

**후속 연구 — CacheClip (2025.10)**:
- 보조 소형 LLM이 어떤 토큰을 재계산할지 식별
- 프리필 **1.92배** 추가 가속, NIAH 과제에서 CacheBlend 대비 35.1% 개선

### 1.3 CAG — 검색 자체를 없앤다 (ACM Web 2025)

- **논문**: "Don't Do RAG: When Cache-Augmented Generation is All You Need"
- **저자**: Cheng-Yu Hsieh, Yun-Nung Chen et al.

**아이디어**: 전체 지식 베이스를 LLM의 긴 컨텍스트 윈도우에 **미리 전부 넣고** KV 캐시를 오프라인 계산. 추론 시 검색 = 0.

```
[오프라인] 전체 문서 → LLM 프리필 → KV 캐시 저장

[온라인]  질의 → KV 캐시 로드 → 바로 생성 (검색 없음!)
```

**제약**: 지식 베이스가 컨텍스트 윈도우에 들어가야 함. 정적이고 범위가 한정된 도메인(제품 매뉴얼, 사내 정책)에 최적.

---

## 2. 검색 인덱스 가속 — 서브밀리초를 향해

### 2.1 Seismic (SIGIR 2024 준최우수 논문상)

- **논문**: "Efficient Inverted Indexes for Approximate Retrieval over Learned Sparse Representations"
- **저자**: Sebastian Bruch, Franco Maria Nardini et al.
- **핵심 결과**: 기존 대비 **84-105배** 빠름. MS MARCO에서 **서브밀리초** 달성 (단일 스레드!)

**아이디어**: 역인덱스의 각 리스트를 **기하학적으로 응집된 블록**으로 구성하고 블록마다 요약 벡터를 배치. 쿼리 시 요약 벡터로 블록 단위 가지치기 → 살아남은 블록만 개별 스코어링.

```
[기존 역인덱스]  쿼리 → 모든 포스팅 리스트 순회 → 느림

[Seismic]       쿼리 → 요약 벡터로 블록 가지치기 → 소수 블록만 스코어링 → 빠름!
```

### 2.2 PLAID / ColBERTv2 (CIKM 2022)

- **핵심 결과**: GPU **7배** / CPU **45배** 속도 향상. 1.4억 패시지에서 수십 ms.

문서를 **중심점(centroid) 가방**으로 표현한다. 중심점 상호작용으로 점수가 낮은 문서를 먼저 제거하고 남은 소수 문서에만 전체 late interaction을 적용한다.

### 2.3 바이너리 양자화 — 32배 압축, 25-40배 속도

| 수준 | 저장 절감 | 속도 향상 | 품질 손실 |
|------|---------|---------|---------|
| Binary (1-bit) | **32배** | **25-40배** | ~4% (리스코어링 없이) |
| Scalar (int8) | **4배** | ~4배 | < 1% |
| Float8 | **4배** | ~4배 | < 0.3% |

**3단계 파이프라인**:

$$\text{1-bit 이진 검색 (초고속)} \to \text{int8 리스코어링} \to \text{크로스인코더 리랭킹}$$

이진 단계에서 ~96% 검색 성능 유지. Qdrant, MongoDB Atlas, Azure AI Search에서 프로덕션 GA.

### 2.4 CAGRA — GPU 가속 ANN (NVIDIA)

- **핵심 결과**: 인덱스 빌드 **10-30배**, 검색 **4.7배**, 처리량 **8-18배** 향상

NVIDIA의 cuVS 라이브러리에 포함. Faiss, Milvus, Weaviate, OpenSearch, Elasticsearch에 통합 진행 중.

### 2.5 B+ANN (2025.11)

- HNSW보다 높은 리콜 **AND** QPS
- DiskANN 대비 메모리/빌드 시간 **24배** 절감
- B+ 트리 변형으로 시맨틱 블록 파티셔닝 → **캐시 미스 19.23% 감소**

### 2.6 Matryoshka 임베딩 — 점진적 차원 검색

NeurIPS 2022. 임베딩의 앞쪽 차원에 중요 정보를 집중시킨다. 2단계 검색:

```
[1단계] 64차원으로 후보 축소 (초고속)
[2단계] 768차원으로 후보 리랭킹 (정밀)
```

**14배** 벽시계 속도 향상, 리콜 손실 < 1%. OpenAI text-embedding-3에 적용.

---

## 3. 예측적 프리페칭 — 미래를 예측하여 검색

### 3.1 VoiceAgentRAG (Salesforce, 2026.03)

- **논문**: "VoiceAgentRAG: Solving the RAG Latency Bottleneck in Real-Time Voice Agents"
- **핵심 결과**: 캐시 히트 시 **316배** 속도 향상 (110ms → 0.35ms). 75% 히트율.

**듀얼 에이전트 아키텍처**:

```
[백그라운드] Slow Thinker
  대화 모니터링 → 후속 주제 예측 → 관련 청크 미리 검색 → 시맨틱 캐시 적재

[포그라운드] Fast Talker
  캐시에서만 읽음 → 서브밀리초 응답!
```

**한계**: 주제 전환 시 캐시 미스. 배경 GPU 컴퓨팅 필요. 대화 흐름이 일관된 경우 최적.

### 3.2 Speculative RAG (Google, 2024.07)

- 소형 전문 모델이 검색 문서 하위집합으로 여러 초안을 **병렬** 생성
- 대형 모델이 초안을 **한 번에** 검증·통합
- 지연 **51%** 감소, 정확도 **최대 12.97%** 향상

---

## 4. 컨텍스트 최적화 — 생성 단계 가속

### 4.1 Sparse RAG (ICLR 2025)

- **논문**: "Accelerating Inference of RAG via Sparse Context Selection"
- 20개 검색 문서 중 **4-8개만** 선택적으로 어텐딩
- 학습된 제어 토큰이 문서별 관련성 평가 → 불필요 문서 디코딩에서 제거
- 디코딩 **2-3배** 가속

### 4.2 RAGCache (ACM ToCS, 2024)

- 검색된 지식의 중간 KV 상태를 **지식 트리**로 구성
- GPU/호스트 메모리 계층에 걸쳐 캐싱
- TTFT **4배**↓, 처리량 **2.1배**↑
- 반복/중복 검색 패턴에서 최적

### 4.3 Proximity (EuroMLSys 2025)

- LSH 기반 **근사 쿼리 캐시** — 유사한 과거 쿼리의 검색 결과 재사용
- 검색 지연 **59%**↓, DB 호출 **77.2%** 감소
- 편향된(skewed) 워크로드에서 효과적

---

## 5. API 수준 프롬프트 캐싱

| 제공자 | 비용 절감 | 지연 절감 | TTL | 방식 |
|--------|---------|---------|-----|------|
| **Anthropic** | 90% | 85% | 5분 | 명시적 (`cache_control`) |
| **Google Gemini** | 75% | 자동 | 설정 가능 | 암묵적 (코드 변경 0) |
| **OpenAI** | 50% | 자동 | 자동 | 접두사 기반 자동 캐싱 |

공통 원리: 이전 요청과 접두사가 일치하면 KV 텐서를 재사용해 프리필 건너뜀.

**RAG에서의 활용**: 시스템 프롬프트 + 공통 지식을 접두사에 배치하면, 검색 결과만 달라지는 후반부만 새로 계산.

---

## 6. 아키텍처 수준 접근

### 6.1 RETRO — 검색 증강 사전학습 (ICML 2022)

- DeepMind. **7.5B RETRO가 175B GPT-3 성능**을 달성. 파라미터 **25배** 절약.
- 2T 토큰 데이터베이스에서 청크 단위 교차 어텐션으로 검색
- **InstructRetro** (NVIDIA): 48B까지 스케일 업. 장문 QA +10%, 요약 +16%.

### 6.2 MemoryFormer (2024.11)

- FC 레이어를 **인메모리 룩업 테이블**로 대체
- LSH로 입력을 메모리 주소에 매핑 → 사전 저장된 벡터 검색
- $O(d^2)$ 행렬곱 → $O(d)$ 룩업으로 FC FLOP **81% 감소**

### 6.3 NEST — kNN + Speculative Decoding (NeurIPS 2024)

- 토큰 수준 검색 + 추측적 디코딩
- Llama-2-Chat 70B에서 추론 **1.8배** 가속
- 소스 텍스트로의 **어트리뷰션** 자동 제공

---

## 7. 서브밀리초 달성 조건 정리

| 시스템 | 지연 | 조건 |
|--------|------|------|
| **Seismic** | < 1ms | 단일 스레드, SPLADE, MS MARCO |
| **VoiceAgentRAG** | 0.35ms | 캐시 히트 시 |
| **Redis 벡터** | < 1ms | 인메모리 인덱스 |
| **바이너리 양자화** | < 1ms | 인메모리, 중소규모 |
| **CAGRA (GPU)** | < 1ms | 배치=1, 중규모 |

**패턴**: 서브밀리초는 (a) 양자화된 인메모리 인덱스, (b) 시맨틱 캐시, (c) 학습된 역인덱스의 공격적 가지치기로 달성 가능. 수십억 규모에서는 한 자릿수 ms가 현실적 하한.

---

## 8. 실무 권장 조합

### 소규모 지식 베이스 (< 컨텍스트 윈도우)

```
CAG 방식: 전체 문서 → KV 캐시 사전 계산 → 검색 없이 바로 생성
```

### 중규모 프로덕션

```
[인덱싱] 청크 → 바이너리 양자화 임베딩 + SPLADE
         + TurboRAG/CacheBlend로 KV 캐시 사전 계산

[검색]   하이브리드 (Dense+Sparse) → RRF → 리랭커
         Matryoshka 2단계 검색 (64d → 768d)

[생성]   사전 계산된 KV 캐시 로드 → Sparse RAG로 불필요 문서 제거
         프롬프트 캐싱(Anthropic/Google) 활용
```

### 실시간 음성/채팅

```
VoiceAgentRAG 패턴:
  백그라운드 에이전트가 대화 흐름 예측 → 미리 검색 → 시맨틱 캐시
  포그라운드 에이전트는 캐시에서만 읽음 (0.35ms)
```

---

## 참고 자료

| 논문 | 핵심 | 학회 |
|------|------|------|
| TurboRAG (arXiv: 2410.07590) | KV 캐시 사전 계산, TTFT 9.4배↓ | EMNLP 2025 |
| CacheBlend (arXiv: 2405.16444) | 비접두사 KV 캐시 재사용 | EuroSys 2025 Best Paper |
| CacheClip (arXiv: 2510.10129) | 보조 LLM으로 재계산 토큰 식별 | 2025.10 |
| CAG (arXiv: 2412.15605) | 검색 제거, 전체 KV 캐시 | ACM Web 2025 |
| Seismic (arXiv: 2404.18812) | 학습된 역인덱스, 105배 | SIGIR 2024 |
| PLAID (arXiv: 2205.09707) | 중심점 기반 late interaction 가속 | CIKM 2022 |
| CAGRA (arXiv: 2308.15136) | GPU ANN, 10-30배 빌드 가속 | NVIDIA 2023 |
| B+ANN (arXiv: 2511.15557) | 디스크 기반 ANN, 메모리 24배↓ | 2025.11 |
| VoiceAgentRAG (arXiv: 2603.02206) | 예측적 프리페칭, 316배 | Salesforce 2026 |
| Speculative RAG (arXiv: 2407.08223) | 병렬 초안+검증, 51%↓ | ICLR 2025 |
| Sparse RAG (arXiv: 2405.16178) | 희소 컨텍스트 선택, 2-3배 | ICLR 2025 |
| RAGCache (arXiv: 2404.12457) | 다중 레벨 KV 캐싱 | ACM ToCS 2024 |
| Proximity (arXiv: 2503.05530) | 근사 쿼리 캐시, 59%↓ | EuroMLSys 2025 |
| Cache-Craft (arXiv: 2502.15734) | 캐시 재사용 품질 보장 | SIGMOD 2025 |
| Matryoshka (NeurIPS 2022) | 점진적 차원 검색, 14배 | NeurIPS 2022 |
| RETRO (arXiv: 2112.04426) | 검색 증강 사전학습, 25배 절약 | ICML 2022 |
| InstructRetro (arXiv: 2310.07713) | 48B RETRO, QA +10% | NVIDIA 2023 |
| MemoryFormer (arXiv: 2411.12992) | FC→룩업, FLOP 81%↓ | 2024.11 |
| NEST (arXiv: 2405.19325) | kNN+추측적 디코딩, 1.8배 | NeurIPS 2024 |

<!-- LEARNING-END -->
---

## 읽은 뒤 확인하고 다음으로

**확인 질문:** 검색·입력 처리·생성 중 느린 단계 찾기를 목표로, 본문에서 다룬 입력·중간 과정·결과를 자기 말로 설명해 보자. 실험이나 사례가 있다면 어떤 조건에서 확인했는지도 덧붙여 보자.

**다음 글:** [\[논문 리뷰\] TurboRAG — KV 캐시 사전 계산으로 RAG 추론 9.4배 가속]({{< relref "posts/turborag-review" >}}) — 문서 계산을 미리 해 두는 방법 이해하기.

[전체 학습 순서]({{< relref "learn" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- LEARNING-END-END -->
