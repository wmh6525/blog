---
title: "[코드 분석] DSF & EqProp — 비역전파 RNN 학습의 두 가지 접근"
date: 2026-03-20
tags: ["코드분석", "비역전파", "DSF", "EqProp"]
categories: ["ML/AI"]
summary: "DSF(대각 고정 피드백)와 EqProp-SeqLearning(평형전파 NLP) 레포를 소스코드 수준에서 분석한다. BPTT 대체를 위한 고정 피드백 행렬 구현과 에너지 기반 수렴 RNN + Hopfield 어텐션 구현을 추적한다."
math: true
toc: true
draft: false
---

<!-- LEARNING-START -->
## 처음 읽는 분을 위한 안내

**학습 위치:** [역전파를 바꾸면 무엇이 달라질까?]({{< relref "learn" >}}#no-bp) · 7/7

오차를 전달할 행렬을 고정해 쓰는 방법과, 모델이 안정된 상태에 도달한 뒤 상태 차이로 학습하는 방법을 비교한다. 목적은 비슷해도 계산 절차가 다르다. 코드를 보기 전에 두 방법의 반복 순서와 저장할 값을 나눠 적어 보자.

**먼저 알아둘 말**

- [그래디언트·역전파]({{< relref "ai-glossary" >}}#gradient): 그래디언트는 숫자를 어느 방향으로 바꾸면 손실이 달라지는지 나타낸다. 역전파는 연결된 계산을 거슬러 그 값을 구하는 방법이다.
- [Predictive Coding·예측 오차]({{< relref "ai-glossary" >}}#pc): 각 층의 예측과 현재 상태의 차이를 이용해 상태와 가중치를 조정하는 계산 방식. 상태 조정과 가중치 학습의 순서를 구분해 본다.
- [파라미터·가중치]({{< relref "ai-glossary" >}}#parameter): 모델 안에서 학습으로 조정하는 숫자. 모든 내부 숫자가 항상 학습 대상인 것은 아니다.

**이번 글의 목표:** 고정 피드백과 평형 상태 비교의 구현 차이 이해하기.

**이 학습 경로의 앞 글:** [\[코드 분석\] Prospective Configuration — PCLayer와 PCTrainer로 구현하는 비역전파 학습]({{< relref "posts/prospective-config-code-analysis" >}}).

**본문에서 먼저 볼 순서:** **Part 1: DSF — 대각 고정 피드백으로 BPTT 대체** → **Part 2: EqProp-SeqLearning — 평형전파로 NLP 분류** → **DSF vs EqProp 비교**.

이 글은 구현을 다루는 심화 글이다. 첫 독서에서는 설명과 함수의 역할을 보고, Python과 해당 라이브러리를 익힌 뒤 코드와 수식을 다시 따라가자.

<!-- PAPER-PREREQUISITES -->
### 선수 개념과 읽을 논문

**먼저 이해할 개념:** 손실 함수, 미분·연쇄법칙, 가중치와 활성값의 구분; 에너지 기반 방법은 최적화와 수렴.

아래 순서는 이 글을 이해하기 위한 공부 순서다. ‘개념 준비’는 읽기 안내이며, 직접적인 연구 계보는 연결 칸에 따로 표시했다. 이미 해당 개념을 설명할 수 있다면 아는 부분은 넘어가도 된다.

| 순서·논문 | 연결·익힐 개념 | 먼저 볼 부분 |
|---|---|---|
| 1. [Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation](https://arxiv.org/abs/1602.05179) | **구현 대상 이론**: 에너지·평형 상태·두 단계 학습 | 자유 상태와 목표로 살짝 유도한 상태의 비교 |
| 2. [Inferring neural activity before plasticity as a foundation for learning beyond backpropagation](https://doi.org/10.1038/s41593-023-01514-1) · [블로그 해설]({{< relref "posts/prospective-configuration-review" >}}) | **개념 준비**: 상태 최적화·가중치 학습·예측 오차 | 활성 상태를 먼저 조정하고 가중치를 나중에 바꾸는 순서 |

**현재 글과 연결하기:** 위 개념을 익힌 뒤, 이 글에서 ‘고정 피드백과 평형 상태 비교의 구현 차이 이해하기’에 해당하는 부분을 읽자. 기존 방법의 어느 단계가 바뀌었는지 그림이나 입력·출력으로 설명해 보자.

[선수 논문 전체 지도와 원문 읽는 법]({{< relref "paper-prerequisites" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- PAPER-PREREQUISITES-END -->

<!-- LEARNING-START-END -->

---

## Part 1: DSF — 대각 고정 피드백으로 BPTT 대체

### 레포 정보

- **GitHub**: [p0lcAi/DSF](https://github.com/p0lcAi/DSF)
- **논문**: arXiv 2503.23104 (2025.03)
- **저자**: Paul Caillon, Erwan Fagnou, Alexandre Allauzen (LAMSADE, Paris Dauphine)

### 디렉토리 구조

```
dsf/
├── dsf.py              # 핵심: StandardRnn, NoBpttRnn, DfaRnn
├── utils/
│   ├── engine.py       # 학습 루프 (train_model)
│   └── model_utils.py  # Model, RNN Cell 정의, Config
configs/
├── ptb_word/           # PTB 설정 4개
└── wikitext103_word/   # Wikitext-103 설정 10개
tasks/
├── load_dataset.py     # 데이터셋 라우터
└── huggingface_dataset.py  # Wikitext-103 로딩
```

---

### 핵심 구현: 대각 피드백 행렬

**파일**: `dsf/dsf.py`

이 논문은 BPTT의 시간 방향 그래디언트 전파를 **고정된 대각 행렬**로 대체한다.

```python
# dsf.py, DfaRnn.initialize_feedback_matrix, line 155-173
elif config.model.state_transition == 'diagonal':
    A = torch.diag(torch.linspace(0.0, 1.0, h_dim))
```

대각 원소가 `[0.0, ..., 1.0]`으로 선형 배치된다. 이 행렬은 학습 중 **절대 업데이트되지 않는다** (`self.register_buffer("A", A)`).

### 그래디언트 근사

```python
# dsf.py, DfaRnn.get_state_gradients, line 142-153
def get_state_gradients(self, output_gradients):
    seq_len = output_gradients.shape[1]
    gt = output_gradients[:, seq_len-1]
    gradients = [gt]
    for t in range(seq_len-2, -1, -1):
        gt = output_gradients[:, t] + torch.matmul(gt, self.A)
        gradients.append(gt)
    return torch.stack(gradients[::-1], dim=1)
```

시간을 거슬러 가면서 $g_t = \frac{\partial L}{\partial h_t} + g_{t+1} \cdot A$를 누적한다. **RNN 전이 함수의 야코비안**을 고정 대각 행렬 $A$로 대체한 것이다.

이 그래디언트는 **gradient hook**으로 주입된다:

```python
# dsf.py, ParallelBackwardRnn.forward, line 116-121
def custom_backward(grad_output):
    return self.get_state_gradients(grad_output)
output.register_hook(custom_backward)
```

### 세 가지 학습 모드 비교

```python
# train.py, line 41-45
state_transitions = ['BPTT', 'FT_BPTT', 'diagonal']
```

| 모드 | 클래스 | 핵심 차이 |
|------|-------|----------|
| **BPTT** | `StandardRnn` | 전체 계산 그래프 유지, 표준 autograd |
| **FT-BPTT** | `NoBpttRnn` | `state.detach()` — 시간 그래디언트 완전 절단 |
| **DSF** | `DfaRnn` | detach + 고정 대각 행렬로 시간 그래디언트 근사 |

### RNN Cell 구현

```python
# model_utils.py, GRUCell, line 59-77
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        self.cell = nn.GRUCell(input_size, hidden_size)
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x, h): return self.cell(x, h)
    def rnn_forward(self, x): return self.rnn(x)  # cuDNN 가속
```

`rnn_forward`는 cuDNN을 사용한 병렬 순방향 패스로, FT-BPTT와 DSF에서 속도 이점을 제공한다.

---

## Part 2: EqProp-SeqLearning — 평형전파로 NLP 분류

### 레포 정보

- **GitHub**: [NeuroCompLab-psu/EqProp-SeqLearning](https://github.com/NeuroCompLab-psu/EqProp-SeqLearning)
- **논문**: "Sequence Learning Using Equilibrium Propagation", IJCAI 2023
- **저자**: Malyaban Bal, Abhronil Sengupta

### 디렉토리 구조

```
main.py              # 진입점, 모델/데이터셋 선택
model_utils.py       # 핵심: EP 학습, 수렴 RNN, Hopfield 어텐션 (507줄)
preprocessIMDB.py    # IMDB 데이터 + word2vec
preprocessSNLI.py    # SNLI 데이터 + word2vec
```

매우 컴팩트한 코드베이스 — 핵심 로직이 `model_utils.py` 507줄에 집중되어 있다.

---

### 평형전파 핵심 구현

**파일**: `model_utils.py`

EP의 핵심은 **자유상(free phase)**과 **넛지상(nudge phase)**의 정상 상태 차이로 가중치를 업데이트하는 것이다.

```python
# model_utils.py, train 함수, line 383-491

# 1. 자유상: beta=0으로 에너지 최소화
neurons = model(x, y, neurons, T1, beta=0.0, criterion=criterion)
neurons_1 = copy(neurons)

# 2. 넛지상: beta>0으로 출력을 타겟 방향으로 넛지
neurons = model(x, y, neurons, T2, beta=beta_2, criterion=criterion)
neurons_2 = copy(neurons)

# 3. (선택) 3상: 대칭 그래디언트 추정
neurons = copy(neurons_1)
neurons = model(x, y, neurons, T2, beta=-beta_2, criterion=criterion)
neurons_3 = copy(neurons)

# 가중치 업데이트: EP 학습 규칙
model.compute_syn_grads(x, y, neurons_2, neurons_3,
                         (beta_2, -beta_2), criterion)
```

### 가중치 업데이트: Phi 차이

```python
# model_utils.py, compute_syn_grads, line 265-279
delta_phi = (phi_2 - phi_1) / (beta_1 - beta_2)
delta_phi.backward()  # PyTorch autograd로 파라미터 그래디언트 계산
```

EP 학습 규칙: 두 정상 상태에서의 원시 함수(Phi) 차이를 넛지 강도로 나누면, 이것이 손실 함수의 그래디언트와 동등하다.

### 수렴 RNN 동역학

```python
# IMDB_model.forward, line 223-255
for t in range(T):
    # 스칼라 에너지 함수 Phi 계산
    phi = sum(W_i @ s_{i-1} * s_i) - beta * Loss

    # Phi의 뉴런에 대한 그래디언트
    grads = torch.autograd.grad(phi, neurons, create_graph=True)

    # 뉴런 상태 업데이트: s = sigma(dPhi/ds)
    neurons[idx] = activation(grads[idx])
```

고정점 반복 $s^{t+1} = \sigma(\partial\Phi/\partial s)$로 수렴한다.

### Modern Hopfield Attention 통합

```python
# IMDB_model.forward, line 234-240
query = grads[idx]          # Phi의 그래디언트가 query
key = x                     # 입력 임베딩이 key
value = x                   # 입력 임베딩이 value

qk = torch.softmax(scale * torch.bmm(query, key.T), dim=-1)
attention_output = torch.bmm(qk, value)
neurons[idx] = attention_output
```

**에너지 함수의 그래디언트가 query**로 사용된다는 점이 핵심이다. Modern Hopfield Network의 어텐션 메커니즘을 EP 프레임워크에 자연스럽게 통합한 것이다.

---

## DSF vs EqProp 비교

| 항목 | DSF | EqProp-SeqLearning |
|------|-----|-------------------|
| **목표** | BPTT 속도 개선 | 생물학적 타당한 비역전파 학습 |
| **순방향** | 표준 RNN | 수렴 RNN (에너지 고정점) |
| **역방향** | 고정 대각 행렬 피드백 | EP: 자유상/넛지상 정상 상태 비교 |
| **RNN 종류** | Vanilla RNN, GRU, LSTM | 커스텀 수렴 RNN + Hopfield 어텐션 |
| **데이터셋** | Wikitext-103, PTB (언어 모델링) | IMDB, SNLI (분류) |
| **손실** | CrossEntropy | MSE |
| **코드 스타일** | YAML 설정 기반 | Argparse CLI |

두 구현을 비교할 때는 오차를 어디서 얻고, 어느 값으로 가중치를 갱신하는지부터 따라가면 된다. DSF에서는 시간 방향의 피드백 행렬을, EqProp에서는 상태가 수렴하는 과정과 두 상태의 차이를 살펴본다. ‘BPTT를 대체한다’는 목적은 같아도 학습 신호를 만드는 절차가 다르므로, 반복 계산과 저장해야 할 상태도 각각 확인해야 한다.

<!-- LEARNING-END -->
---

## 읽은 뒤 확인하고 다음으로

**확인 질문:** 고정 피드백과 평형 상태 비교의 구현 차이 이해하기를 목표로, 본문에서 다룬 입력·중간 과정·결과를 자기 말로 설명해 보자. 실험이나 사례가 있다면 어떤 조건에서 확인했는지도 덧붙여 보자.

이 경로의 마지막 글이다. [학습 순서]({{< relref "learn" >}})로 돌아가 다른 경로를 고르거나, 어려웠던 수식·코드 부분을 다시 읽어 보자.

[전체 학습 순서]({{< relref "learn" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- LEARNING-END-END -->
