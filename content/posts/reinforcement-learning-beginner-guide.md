---
title: "[입문] 강화학습 완전 정복 — 게임 배우는 아이부터 PPO·DPO까지 그림으로 이해하기"
date: 2026-04-22
tags: ["강화학습", "RL", "PPO", "DPO", "RLHF", "입문"]
categories: ["ML/AI"]
summary: "강화학습을 한 번도 접해본 적 없는 사람도 이해할 수 있도록 비유와 그림으로 설명한다. 강아지 훈련 비유부터 Q-learning, REINFORCE, Actor-Critic, PPO, DPO, GRPO, RLHF까지 — 강화학습의 모든 핵심 개념을 단계적으로 정리."
math: true
toc: true
draft: false
---

## 들어가며 — 강화학습이란?

엄마가 아이에게 자전거를 가르친다고 상상해보자.

```
아이가 페달을 밟는다 (행동)
       ↓
   넘어진다 (결과)
       ↓
엄마가 "다음엔 핸들을 더 똑바로!" (보상/벌점)
       ↓
아이가 다시 시도한다 (학습된 행동)
       ↓
이번엔 좀 더 잘 간다 (보상)
       ↓
   ... 반복 ...
       ↓
결국 자전거를 탈 줄 안다!
```

**이게 강화학습이다.** 어떤 정답도 없이, **시도 → 결과 → 피드백**의 반복으로 배우는 것.

> **강화학습 (Reinforcement Learning, RL)**: 에이전트가 환경과 상호작용하며 보상을 최대화하는 행동 정책을 학습하는 방법.

---

## 1. 강화학습의 4가지 핵심 요소

### 강아지 훈련으로 비유

```
┌─────────────────────────────────────────┐
│                                         │
│   🐕 강아지 (Agent)                      │
│       │                                 │
│       │ "앉아!" 명령에 앉음 (Action)      │
│       ↓                                 │
│   🏠 집안 환경 (Environment)              │
│       │                                 │
│       │ 주인이 간식 줌 (Reward)           │
│       ↓                                 │
│   🐕 다음엔 더 빨리 앉음 (학습)           │
│                                         │
└─────────────────────────────────────────┘
```

### 4가지 핵심 요소

| 요소 | 영어 | 강아지 예시 | RAG 예시 |
|------|------|-----------|---------|
| **에이전트** | Agent | 강아지 | LLM |
| **환경** | Environment | 집안 | 사용자 + 검색기 |
| **행동** | Action | 앉기, 손주기 | 답변 생성, 검색 호출 |
| **보상** | Reward | 간식 (+) / 혼남 (-) | 정답 일치 (+) / 할루시네이션 (-) |

---

## 2. 추가 개념 — State, Policy, Value

### 게임으로 이해하기 — 슈퍼마리오

```
🎮 슈퍼마리오를 플레이한다고 상상

State (상태):
  현재 화면 — 마리오 위치, 적 위치, 동전 위치 등

Action (행동):
  → 오른쪽 이동
  ↑ 점프
  ↓ 숙이기

Reward (보상):
  +10: 동전 먹기
  +100: 적 밟기
  -1000: 죽음

Policy (정책) π:
  "각 상태에서 어떤 행동을 할지 결정하는 규칙"
  예: "적이 가까이 있으면 점프"
```

### 수학적 표현

**정책(Policy)**: 상태 $s$가 주어졌을 때 행동 $a$를 선택할 확률

$$\pi(a | s) = P(\text{행동} = a \mid \text{상태} = s)$$

**가치 함수 (Value Function)**: 상태 $s$에 있을 때 미래 누적 보상의 기대값

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s\right]$$

- $\gamma$ (감마): **할인율** (0~1) — 먼 미래의 보상은 덜 중요하게

**Q-함수 (Action-Value)**: 상태 $s$에서 행동 $a$를 했을 때 미래 누적 보상

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s, a_0 = a\right]$$

**한 줄 정리**:
- $V$: "이 상태가 좋아?"
- $Q$: "이 상태에서 이 행동이 좋아?"

---

## 3. 강화학습은 왜 어려운가?

### 어려움 1: Delayed Reward (지연 보상)

```
체스 게임:
  1수, 2수, 3수, ..., 50수 (체크메이트!)
  
어느 수가 결정적이었는지 어떻게 알아? 
→ "Credit Assignment Problem"
```

### 어려움 2: Exploration vs Exploitation (탐험 vs 활용)

```
새로운 식당에 가서:
  - 항상 똑같은 메뉴만 시킴 (Exploitation, 활용)
  - 새 메뉴 시도 (Exploration, 탐험)

너무 안전하면? 더 좋은 메뉴 못 찾음
너무 도전하면? 맛없는 거 자주 먹음
```

### 어려움 3: 분포 변화 (Non-stationary)

```
정책이 바뀌면 → 방문하는 상태가 바뀜 → 학습 데이터가 바뀜
"내가 학습하는 동안 시험 범위가 계속 바뀐다!"
```

---

## 4. 알고리즘 분류 지도

```
              강화학습
                 │
        ┌────────┴────────┐
        ↓                 ↓
  Model-Based       Model-Free
  (환경 모델 학습)    (환경 모델 X, 경험으로만)
        │                 │
                 ┌────────┴────────┐
                 ↓                 ↓
            Value-Based       Policy-Based
            (Q값 학습)         (정책 직접 학습)
                 │                 │
                 │      ┌──────────┴──────────┐
                 │      ↓                     ↓
                 │  Actor-Critic         REINFORCE
                 │  (둘 다)              (정책만)
                 │      │
            ┌────┴┐  ┌──┴──┐
            ↓    ↓  ↓     ↓
         Q-Learning  A2C    PPO
         DQN         A3C    SAC
                     TRPO   DDPG
```

---

## 5. 알고리즘 1: Q-Learning (가장 기본)

### 아이디어

> "모든 (상태, 행동) 쌍에 대해 Q값을 표로 저장하고, 경험으로 업데이트하자."

### Q-Table 예시 (격자 미로)

```
         상     하     좌     우
S1 [   0.5,  0.2,  0.1,  0.8 ]   ← 우로 가는 게 가장 좋음
S2 [   0.3,  0.9,  0.2,  0.4 ]
S3 [   0.1,  0.7,  0.5,  0.3 ]
...
```

### 업데이트 공식

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

해석:
- 새 추정 = 옛 추정 + 학습률 × (실제 본 보상 + 다음 상태의 최대 Q값 − 옛 추정)
- "실제 결과가 예상보다 좋으면 Q값 ↑, 나쁘면 ↓"

### 한계

상태가 많으면 Q-table이 너무 커진다 (체스: 10^120개 상태!).

---

## 6. 알고리즘 2: DQN (Deep Q-Network) — 딥러닝 결합

### 아이디어

> "Q-table 대신 **신경망**으로 Q값을 근사하자."

```
[상태 s] → 신경망 → [Q(s, a1), Q(s, a2), ..., Q(s, an)]
            ↑
        파라미터 θ를 학습
```

### 핵심 트릭 2가지

**1. Experience Replay** (경험 재생):
```
경험을 (s, a, r, s') 튜플로 저장 → 랜덤 샘플링하여 학습
→ 연속 데이터의 강한 상관 제거
```

**2. Target Network** (목표 네트워크):
```
Q-network 두 개를 사용:
  - Online network: 매 스텝 업데이트
  - Target network: 천천히 업데이트 (안정성)
```

### 성과

DeepMind의 DQN이 Atari 게임 49개를 사람 수준으로 플레이 (2015).

---

## 7. 알고리즘 3: REINFORCE (Policy Gradient의 기본)

### 아이디어

> "Q값 말고 **정책 자체**를 직접 학습하자."

### 정책을 신경망으로

```
[상태 s] → 신경망 → [π(a1|s), π(a2|s), ..., π(an|s)]  ← 행동 확률
            ↑
        파라미터 θ를 학습
```

### 학습 목표

기대 누적 보상 $J(\theta)$를 최대화:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t r_t\right]$$

### 정책 경사 (Policy Gradient)

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot R_t\right]$$

해석:
- 좋은 결과($R_t$가 큼) → 그 행동의 확률을 **올린다**
- 나쁜 결과($R_t$가 작음) → 그 행동의 확률을 **내린다**

### 한계

분산이 크다 → 학습이 불안정.

---

## 8. 알고리즘 4: Actor-Critic (둘의 결합)

### 아이디어

> "Policy(Actor)와 Value(Critic)를 같이 쓰자."

```
[상태 s]
   │
   ├─→ Actor (정책) → 행동 선택
   │
   └─→ Critic (가치) → "이 행동 얼마나 좋은데?"
                         ↑
                    Actor에게 피드백
```

### 직관

```
🎭 Actor (배우): "이렇게 연기할게요!"
🧐 Critic (평론가): "오, 7점! 더 감정을 넣어봐"
🎭 Actor: "이번엔 이렇게!"
🧐 Critic: "9점! 좋아!"
```

### Advantage 함수

$$A(s, a) = Q(s, a) - V(s)$$

해석: "이 행동이 평균보다 얼마나 더 좋은가?"
- $A > 0$: 이 행동이 평균보다 좋다 → 확률 ↑
- $A < 0$: 평균보다 나쁘다 → 확률 ↓

---

## 9. 알고리즘 5: TRPO (Trust Region Policy Optimization)

### 문제: 정책이 너무 크게 바뀌면 망함

```
반복마다 정책 업데이트
  │
  ├─ 작게 업데이트: 학습 느림
  └─ 크게 업데이트: 망가짐 (학습 안 됨)
```

### 해결: 신뢰 영역 (Trust Region)

> "한 번에 정책을 너무 멀리 바꾸지 마. KL divergence로 거리 제한!"

$$\max_\theta J(\theta) \quad \text{s.t.} \quad D_{KL}(\pi_{\theta_{\text{old}}} \| \pi_\theta) \leq \delta$$

**한계**: 구현 복잡, 계산 비싸다.

---

## 10. 알고리즘 6: PPO (Proximal Policy Optimization) ⭐

### 한 줄 요약

> **TRPO를 단순화 — KL 제약 대신 ratio를 클립(clip)으로 제한.**

OpenAI의 표준 알고리즘. **현재 가장 널리 쓰이는 RL 알고리즘**.

### 핵심 아이디어

비율 정의:

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$

**Clipped Surrogate Objective**:

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \cdot A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t\right)\right]$$

### 그림으로

```
정책 비율 r:
  0.5    1.0    1.5
   ├──────┼──────┤
   ↑      ↑      ↑
  너무   현재   너무
  멀어   정책   멀어

PPO는 이 범위 [1-ε, 1+ε] (보통 ε=0.2)를 벗어나면 무시!
→ 안정적인 학습
```

### PPO 학습 루프

```
for iteration in epochs:
    # 1. 현재 정책으로 환경에서 trajectory 수집
    trajectories = collect(env, π_old)
    
    # 2. Advantage 계산
    A_t = compute_advantage(trajectories)  # GAE 사용
    
    # 3. 미니배치로 여러 epoch 학습
    for epoch in range(K):
        for batch in trajectories:
            loss = -clipped_objective(π_θ, π_old, A_t)
            θ = θ - lr * ∇loss
    
    # 4. π_old ← π_θ
```

### 왜 표준이 되었나?

- **TRPO보다 단순** — clip 한 줄로 처리
- **A2C/A3C보다 안정적** — 정책이 너무 멀어지지 않음
- **다양한 작업에 잘 통함** — 게임, 로봇, LLM 등

---

## 11. 연속 액션 알고리즘들

지금까지는 이산 행동 (좌/우/점프). **연속 행동** (조향각 30.5도)이라면?

### DDPG (Deep Deterministic Policy Gradient)

- Actor-Critic + 결정적 정책 (확률 X, 액션 직접 출력)
- Target network + Replay buffer

### TD3 (Twin Delayed DDPG)

- DDPG의 개선판 — 두 개의 Critic, 지연 업데이트

### SAC (Soft Actor-Critic) ⭐

- **현재 연속 제어 표준**
- 보상에 **엔트로피 보너스** 추가 → 자연스러운 탐험

$$J(\pi) = \mathbb{E}\left[\sum_t r_t + \alpha \mathcal{H}(\pi(\cdot | s_t))\right]$$

---

## 12. RL의 핵심 설계 4단계

### Step 1: 환경 정의

```
질문: 무엇을 학습시킬 것인가?

예시:
- 자율주행 → 환경: 도로, 다른 차, 신호
- 게임 AI  → 환경: 게임 화면, 점수
- LLM 정렬 → 환경: 사용자 + 보상 모델
```

### Step 2: State / Action 설계

```
State:
  - 무엇을 관찰할 수 있나?
  - 충분한 정보인가?
  - 너무 큰가? (차원의 저주)

Action:
  - 이산 vs 연속?
  - 행동 공간이 적절한가?
```

### Step 3: 보상 함수 설계 (가장 어려운 부분)

```
🚨 잘못된 보상 설계의 함정 (Reward Hacking)

예시 1: 청소 로봇
  보상 = "쓰레기 적게 보임"
  → 로봇이 카메라 눈을 가림! (쓰레기를 안 봐서 0개로 측정)

예시 2: 보트 레이싱
  보상 = "체크포인트 도달"
  → 보트가 같은 체크포인트만 무한 루프
```

#### 좋은 보상 설계 원칙

1. **목표를 직접 인코딩**: "잘 작동하는" 보상 X, "원하는 결과" O
2. **Sparse vs Dense**:
   - Sparse: 게임 끝에만 보상 → 학습 어려움
   - Dense: 매 스텝 보상 → 학습 빠름, but reward hacking 위험
3. **Reward Shaping**: 보조 보상 추가 (단, 최적 정책 변경 X)
4. **정규화**: 보상 크기 비슷하게

### Step 4: 알고리즘 선택

| 상황 | 추천 |
|------|------|
| 이산 행동, 시뮬레이터 빠름 | DQN, PPO |
| 연속 행동 | SAC, TD3 |
| 정책 안정성 중요 | PPO |
| 데이터 효율 중요 | SAC, off-policy 계열 |
| LLM 정렬 | PPO, DPO, GRPO |

---

## 13. LLM에서의 강화학습 — RLHF

### RLHF란?

> **Reinforcement Learning from Human Feedback** — 인간 피드백으로 LLM을 정렬

### 3단계 파이프라인

```
[Stage 1] SFT (Supervised Fine-Tuning)
  GPT-3 → 지도학습 → InstructGPT (instruction-following)
        ↓
[Stage 2] Reward Model 학습
  사람이 답변 쌍 (A, B)를 보고 "A가 더 좋아"
        ↓
  Reward Model rφ(prompt, response) 학습
        ↓
[Stage 3] PPO로 LLM 정렬
  LLM이 답변 생성 → Reward Model이 평가 → PPO 업데이트
        + KL penalty (원본에서 너무 멀어지지 않게)
```

### 왜 KL penalty?

```
LLM이 보상 hacking을 할 수 있음:
  "Reward Model이 좋아하는 가짜 답변" 생성

방지: KL(π || π_ref) 제약
  → 원본 SFT 모델에서 너무 벗어나지 않도록
```

### RLHF 목적함수

$$\mathcal{L} = \mathbb{E}_{x, y \sim \pi_\theta} [r_\phi(x, y)] - \beta \cdot D_{KL}(\pi_\theta \| \pi_{\text{ref}})$$

---

## 14. DPO — RL을 안 쓰는 RLHF ⭐

### 동기

> **RLHF는 너무 복잡하다!** SFT → RM → PPO (3단계, 4개 모델, 불안정)
> 정말 RL이 필요한가?

### 핵심 통찰

> "KL 제약 보상 최대화의 닫힌 해를 사용하면, 보상 모델 = 정책 그 자체로 표현 가능."

### 수학 (간단 버전)

RLHF의 최적 정책:

$$\pi^*(y|x) \propto \pi_{\text{ref}}(y|x) \cdot \exp\left(\frac{r(x, y)}{\beta}\right)$$

→ 보상을 정책으로 표현:

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \text{const}$$

→ Bradley-Terry 모델에 대입 → **단순 분류 손실**:

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

### RLHF vs DPO

| | RLHF (PPO) | DPO |
|--|-----------|-----|
| 단계 수 | 3 | 2 |
| 모델 수 | 4 (정책+참조+보상+가치) | 2 (정책+참조) |
| RL 사용 | ✅ | ❌ |
| 안정성 | 낮음 | **높음** |
| 구현 복잡도 | 매우 높음 | **PyTorch 10줄** |

### DPO의 핵심 코드

```python
def dpo_loss(pi_logps, ref_logps, yw, yl, beta=0.1):
    pi_yw = pi_logps[yw];  pi_yl = pi_logps[yl]
    ref_yw = ref_logps[yw]; ref_yl = ref_logps[yl]
    
    # 차이 계산
    pi_diff  = pi_yw  - pi_yl
    ref_diff = ref_yw - ref_yl
    
    # log-sigmoid 손실
    return -F.logsigmoid(beta * (pi_diff - ref_diff))
```

[상세 리뷰: DPO 리뷰](dpo-review.md)

---

## 15. GRPO — DeepSeek-R1의 비밀

### GRPO = Group Relative Policy Optimization

PPO의 문제: **Critic (가치 모델)이 비싸다** — 정책 모델만큼 큰 두 번째 네트워크.

### GRPO의 해결

> "Critic 빼고, **그룹 내 상대 점수**로 advantage 계산하자."

```
같은 질문에 G개 답변 생성 (예: G=8)
       ↓
각 답변의 보상 r1, r2, ..., r8
       ↓
Advantage = (r_i - mean) / std   ← 그룹 정규화
       ↓
PPO 스타일 업데이트
```

### 결과

- 메모리 절약 (Critic 제거)
- DeepSeek-R1, Search-R1 등이 채택

### GRPO 손실 (단순화)

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[\frac{1}{G} \sum_i \min\left(r_i \cdot A_i,\ \text{clip}(r_i) \cdot A_i\right)\right] + \beta \cdot D_{KL}$$

---

## 16. RLHF의 후예들 — 한눈에 보기

| 방법 | 핵심 | 특징 |
|------|------|------|
| **PPO-RLHF** | 표준 RL 방식 | 복잡, 불안정 |
| **DPO** | RL 제거, 분류로 환원 | 안정, 단순 |
| **IPO** | DPO sigmoid 포화 해결 | DPO 개선 |
| **KTO** | Unpaired binary label만 필요 | 데이터 효율 |
| **ORPO** | SFT + DPO를 1-stage로 | 통합 |
| **SimPO** | 참조 모델 제거 + 길이 정규화 | DPO 능가 |
| **GRPO** | Critic 제거, 그룹 상대 보상 | 메모리 효율 |
| **RLAIF** | 인간 대신 AI 피드백 | Constitutional AI |
| **DPO + RAG** | 충실도 선호 학습 | RAG 정렬 |

---

## 17. 실전 — 강화학습 시스템 설계도

### 예시: 게임 AI 만들기

```
┌─────────────────────────────────────────────┐
│ Step 1: 환경 구성                            │
│   - OpenAI Gym 또는 Unity ML-Agents          │
│   - State: 게임 화면 (84x84 픽셀)            │
│   - Action: 5가지 (상하좌우, 가만히)          │
└─────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│ Step 2: 보상 설계                            │
│   - +10: 동전                                │
│   - +100: 적 처치                            │
│   - -1: 매 스텝 (빠른 클리어 유도)           │
│   - -1000: 사망                              │
└─────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│ Step 3: 알고리즘 선택                        │
│   PPO + CNN policy network                   │
└─────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│ Step 4: 학습 루프                            │
│   for episode in range(10000):               │
│       state = env.reset()                    │
│       while not done:                        │
│           action = policy(state)             │
│           next_state, reward, done = env.step│
│           buffer.add(transition)             │
│       train_PPO(buffer)                      │
└─────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│ Step 5: 평가 + 디버깅                        │
│   - Reward curve 확인                        │
│   - 행동 시각화                              │
│   - Reward hacking 모니터링                  │
└─────────────────────────────────────────────┘
```

### 예시: LLM RAG에 DPO 적용

```
Step 1: 데이터 준비
  (질의, 컨텍스트, 좋은 답변 y_w, 나쁜 답변 y_l)
  - y_w: 문서를 정확히 인용
  - y_l: 할루시네이션

Step 2: SFT 모델 (참조 정책)
  pi_ref = SFT 모델 (frozen)

Step 3: DPO 학습
  pi_θ를 DPO loss로 학습
  - 좋은 답변 확률 ↑
  - 나쁜 답변 확률 ↓
  - KL로 pi_ref에서 너무 멀어지지 않게

Step 4: 평가
  - Faithfulness (RAGAS)
  - Hallucination rate
  - Win rate (vs SFT)
```

---

## 18. 자주 헷갈리는 개념들

### On-policy vs Off-policy

| | On-policy | Off-policy |
|--|----------|------------|
| **학습 데이터** | 현재 정책으로 수집한 것만 | 과거 정책 데이터도 OK |
| **예시** | PPO, A2C, REINFORCE | Q-Learning, DQN, SAC |
| **데이터 효율** | 낮음 (재사용 불가) | 높음 (replay buffer) |
| **안정성** | 높음 | 낮음 (수정 필요) |

### Model-based vs Model-free

| | Model-based | Model-free |
|--|------------|------------|
| **환경 모델** | 학습함 (P(s'|s,a)) | 학습 안 함 |
| **장점** | 데이터 효율 ↑ | 단순함 |
| **단점** | 모델 오류 누적 | 데이터 많이 필요 |
| **예시** | MuZero, Dreamer | DQN, PPO, SAC |

### Online vs Offline RL

| | Online | Offline |
|--|--------|---------|
| **환경 상호작용** | 학습 중 가능 | 고정 데이터셋만 |
| **상황** | 시뮬레이터 있음 | 실제 데이터만 (의료 등) |

### Episode vs Step

```
Episode (에피소드): 게임 한 판 (시작 → 끝)
Step (스텝): 한 번의 행동
```

---

## 19. RL 학습의 흔한 문제와 해결

### 문제 1: "보상이 0이라 학습이 안 돼요"

**원인**: Sparse reward — 게임 끝에만 보상

**해결**:
- Reward shaping (보조 보상)
- Curriculum learning (쉬운 것부터)
- Hindsight Experience Replay (실패도 활용)

### 문제 2: "정책이 한 행동만 해요"

**원인**: 탐험 부족, 또는 너무 자신 있음

**해결**:
- Entropy bonus (SAC 스타일)
- ε-greedy exploration
- 보상 스케일 조정

### 문제 3: "학습이 갑자기 망가져요"

**원인**: 정책이 한꺼번에 너무 크게 바뀜

**해결**:
- PPO clip ratio 줄이기 (ε=0.1)
- Learning rate 줄이기
- KL penalty 강화

### 문제 4: "Reward hacking이 의심돼요"

**원인**: 보상 함수가 진짜 목표를 정확히 인코딩하지 못함

**해결**:
- 보상 디버깅 (행동 영상 보기)
- Constitutional AI (AI가 평가)
- Multi-objective reward

---

## 20. 알고리즘 한눈에 비교

| 알고리즘 | 종류 | 액션 | 정책 | 특징 |
|---------|------|------|------|------|
| **Q-Learning** | Value | 이산 | Off-policy | 표 기반 (작은 문제) |
| **DQN** | Value | 이산 | Off-policy | 신경망 + Replay |
| **REINFORCE** | Policy | 둘 다 | On-policy | 가장 단순 PG |
| **A2C/A3C** | Actor-Critic | 둘 다 | On-policy | 병렬 버전 |
| **TRPO** | Actor-Critic | 둘 다 | On-policy | KL 제약 |
| **PPO** ⭐ | Actor-Critic | 둘 다 | On-policy | **가장 표준** |
| **DDPG** | Actor-Critic | 연속 | Off-policy | 결정적 정책 |
| **SAC** ⭐ | Actor-Critic | 연속 | Off-policy | 엔트로피 보너스 |
| **GRPO** | Policy | 둘 다 | On-policy | Critic 없음 |
| **DPO** | (RL 아님) | 텍스트 | - | 분류 손실 |

---

## 21. 추천 학습 경로

### 입문 (1-2주)

1. OpenAI Gym으로 CartPole 환경 익히기
2. Q-Learning으로 FrozenLake 풀기
3. DQN으로 CartPole 풀기 (`stable-baselines3` 사용)

### 중급 (1개월)

4. PPO로 Atari Pong 풀기
5. SAC로 MuJoCo 환경
6. 자신만의 환경 만들기

### 고급 (3개월+)

7. RLHF 구현 (TRL 라이브러리)
8. DPO 직접 구현
9. GRPO로 추론 모델 학습 (DeepSeek-R1 스타일)

### 추천 도구

```
강화학습 라이브러리:
  - stable-baselines3 (입문 최고)
  - CleanRL (코드가 단일 파일, 학습용)
  - Ray RLlib (대규모)
  - TRL (LLM RL, DPO/PPO/GRPO)

환경:
  - OpenAI Gym / Gymnasium
  - PettingZoo (멀티 에이전트)
  - Unity ML-Agents
  - Atari, MuJoCo
```

---

## 22. 한눈에 정리

### 강화학습 = 시도 + 보상 + 학습

```
시도 → 결과 → 평가 → 정책 업데이트 → 더 나은 시도 → ...
```

### 5가지 알고리즘만 외우면 됨

1. **Q-Learning** — 가장 기본
2. **DQN** — Q-Learning + 딥러닝
3. **PPO** ⭐ — 가장 널리 쓰임
4. **SAC** — 연속 제어 표준
5. **DPO** ⭐ — LLM 정렬, RL 없이

### LLM RL은 RLHF가 시작점

```
SFT → Reward Model → PPO  (전통)
SFT → DPO              (단순화)
SFT → GRPO             (추론 모델)
```

---

## 23. 관련 블로그 포스트

### RL 응용 (RAG 분야)

- [DPO 상세 리뷰](dpo-review.md) — RL 없이 정렬
- [Search-R1 상세 리뷰](search-r1-review.md) — PPO/GRPO로 검색 학습
- [CCS 상세 리뷰](cycle-consistent-search-review.md) — 정답 없는 RL
- [OPQE 상세 리뷰](opqe-review.md) — HyDE에 RL 결합

### RAG 전반

- [RAG 동향 총정리](rag-survey-2026.md)
- [쿼리 최적화 서베이](rag-query-optimization-survey.md)
- [End-to-end 학습 서베이](rag-end-to-end-training-survey.md)

---

## 마무리

강화학습은 처음에는 어려워 보이지만, 결국 **"시도하고, 결과를 보고, 학습한다"**의 반복이다.

핵심 4가지만 기억하자:
1. **Agent, Environment, Action, Reward** — 네 가지 요소
2. **Policy** — 상태에서 행동을 결정하는 함수
3. **Reward** — 무엇을 최적화할지 정의 (가장 중요!)
4. **PPO** — 안정적인 표준 알고리즘 (먼저 익히자)

LLM 시대에는 **RLHF(PPO) → DPO → GRPO**의 진화를 알면 충분하다. 더 깊게 들어가고 싶다면 DeepSeek-R1, Search-R1 같은 최신 RL 기반 모델을 직접 분석해보면 좋다.

> **강화학습은 결국 "잘 한 일에 보상을 주고, 잘못한 일에 벌을 주는" 단순한 원리. 그 원리로 알파고도, ChatGPT도 만들어졌다.**
