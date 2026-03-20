---
title: "[코드 분석] Prospective Configuration — PCLayer와 PCTrainer로 구현하는 비역전파 학습"
date: 2026-03-20
tags: ["코드분석", "Predictive Coding", "비역전파"]
categories: ["ML/AI"]
summary: "YuhangSong/Prospective-Configuration 레포를 소스코드 수준에서 분석한다. PCLayer가 예측 오차를 계산하고, PCTrainer가 추론→학습 2단계 루프를 어떻게 오케스트레이션하는지 추적한다."
math: true
toc: true
draft: false
---

## 레포 정보

- **GitHub**: [YuhangSong/Prospective-Configuration](https://github.com/YuhangSong/Prospective-Configuration)
- **논문**: Nature Neuroscience, Volume 27, February 2024
- **라이선스**: MIT

---

## 1. 디렉토리 구조

```
predictive_coding/predictive_coding/    # 핵심 라이브러리
├── pc_layer.py       # PCLayer — 레이어 사이에 삽입되는 핵심 모듈
├── pc_trainer.py     # PCTrainer — 2단계 학습 알고리즘 오케스트레이터
└── utils.py          # 유틸리티

experiments/          # 14개 실험 설정 (YAML)
├── nature_small-arch-small-data/    # FashionMNIST, BP vs PC 비교
├── nature_forgetting/               # Continual learning
├── nature_target_alignment/         # Target alignment 측정
├── nature_search_depth/             # 깊이 실험 (2-10층)
├── nature_concept_drift/            # 비정상 환경
└── nature_cnn_v2/                   # CNN 아키텍처
```

---

## 2. PCLayer: 핵심 계산 단위

**파일**: `pc_layer.py`

`PCLayer`는 표준 PyTorch 레이어 **사이에** 삽입되는 모듈이다. 핵심 역할은 latent state `_x`를 유지하고 prediction error(에너지)를 계산하는 것이다.

### 이중 동작 모드

```python
# 평가 모드: 단순 identity — 일반 feedforward 네트워크처럼 동작
if not self.training:
    return mu  # 예측값을 그대로 통과

# 학습 모드: latent state x를 반환 (예측값 mu 대신)
self._x = nn.Parameter(x_data, True)
return self._x
```

**핵심**: 학습 시 downstream 레이어는 `mu`(순방향 예측)가 아닌 `_x`(latent state)를 받는다. 이것이 Prospective Configuration의 핵심 메커니즘이다.

### 에너지 함수

```python
# pc_layer.py, line 17-18 (기본값)
energy_fn = lambda inputs: 0.5 * (inputs['mu'] - inputs['x'])**2
```

논문의 $E_l = \frac{1}{2} \lVert \mu_l - x_l \rVert^2$에 정확히 대응한다. `mu`는 상위 레이어의 예측, `x`는 현재 latent state이다.

### Latent State 초기화

```python
# 기본 sample_x_fn
sample_x_fn = lambda inputs: inputs['mu'].detach().clone()
```

매 배치 시작 시 `_x`를 **순방향 예측값의 detach된 복사본**으로 초기화한다. `.detach()`가 핵심 — 계산 그래프를 끊어 `_x`가 자유 변수가 되도록 한다.

---

## 3. PCTrainer: 2단계 알고리즘

**파일**: `pc_trainer.py`

### 두 개의 옵티마이저

```python
optimizer_x  # latent state x 전용 (기본: SGD, lr=0.1)
optimizer_p  # 가중치 파라미터 전용 (기본: Adam, lr=0.001)
```

`optimizer_x`는 PCLayer의 `_x` 파라미터만, `optimizer_p`는 나머지 모든 모델 파라미터를 관리한다.

### 핵심 학습 루프

```python
for t in range(T):  # T: 추론 반복 횟수 (기본 512)

    # t=0: 순방향 패스로 x 초기화
    if t == 0:
        for pc_layer in pc_layers:
            pc_layer.set_is_sample_x(True)
        outputs = model(inputs)
        self.recreate_optimize_x()

    else:
        outputs = model(inputs)

    # 손실 계산 (출력 클램핑)
    loss = loss_fn(outputs, target)

    # 전체 PCLayer 에너지 합산
    energy = sum(self.get_energies())

    # 전체 목적 함수
    overall = loss + energy * energy_coefficient

    # 역전파
    overall.backward()

    # Phase 1: Latent state 업데이트 (t < T-1)
    if t in update_x_at:
        optimizer_x.step()

    # Phase 2: 가중치 업데이트 (마지막 반복)
    if t in update_p_at:
        optimizer_p.step()
```

### 논문과의 대응

| 논문 개념 | 코드 |
|----------|------|
| 자유 에너지 $\mathcal{F} = \sum E_l + \text{Loss}$ | `overall = loss + energy * coefficient` |
| 레이어 에너지 $E_l = \frac{1}{2} \lVert \mu_l - x_l \rVert^2$ | `energy_fn` |
| 추론: $\Delta x = -\frac{\partial \mathcal{F}}{\partial x}$ | `overall.backward()` → `optimizer_x.step()` |
| 학습: $\Delta W = -\frac{\partial \mathcal{F}}{\partial W}$ | `overall.backward()` → `optimizer_p.step()` |
| 입력 클램핑 $x_0 = \text{data}$ | 입력층에 PCLayer 없음 → 데이터 직접 입력 |
| 출력 클램핑 $x_L = \text{target}$ | `loss_fn(outputs, target)` |

---

## 4. BP vs PC 비교: 단일 플래그로 전환

```yaml
# nature_small-arch-small-data/bp-il.yaml
predictive_coding:
    grid_search:
        - True    # PC 모드
        - False   # BP 모드
```

```python
# predictive_coding=False 일 때
if not self.config['predictive_coding']:
    for model_ in self.model:
        if isinstance(model_, pc.PCLayer):
            self.model.remove(model_)  # PCLayer 모두 제거
```

PCLayer를 제거하면 표준 `nn.Sequential`이 되어 일반 역전파로 학습한다. 동일 아키텍처, 동일 데이터, 동일 하이퍼파라미터에서 공정한 A/B 비교가 가능하다.

---

## 5. 모델 구조 패턴

모든 실험에서 반복되는 패턴:

```
Linear → PCLayer → Activation → Linear → PCLayer → Activation → ... → Linear
```

- PCLayer는 가중치 레이어 **뒤**, 활성화 함수 **앞**에 위치
- 마지막 Linear 뒤에는 PCLayer **없음** (출력은 loss로 클램핑)
- 입력 앞에도 PCLayer **없음** (데이터로 클램핑)

---

## 6. Target Alignment 측정

```python
prediction_before = model(data)          # 학습 전 출력
pc_trainer.train_on_batch(...)           # 1 step 학습
prediction_after = model(data)           # 학습 후 출력

predictive_vector = prediction_after - prediction_before  # 실제 변화
target_vector = target - prediction_before                # 원하는 변화

alignment = cosine_similarity(predictive_vector, target_vector)
```

이 값이 1에 가까울수록 가중치 업데이트가 목표 방향과 일치한다. 논문의 핵심 발견: BP는 깊은 네트워크에서 alignment가 급격히 하락하지만, PC는 ~1.0을 유지한다.
