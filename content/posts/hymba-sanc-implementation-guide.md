---
title: "Hymba-SANC: 유동 청킹 기반 개념 토큰 언어 모델 구현 가이드"
date: 2026-03-23
tags: ["연구노트", "Hymba", "SANC", "Mamba", "SSM"]
categories: ["ML/AI"]
summary: "Hymba 하이브리드 아키텍처와 SANC(E3) 공리 체계를 결합한 '작은 뇌' 모델 설계. 원시 토큰이 아닌 '다음 개념 토큰'을 예측하는 모델의 전체 파이프라인, 아키텍처, 손실 함수, 구현 마일스톤을 정리한다."
math: true
toc: true
draft: false
---

## 1. 프로젝트 개요

### 목표

현재 LLM은 BPE 같은 **고정된 토크나이저**가 의미 단위를 사전에 결정한다. 하지만 인간의 뇌는 경험을 통해 "어디서 끊을지"를 스스로 학습하며, 그 결과로 만들어진 상위 개념(게슈탈트)으로 사고한다.

이 프로젝트의 목표는 **"작은 뇌"**를 구축하는 것이다:

```
원시 토큰 → 유동 청킹(경계를 학습) → 개념 토큰 형성
→ 다음 "개념 토큰" 예측 → (필요시) 원시 토큰으로 디코딩
```

> **핵심 원칙: 이 모델은 다음 "원시 토큰"이 아니라 다음 "상위 개념 토큰"을 예측한다.**

### 이론적 기반

| 논문 | 역할 |
|------|------|
| **Hymba** (NVIDIA, 2024) | 하이브리드 헤드 병렬 아키텍처 (어텐션 + SSM) |
| **SANC(E3)** (Kwon & Paeng, 2026) | 공리적 지능 프레임워크 — 자기조직화 개념 네트워크 |

### 핵심 아이디어

- Hymba의 어텐션 헤드 → SANC의 "고해상도 리콜" (단기 기억 / 정밀 검색)
- Hymba의 SSM 헤드 + 유동 청킹 → SANC의 "게슈탈트 형성과 압축" (장기 기억 / 개념 네트워크)
- 두 경로가 병렬로 실행되어 상보적으로 동작

---

## 2. 전체 파이프라인

```
=== 인코딩 (지각) ===

Phase 1: 원시 입력         [나는] [오늘] [학교에서] [수학을] [공부했다] [.]
Phase 2: 경계 감지         AdaptiveBoundaryDetector (SSM 기반)
Phase 3: 유동 청킹         [나는, 오늘] [학교에서, 수학을] [공부했다, .]
Phase 4: 개념 압축         ChunkToConceptEncoder (SSM → 개념 벡터)
Phase 5: 경쟁 선택         ConceptCompetition (A1: 유한 용량, A2: 유사성 경쟁)
Phase 6: 개념 토큰         C1(나+오늘)  C2(학교+수학)  C3(공부+완료)

=== 사고 (추론 + 예측) — 핵심 ===

Phase 7: 개념 추론         ConceptReasoningLayer (Hymba 병렬: 어텐션 + SSM)
Phase 8: 다음 개념 예측    C1, C2, C3 → 다음 개념 C4 예측  ← 핵심 목적

=== 디코딩 (표현) — 출력이 필요할 때만 ===

Phase 9: 개념 → 토큰      ConceptDecoder: C4 → "그래서 피곤했다"
```

> **기존 LLM과의 근본적 차이:**
> - GPT: 토큰 → 토큰 → 토큰 (원시 수준에서 사고)
> - Hymba-SANC: 토큰 → **개념** → **개념** → **개념** → 토큰 (개념 수준에서 사고)

---

## 3. SANC(E3) 공리 체계

### 5대 핵심 공리

| 공리 | 이름 | 정의 |
|------|------|------|
| **A1** | 유한 활성 용량 | $\kappa(A_t) \leq C < \infty$. 매 시점 활성 상태는 유한 |
| **A2** | 유사성 기반 경쟁 | 유사한 후보들이 경쟁, 클래스당 최대 하나만 선택 |
| **A3** | 안정화와 삭제 | 반복 관찰 시 $co^{\ast}$ 증가, 미관찰 시 감소 |
| **A4** | 에너지 E3 | $E_3 = \lambda_1 \cdot L_{rec} + \lambda_2 \cdot C_{struct} + \lambda_3 \cdot C_{update}$ |
| **A5** | 공동 발생 → 연관 | 함께 감지된 이벤트들은 게슈탈트의 후보가 됨 |

### SANC와 Hymba SSM의 대응

| SANC 공리 | Hymba SSM + 청킹 대응 |
|-----------|----------------------|
| A1 유한 용량 | SSM 고정 상태 + 청크별 용량 제한 |
| A2 경쟁적 선택 | 청크 경계 게이팅 + sparse selection |
| A3 안정화/삭제 | Selective mechanism + 상태 감쇠 |
| A4 E3 에너지 | Auxiliary loss 추가 |
| A5 공동발생 연관 | 청크 내 SSM 상태 축적 = 게슈탈트 |

---

## 4. 핵심 모듈 설계

### 4.1 유동 경계 감지기 (AdaptiveBoundaryDetector)

SSM이 시퀀스를 읽으면서 "의미적 전환점"을 감지하는 모듈.

```python
class AdaptiveBoundaryDetector(nn.Module):
    def __init__(self, d_model, d_state=16):
        self.context_ssm = MambaBlock(d_model, d_state)
        self.boundary_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1)
        )
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        context = self.context_ssm(x)
        logits = self.boundary_head(context).squeeze(-1)
        boundary_probs = gumbel_sigmoid(logits, tau=self.temperature, hard=True)
        return boundary_probs  # 1 = 경계, 0 = 연속
```

### 4.2 개념 수준 추론 (ConceptReasoningLayer)

Hymba 스타일 병렬 구조로 개념 토큰 간 관계를 추론.

```python
class ConceptReasoningLayer(nn.Module):
    def __init__(self, d_concept, n_heads=8, d_state=16):
        self.concept_attention = MultiHeadAttention(d_concept, n_heads)
        self.concept_ssm = MambaBlock(d_concept, d_state)
        self.meta_tokens = nn.Parameter(torch.randn(1, 3, d_concept) * 0.02)

    def forward(self, concept_tokens):
        meta = self.meta_tokens.expand(B, -1, -1)
        x = torch.cat([meta, concept_tokens], dim=1)
        attn_out = self.concept_attention(self.norm(x))
        ssm_out = self.concept_ssm(self.norm(x))
        x = x + attn_out + ssm_out
        return x[:, 3:, :]  # meta tokens 제거
```

### 4.3 다음 개념 토큰 예측 (NextConceptPredictor)

**모델의 핵심 목적**. 개념 시퀀스로부터 다음 개념 토큰을 예측한다.

```python
class NextConceptPredictor(nn.Module):
    def __init__(self, d_concept, n_concept_vocab=None):
        self.concept_predictor = nn.Sequential(
            nn.Linear(d_concept, d_concept * 4),
            nn.SiLU(),
            nn.Linear(d_concept * 4, d_concept)
        )

    def forward(self, concept_tokens):
        last_concept = concept_tokens[:, -1, :]
        predicted_concept = self.concept_predictor(last_concept)
        return predicted_concept
```

---

## 5. 손실 함수 체계

### 우선순위

| 우선순위 | 손실 | 가중치 | 역할 |
|----------|------|--------|------|
| **1순위** | 개념 예측 손실 ($L_{concept}$) | $\lambda = 2.0$ | 다음 개념 토큰 예측 — **핵심 목적** |
| 2순위 | 재구성 손실 ($L_{rec}$) | $\lambda = 1.0$ | 개념이 원래 정보를 보존하는지 |
| 3순위 | 구조적 비용 ($C_{struct}$) | $\lambda = 0.1$ | 경계/개념 수 최소화 |
| 4순위 | 경계 안정성 ($C_{update}$) | $\lambda = 0.05$ | 경계 수렴 |
| (선택) | 디코딩 손실 ($L_{decode}$) | $\lambda = 0.5$ | 개념 → 원시 토큰 복원 보조 |

### 전체 손실

$$\text{Total Loss} = \lambda_{concept} \cdot L_{concept} + \lambda_{rec} \cdot L_{rec} + \lambda_{struct} \cdot C_{struct} + \lambda_{update} \cdot C_{update}$$

### 개념 예측 손실 옵션

| 방식 | 장점 | 단점 |
|------|------|------|
| **코사인 유사도** (권장) | 방향 중심, 스케일 불변 | 크기 정보 무시 |
| **MSE** | 직관적 | 스케일에 민감 |
| **결합** | 균형잡힌 | 하이퍼파라미터 추가 |
| **코드북 분류** (이산) | 명확한 개념 경계 | 코드북 크기 결정 필요 |

---

## 6. 설계 결정 사항

| 항목 | 권장 | 비고 |
|------|------|------|
| 가변 청킹 방식 | 패딩 기반 (프로토타입) | 이후 NestedTensor 업그레이드 |
| 경계 미분 가능성 | Gumbel-Sigmoid + STE | 가장 일반적, 안정적 |
| 개념 예측 방식 | 연속 벡터 회귀 (초기) | 이후 이산 코드북 실험 |
| 개념 손실 함수 | 코사인 유사도 | 방향이 중요 |
| 위치 정보 | 순서 보존만 (초기) | 추가 임베딩 없음 |
| Mamba 구현 | 순수 PyTorch (프로토타입) | 이후 mamba-ssm 전환 |

---

## 7. 구현 마일스톤

| 마일스톤 | 목표 | 핵심 검증 |
|----------|------|----------|
| **M1** | 뼈대 + 경계 감지기 | 의미 있는 경계가 출현하는지 |
| **M2** | 청크 압축 + 개념 토큰 | $L_{rec}$ 수렴 확인 |
| **M3** | 경쟁 + 추론 레이어 | 개념 경쟁 동작 확인 |
| **M4** | 전체 통합 + 학습 | $L_{concept}$ 수렴 (핵심) |
| **M5** | 분석 + 최적화 | 개념 예측 패턴 분석 |

---

## 8. A100 40GB 실전 주의사항

| 원인 | 대응 |
|------|------|
| SSM 텐서 코어 미활용 | Mamba-2 스타일 행렬곱 재구성 |
| 메모리 바운드 | 커널 퓨전, 배치 크기 최적화 |
| fp32 필요성 (SSM 안정성) | AMP: 파라미터 fp32 + 연산 bf16 |
| 40GB 제한 | Gradient checkpointing 필수 |

### 예상 메모리

```
모델 파라미터 (d_model=512, 8 layers): ~50M params → ~200MB (fp32)
옵티마이저 상태 (AdamW): ~400MB
활성화 메모리 (batch=8, seq=1024): ~2-4GB
경계 감지기 + 청크 상태: ~1-2GB
→ A100 40GB에서 충분히 실행 가능
```

---

## 9. 프로젝트 디렉토리 구조

```
hymba-sanc/
├── configs/model_config.yaml
├── src/
│   ├── model/
│   │   ├── hymba_sanc.py              # 전체 모델
│   │   ├── boundary_detector.py        # 유동 경계 감지
│   │   ├── chunk_encoder.py            # 청크 → 개념
│   │   ├── competition.py              # 경쟁적 선택
│   │   ├── concept_reasoning.py        # 개념 추론 레이어
│   │   ├── next_concept_predictor.py   # 다음 개념 예측 (핵심)
│   │   ├── concept_decoder.py          # 개념 → 토큰 (보조)
│   │   └── concept_loss.py             # 손실 함수
│   ├── layers/
│   │   ├── mamba_block.py
│   │   ├── attention.py
│   │   └── gumbel.py
│   └── training/
│       └── trainer.py
├── scripts/
│   ├── train.py
│   ├── visualize_boundaries.py
│   └── analyze_concepts.py
└── tests/
```

---

## 참고 자료

| 자료 | URL |
|------|-----|
| Hymba 논문 | arXiv:2411.13676 |
| Hymba GitHub | github.com/NVlabs/hymba |
| SANC(E3) 논문 | arXiv:2601.08224 |
| Mamba GitHub | github.com/state-spaces/mamba |
| Mamba-2 논문 | arXiv:2405.21060 |
