---
title: "[시험 대비] 회귀/분류 연습 데이터셋 총정리 — sklearn, seaborn, UCI, Kaggle 즉시 사용 가이드"
date: 2026-04-22
tags: ["데이터분석", "데이터셋", "시험대비", "sklearn", "kaggle"]
categories: ["ML/AI"]
summary: "데이터 분석 시험 대비를 위한 회귀/분류 연습 데이터셋을 sklearn 내장부터 Kaggle까지 한 번에 정리. 각 데이터의 특징, 학습 포인트, 즉시 실행 가능한 베이스라인 코드 포함."
math: true
toc: true
draft: false
---

<!-- LEARNING-START -->
## 처음 읽는 분을 위한 안내

**학습 위치:** [직접 데이터로 예측해 보기]({{< relref "learn" >}}#data) · 1/5

가격을 맞히는 연습과 생존 여부를 구분하는 연습에는 서로 다른 데이터가 필요하다. 여기서는 어떤 예시로 무엇을 연습할 수 있는지 고른다. 데이터를 많이 모으는 것보다 이번에 익힐 예측 목표와 처리 과정을 먼저 정하면 좋다.

**먼저 알아둘 말**

- [데이터셋·라벨]({{< relref "ai-glossary" >}}#dataset): 데이터셋은 학습이나 평가에 쓰는 예시의 모음이다. 라벨은 예시에 붙인 목표 값이나 분류다.
- [회귀·분류]({{< relref "ai-glossary" >}}#regression): 회귀는 가격처럼 수치를 예측하고, 분류는 스팸 여부처럼 범주를 예측한다. 목표의 종류에 따라 평가 방식이 달라진다.
- [벤치마크·베이스라인]({{< relref "ai-glossary" >}}#benchmark): 벤치마크는 비교를 위한 시험이고, 베이스라인은 비교의 출발점이 되는 방법이다. 결과는 시험 조건과 함께 읽는다.

**이번 글의 목표:** 예측 목표에 맞는 연습 데이터 고르기.

**이 학습 경로의 앞 글:** [AI 기초 설명]({{< relref "ai-basics" >}}).

**본문에서 먼저 볼 순서:** **개요** → **1. 회귀 (Regression) 데이터셋** → **2. 분류 (Classification) 데이터셋**.

첫 독서에서는 데이터가 어떻게 바뀌고 무엇으로 평가하는지 읽자. 코드를 실행하려면 Python과 필요한 라이브러리를 익힌 뒤 한 단계씩 따라가면 된다.

<!-- PAPER-PREREQUISITES -->
### 선수 개념과 읽을 논문

**먼저 이해할 개념:** Python의 변수·리스트·함수, 데이터 표와 결측치, 학습/검증 분리, 평가 지표.

이 글의 출발점은 [AI 기초 설명]({{< relref "ai-basics" >}})과 위의 실습·시스템 개념이다. 학습 경로의 앞 글에서 입력·처리·출력의 흐름을 확인한 뒤 읽으면 된다.

<!-- PAPER-PREREQUISITES-END -->

<!-- LEARNING-START-END -->

---

## 개요

데이터 분석 시험을 준비하며 바로 써 볼 수 있는 회귀/분류 데이터셋을 모았다. 각 데이터에는 다음 내용을 붙였다.

- 어디서 어떻게 가져오는지
- 무엇을 학습할 수 있는지
- 베이스라인 코드 (복사해서 바로 실행)

연습 데이터를 고를 때는 크기보다 이번에 익힐 작업을 먼저 정하면 좋다. 범주형 변수가 많은 데이터에서는 인코딩을, 결측치가 있는 데이터에서는 대체 방법을, 불균형한 데이터에서는 평가 지표와 샘플링을 연습할 수 있다. 아래 베이스라인은 출발점을 만드는 코드다. 한 번에 여러 방법을 바꾸기보다 같은 검증 조건에서 하나씩 비교하면 변화의 원인을 찾기 쉽다.

---

## 1. 회귀 (Regression) 데이터셋

### 1.1 sklearn 내장 — 즉시 사용

```python
from sklearn.datasets import fetch_california_housing, load_diabetes
```

#### California Housing (가장 표준)

```python
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
print(X.shape)  # (20640, 8)
print(data.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
#  'Population', 'AveOccup', 'Latitude', 'Longitude']
```

| 항목 | 값 |
|------|-----|
| 샘플 수 | 20,640 |
| 특성 수 | 8 (모두 수치형) |
| 타겟 | 주택 중간 가격 (단위: $100,000) |
| 학습 포인트 | 큰 데이터, 다중공선성, 비선형 관계, 지리 정보 |

#### Diabetes (작은 회귀)

```python
data = load_diabetes(as_frame=True)
X, y = data.data, data.target
# 442 samples, 10 features
# 모든 특성이 이미 표준화됨
```

| 항목 | 값 |
|------|-----|
| 샘플 수 | 442 |
| 특성 수 | 10 (이미 표준화됨) |
| 타겟 | 1년 후 당뇨 진행도 (정량적) |
| 학습 포인트 | 작은 데이터, 선형 모델에 적합 |

> **참고**: Boston Housing은 인종 차별 이슈로 sklearn에서 deprecation됨. California Housing 사용 권장.

### 1.2 seaborn 내장

```python
import seaborn as sns
```

#### Tips (팁 금액 예측)

```python
df = sns.load_dataset('tips')
y = df['tip']
X = df.drop('tip', axis=1)
print(X.dtypes)
# total_bill    float64
# sex           category
# smoker        category
# day           category
# time          category
# size          int64
```

| 특징 | 학습 포인트 |
|------|-----------|
| 244 샘플 | 빠른 실험 |
| 범주형 4개 | **인코딩 연습** 필수 |
| 직관적 도메인 | 결과 해석 쉬움 |

#### MPG (자동차 연비)

```python
df = sns.load_dataset('mpg').dropna()
y = df['mpg']
X = df.drop(['mpg', 'name'], axis=1)
# 학습 포인트: 결측치, 범주형(origin), 시간(year), 다중공선성(displacement-cylinders)
```

### 1.3 UCI Repository

```bash
pip install ucimlrepo
```

```python
from ucimlrepo import fetch_ucirepo

# Wine Quality (회귀 또는 다중분류)
wine = fetch_ucirepo(id=186)
X, y = wine.data.features, wine.data.targets
# 1599+4898 샘플, 11 화학 특성
# 학습 포인트: 품질 점수 1-10 → 회귀 또는 분류 모두 가능

# Auto MPG
auto = fetch_ucirepo(id=9)
# 학습 포인트: 결측치, 범주형 혼합

# Concrete Compressive Strength
concrete = fetch_ucirepo(id=165)
# 학습 포인트: 모든 수치형, 공학 도메인, 비선형 관계
```

### 1.4 Kaggle 추천 (입문용)

| 데이터셋 | 특징 | 학습 포인트 |
|---------|------|-----------|
| **House Prices: Advanced Regression** | 80개 특성, 결측치 많음 | EDA, 특성 공학, 이상치, 로그 변환 |
| **Bike Sharing Demand** | 시계열 + 회귀 | 날짜 분해, 계절성 |
| **NYC Taxi Fare** | 5500만 행 대용량 | 메모리 효율, 거리 계산 |
| **Mercedes Benz Greener Manufacturing** | 익명 특성 | 차원 축소, PCA |

---

## 2. 분류 (Classification) 데이터셋

### 2.1 sklearn 내장

```python
from sklearn.datasets import (
    load_iris, load_breast_cancer, load_wine, load_digits, fetch_openml
)
```

#### Iris (다중 분류 클래식)

```python
data = load_iris(as_frame=True)
X, y = data.data, data.target
# 150 samples, 4 features, 3 classes (50/50/50 균형)
# 학습 포인트: 다중 클래스 베이스라인, 너무 쉬워서 과적합 주의
```

#### Breast Cancer (이진 분류)

```python
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
# 569 samples, 30 features
# 양성: 357 (62.7%), 악성: 212 (37.3%) — 약간 불균형
# 학습 포인트: 의료 도메인, 고차원, Recall 중요
```

#### Wine (다중 분류)

```python
data = load_wine(as_frame=True)
# 178 samples, 13 features, 3 classes
# 학습 포인트: 화학 성분 → 와인 종류 분류, 작은 데이터
```

#### Digits (10 클래스)

```python
data = load_digits(as_frame=True)
# 1797 samples, 64 features (8x8 픽셀)
# 학습 포인트: 다중 클래스, 시각화, 차원 축소(t-SNE)
```

### 2.2 seaborn 내장

#### Titanic (가장 유명한 입문 데이터)

```python
df = sns.load_dataset('titanic')
y = df['survived']
X = df.drop(['survived', 'alive'], axis=1)
print(df.isnull().sum())
# age          177
# embarked       2
# deck         688  (대부분 결측)
```

| 특징 | 학습 포인트 |
|------|-----------|
| 891 샘플 | 적당한 크기 |
| 결측치 多 | **결측 처리 종합 연습** |
| 수치형+범주형 혼합 | 인코딩 |
| 약간 불균형 (생존 38%) | stratify 필요 |
| 특성 공학 가능 | 가족 크기, 호칭, 객실 등급 등 |

**시험 종합 연습으로 1순위 추천**.

#### Penguins (3종 분류)

```python
df = sns.load_dataset('penguins').dropna()
y = df['species']
X = df.drop('species', axis=1)
# Adelie, Chinstrap, Gentoo 3종
# 학습 포인트: 새로운 Iris (Iris보다 약간 어려움)
```

### 2.3 UCI / OpenML

```python
from sklearn.datasets import fetch_openml
from ucimlrepo import fetch_ucirepo

# Adult Income (>50k 예측) - 가장 표준 불균형 분류
adult = fetch_openml('adult', version=2, as_frame=True)
X, y = adult.data, adult.target
# 48,842 행, 약 24% 양성
# 학습 포인트: 범주형 많음, 결측치, 클래스 불균형, 큰 데이터

# Bank Marketing
bank = fetch_ucirepo(id=222)
# 학습 포인트: 마케팅 전환율, 약 11% 양성, 불균형

# Mushroom (모두 범주형)
mushroom = fetch_ucirepo(id=73)
# 학습 포인트: 100% 범주형, 100% 분류 가능 (쉬운 데이터)

# Spambase (스팸 분류)
spam = fetch_ucirepo(id=94)
# 학습 포인트: 텍스트 분류 입문, 4601 행, 57 특성
```

### 2.4 Kaggle 추천

| 데이터셋 | 특징 | 학습 포인트 |
|---------|------|-----------|
| **Titanic** | 891 행 | 입문 표준, 결측치 처리 |
| **Credit Card Fraud** | 28만 행, **0.17% 사기** | **극단적 불균형** (SMOTE 필수) |
| **Heart Disease UCI** | 14 특성 | 의료 분류 |
| **Customer Churn (Telco)** | 7천 행 | 약간 불균형, 비즈니스 |
| **Home Credit Default Risk** | 30만 행, 다중 테이블 | 조인, 특성 공학 |
| **IEEE-CIS Fraud Detection** | 60만 행 | 고급 불균형 처리 |

---

## 3. 시험용 추천 우선순위

### 회귀 시험 대비

```python
# 1순위: California Housing — 실전형, 큰 데이터
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)

# 2순위: Tips — 인코딩 연습
import seaborn as sns
tips = sns.load_dataset('tips')

# 3순위: Auto MPG — 결측치 + 범주형 종합
from ucimlrepo import fetch_ucirepo
auto = fetch_ucirepo(id=9)
```

### 분류 시험 대비

```python
# 1순위: Titanic — 모든 전처리 기법 연습
import seaborn as sns
titanic = sns.load_dataset('titanic')

# 2순위: Adult Income — 불균형 + 큰 데이터
from sklearn.datasets import fetch_openml
adult = fetch_openml('adult', version=2, as_frame=True)

# 3순위: Breast Cancer — 의료 이진 분류
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer(as_frame=True)
```

---

## 4. 즉시 실행 가능한 베이스라인

### 4.1 회귀 베이스라인 (California Housing)

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# 1. 로드
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target

# 2. 분할
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 학습
model = RandomForestRegressor(
    n_estimators=200, max_depth=15,
    random_state=42, n_jobs=-1
)
model.fit(X_tr, y_tr)

# 4. 평가
pred = model.predict(X_te)
print(f"RMSE: {np.sqrt(mean_squared_error(y_te, pred)):.4f}")
print(f"MAE:  {mean_absolute_error(y_te, pred):.4f}")
print(f"R²:   {r2_score(y_te, pred):.4f}")
# 예상: RMSE ≈ 0.50, MAE ≈ 0.33, R² ≈ 0.81
```

### 4.2 분류 베이스라인 (Titanic)

```python
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score

# 1. 로드
df = sns.load_dataset('titanic')

# 2. 결측치 처리
df = df.drop(['deck', 'embark_town', 'alive', 'who', 'class', 'adult_male'], axis=1)
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df = df.dropna()

# 3. 인코딩
for col in ['sex', 'embarked']:
    df[col] = LabelEncoder().fit_transform(df[col])

y = df['survived']
X = df.drop('survived', axis=1)

# 4. 분할 (stratify 필수)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. 학습
model = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    class_weight='balanced',
    random_state=42, n_jobs=-1
)
model.fit(X_tr, y_tr)

# 6. 평가
pred = model.predict(X_te)
proba = model.predict_proba(X_te)[:, 1]

print(classification_report(y_te, pred))
print(f"AUC: {roc_auc_score(y_te, proba):.4f}")
# 예상: F1 ≈ 0.78, AUC ≈ 0.86
```

### 4.3 불균형 분류 베이스라인 (Adult Income)

```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

# 1. 로드
adult = fetch_openml('adult', version=2, as_frame=True)
X, y = adult.data, adult.target
y = (y == '>50K').astype(int)  # 이진 변환

# 2. 결측치 처리
X = X.replace('?', np.nan)
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].fillna(X[col].mode()[0])
for col in X.select_dtypes(include=[np.number]).columns:
    X[col] = X[col].fillna(X[col].median())

# 3. 인코딩
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# 4. 분할
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. 불균형 처리 (train만!)
smote = SMOTE(random_state=42)
X_tr_bal, y_tr_bal = smote.fit_resample(X_tr, y_tr)
print(f"Before: {y_tr.value_counts().to_dict()}")
print(f"After:  {y_tr_bal.value_counts().to_dict()}")

# 6. 학습
model = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    random_state=42, eval_metric='logloss', n_jobs=-1
)
model.fit(X_tr_bal, y_tr_bal)

# 7. 평가
pred = model.predict(X_te)
proba = model.predict_proba(X_te)[:, 1]

print(classification_report(y_te, pred))
print(f"AUC: {roc_auc_score(y_te, proba):.4f}")
# 예상: F1(class 1) ≈ 0.71, AUC ≈ 0.92
```

---

## 5. 데이터셋 특징별 매핑

### 학습하고 싶은 기법 → 추천 데이터

| 학습 목표 | 회귀 | 분류 |
|---------|------|------|
| **빠른 실험** | Diabetes | Iris, Cancer |
| **결측치 처리** | Auto MPG | Titanic |
| **인코딩** | Tips, MPG | Titanic, Adult |
| **불균형 처리** | - | Adult, Credit Fraud |
| **스케일링** | California Housing | Cancer |
| **이상치** | California Housing | - |
| **특성 공학** | House Prices (Kaggle) | Titanic |
| **시계열** | Bike Sharing | - |
| **고차원** | - | Cancer (30), Spam (57) |
| **대용량** | NYC Taxi | Higgs Boson |

---

## 6. 데이터 다운로드 가이드

### 6.1 패키지 설치

```bash
# 기본 (sklearn에 데이터 포함)
pip install scikit-learn

# Seaborn 데이터
pip install seaborn

# UCI 데이터셋
pip install ucimlrepo

# Kaggle (가입 + API 키 필요)
pip install kaggle

# imbalanced-learn (SMOTE 등)
pip install imbalanced-learn
```

### 6.2 Kaggle 설정

```bash
# 1. https://www.kaggle.com/account 에서 API Token 다운로드
# 2. ~/.kaggle/kaggle.json 위치에 배치
# 3. 권한 설정
chmod 600 ~/.kaggle/kaggle.json

# 4. 데이터 다운로드
kaggle competitions download -c titanic
kaggle datasets download -d uciml/breast-cancer-wisconsin-data
```

### 6.3 OpenML 직접 접근

```python
from sklearn.datasets import fetch_openml

# 1000개 이상의 데이터셋 접근 가능
data = fetch_openml(name='credit-g', version=1, as_frame=True)
data = fetch_openml(name='blood-transfusion-service-center', as_frame=True)
data = fetch_openml(name='kc_housing', as_frame=True)
```

---

## 7. 시험 시뮬레이션 워크플로우

```python
# === Step 1: 데이터 로드 + EDA ===
import seaborn as sns
import pandas as pd

df = sns.load_dataset('titanic')
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df['survived'].value_counts(normalize=True))

# === Step 2: 전처리 ===
# (블로그의 회귀/분류 가이드 참조)

# === Step 3: 베이스라인 모델 ===
# RandomForest로 빠르게 시작

# === Step 4: 평가 ===
# 회귀: RMSE, R²
# 분류: F1, AUC, classification_report

# === Step 5: 개선 ===
# - 하이퍼파라미터 튜닝 (GridSearch/Optuna)
# - 더 강력한 모델 (XGBoost, LightGBM)
# - 특성 공학
# - 앙상블

# === Step 6: 제출 ===
submission = pd.DataFrame({'id': test_ids, 'target': predictions})
submission.to_csv('submission.csv', index=False)
```

---

## 8. 시험 직전 체크리스트

```
☐ sklearn 내장 데이터 로딩 가능 (fetch_*, load_*)
☐ seaborn Titanic 전처리 익숙
☐ 결측치 처리 3가지 방법 (median, mode, KNN)
☐ 범주형 인코딩 2가지 (Label, OneHot)
☐ Stratify 옵션 기억 (분류 필수)
☐ class_weight='balanced' 사용법
☐ SMOTE는 train에만 적용
☐ 회귀 평가: RMSE, R²
☐ 분류 평가: classification_report, AUC
☐ RandomForest 기본 사용법
☐ XGBoost 기본 사용법
☐ KFold / StratifiedKFold
☐ GridSearchCV 기본 패턴
```

---

## 9. 관련 블로그 포스트

- [회귀 문제 완전 가이드](data-analysis-regression-guide.md) — 전처리, 모델, 평가 종합
- [분류 문제 완전 가이드](data-analysis-classification-guide.md) — 불균형 처리 포함

<!-- LEARNING-END -->
---

## 읽은 뒤 확인하고 다음으로

**확인 질문:** 예측 목표에 맞는 연습 데이터 고르기를 목표로, 본문에서 다룬 입력·중간 과정·결과를 자기 말로 설명해 보자. 실험이나 사례가 있다면 어떤 조건에서 확인했는지도 덧붙여 보자.

**다음 글:** [\[시험 대비\] 데이터 분석 회귀 문제 완전 가이드 — 전처리, 라이브러리, 모델, 평가]({{< relref "posts/data-analysis-regression-guide" >}}) — 수치 예측의 전처리·학습·평가 연결하기.

[전체 학습 순서]({{< relref "learn" >}}) · [용어 사전]({{< relref "ai-glossary" >}})

<!-- LEARNING-END-END -->
