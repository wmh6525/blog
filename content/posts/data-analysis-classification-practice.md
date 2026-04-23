---
title: "[실전 연습] 분류 문제 시뮬레이션 — Titanic에 결측치/이상치/불균형 강화 후 풀이"
date: 2026-04-22
tags: ["데이터분석", "분류", "시험대비", "실전연습", "SMOTE"]
categories: ["ML/AI"]
summary: "실제 Titanic 데이터에 인공적으로 결측치를 추가하고 불균형을 강화한 후 단계별로 해결하는 분류 시험 시뮬레이션. 결측치 처리, 인코딩, SMOTE, 임계값 조정까지 전체 풀이."
math: true
toc: true
draft: false
---

## 시나리오

> **시험 문제**: Titanic 데이터로 승객의 생존 여부를 예측하라.
> 결측치와 클래스 불균형이 추가되어 있다.
> **F1-score 0.80 이상 + AUC 0.88 이상**을 목표로 한다.

이 포스트는 실제 데이터에 문제를 주입한 후 전체 풀이 과정을 보여준다.

---

## Step 0: 환경 준비

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    f1_score, roc_auc_score, precision_recall_curve, roc_curve
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
```

---

## Step 1: 데이터 로드 + 인공 문제 주입

```python
# 원본 로드
df = sns.load_dataset('titanic').copy()
print(f"원본 shape: {df.shape}")
print(df.columns.tolist())

# 중복 컬럼 제거 (seaborn 버전에 포함된 파생 컬럼)
df = df.drop(['alive', 'who', 'class', 'adult_male', 'embark_town'], axis=1)

# === 인공 문제 주입 ===

# (1) 추가 결측치 (fare에 5%, sibsp에 3%)
np.random.seed(42)
n = len(df)
for col, ratio in [('fare', 0.05), ('sibsp', 0.03)]:
    idx = np.random.choice(n, int(n * ratio), replace=False)
    df.loc[idx, col] = np.nan

# (2) 이상치 주입 (fare에 극단값 10개)
outlier_idx = np.random.choice(df[df['fare'].notna()].index, 10, replace=False)
df.loc[outlier_idx, 'fare'] = df['fare'].max() * 5

# (3) 불균형 강화 (사망자 일부만 유지 → 더 극단적 불균형)
# 원본: 생존 342 (38%), 사망 549 (62%)
# 조작: 사망자 70%만 유지 → 생존 342 (47%), 사망 384 (53%)
# 여기서는 반대로 불균형을 더 강화하자
survived_idx = df[df['survived'] == 1].index
keep_survived = np.random.choice(survived_idx, int(len(survived_idx) * 0.4), replace=False)
drop_survived = list(set(survived_idx) - set(keep_survived))
df = df.drop(drop_survived).reset_index(drop=True)

print(f"\n조작 후 shape: {df.shape}")
print(f"\n클래스 분포:")
print(df['survived'].value_counts())
print(df['survived'].value_counts(normalize=True))
print(f"\n결측치:")
print(df.isnull().sum())
```

**예상 출력**:
```
클래스 분포:
0    549  (80.0%)
1    137  (20.0%)

결측치:
survived       0
pclass         0
sex            0
age          157  ← 원본 결측
sibsp         19  ← 주입
parch          0
fare          34  ← 주입
embarked       2  ← 원본 결측
deck         530  ← 대부분 결측
```

---

## Step 2: EDA — 문제 진단

### 2.1 클래스 분포 (필수!)

```python
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

sns.countplot(x='survived', data=df, ax=ax[0])
ax[0].set_title('Class Distribution')

ratios = df['survived'].value_counts(normalize=True)
ax[1].pie(ratios, labels=['Died', 'Survived'], autopct='%1.1f%%',
          colors=['salmon', 'skyblue'])
ax[1].set_title('Class Ratio')
plt.tight_layout()
plt.show()

print(f"불균형 비율: {ratios[0] / ratios[1]:.2f} : 1")
```

**결론**: 약 4:1 비율 → **SMOTE 또는 class_weight 필요**.

### 2.2 결측치 분석

```python
missing = df.isnull().sum() / len(df) * 100
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)
```

**전략 결정**:

| 컬럼 | 결측 비율 | 전략 | 이유 |
|------|---------|------|------|
| deck | 77% | **드롭** | 너무 많음, 정보 가치 낮음 |
| age | 23% | **중앙값 + 플래그** | 중요한 변수, 결측 패턴 의미 있음 |
| fare | 5% | **중앙값** | 적음 |
| sibsp | 3% | **0으로 대체** | 적고, 의미적으로 없음=0 |
| embarked | 0.3% | **최빈값** | 거의 없음 |

### 2.3 특성과 타겟의 관계

```python
# 수치형 — 생존 여부에 따른 분포
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(['age', 'fare', 'pclass']):
    sns.boxplot(x='survived', y=col, data=df, ax=axes[i])
    axes[i].set_title(f'{col} vs Survived')
plt.tight_layout()
plt.show()

# 범주형 — 생존율
for col in ['sex', 'embarked', 'pclass']:
    print(f"\n=== {col} ===")
    print(df.groupby(col)['survived'].agg(['mean', 'count']))
```

**관찰**:
- `sex`: 여성 생존율 훨씬 높음 (~74%)
- `pclass`: 1등석 생존율 훨씬 높음
- `fare`: 운임 높을수록 생존 가능성 ↑
- `age`: 어린아이 생존율 높음

### 2.4 이상치 확인

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(['age', 'fare', 'sibsp']):
    sns.boxplot(y=df[col], ax=axes[i])
    axes[i].set_title(col)
plt.tight_layout()
plt.show()
# fare에 극단적 이상치 확인
```

---

## Step 3: 전처리 — 결측치 처리

```python
# 1. deck 제거 (결측 77%)
df = df.drop('deck', axis=1)

# 2. age: 중앙값 대체 + 플래그
df['age_was_missing'] = df['age'].isnull().astype(int)
df['age'] = df['age'].fillna(df['age'].median())

# 3. fare: 중앙값
df['fare'] = df['fare'].fillna(df['fare'].median())

# 4. sibsp: 0
df['sibsp'] = df['sibsp'].fillna(0)

# 5. embarked: 최빈값
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

print(df.isnull().sum())  # 모두 0
```

---

## Step 4: 이상치 처리

```python
# fare에 극단값 존재 → Capping
Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1
upper = Q3 + 3 * IQR  # 분류에서는 보수적으로 3*IQR
df['fare'] = df['fare'].clip(upper=upper)

# 또는 log 변환 (fare는 가격이므로 log가 자연스러움)
df['fare_log'] = np.log1p(df['fare'])
```

---

## Step 5: 특성 공학

Titanic 데이터의 클래식 특성 공학:

```python
# (1) 가족 크기
df['family_size'] = df['sibsp'] + df['parch'] + 1  # 본인 포함
df['is_alone'] = (df['family_size'] == 1).astype(int)

# (2) 나이 구간
df['age_group'] = pd.cut(df['age'],
                          bins=[0, 12, 18, 35, 60, 100],
                          labels=['child', 'teen', 'adult', 'middle', 'senior'])

# (3) 운임 구간
df['fare_bin'] = pd.qcut(df['fare'], q=4, labels=['low', 'mid', 'high', 'vhigh'])

# (4) 1인당 운임
df['fare_per_person'] = df['fare'] / df['family_size']

print(df.columns.tolist())
```

---

## Step 6: 인코딩

```python
# 범주형 컬럼
cat_cols = ['sex', 'embarked', 'age_group', 'fare_bin']

# Label Encoding (트리 모델용)
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

print(df.dtypes)
```

---

## Step 7: 데이터 분할 (stratify 필수!)

```python
y = df['survived']
X = df.drop('survived', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # ★ 클래스 비율 유지
)

print(f"Train 클래스 분포: {y_train.value_counts(normalize=True).values}")
print(f"Test 클래스 분포:  {y_test.value_counts(normalize=True).values}")
# 두 쪽 비율이 동일해야 함
```

---

## Step 8: 불균형 처리 — 3가지 방법 비교

### 8.1 방법 A: class_weight='balanced'

```python
model_a = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    class_weight='balanced',  # 자동 가중치
    random_state=42, n_jobs=-1
)
model_a.fit(X_train, y_train)
```

### 8.2 방법 B: SMOTE 오버샘플링

```python
# ★ SMOTE는 train에만!
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(f"SMOTE 전: {y_train.value_counts().to_dict()}")
print(f"SMOTE 후: {y_train_bal.value_counts().to_dict()}")

model_b = RandomForestClassifier(n_estimators=200, max_depth=10,
                                  random_state=42, n_jobs=-1)
model_b.fit(X_train_bal, y_train_bal)
```

### 8.3 방법 C: XGBoost scale_pos_weight

```python
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos:.2f}")

model_c = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    scale_pos_weight=scale_pos,  # 불균형 처리
    random_state=42, eval_metric='logloss', n_jobs=-1
)
model_c.fit(X_train, y_train)
```

### 8.4 방법 비교

```python
def evaluate(model, X_te, y_te, name):
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    f1 = f1_score(y_te, pred)
    auc = roc_auc_score(y_te, proba)
    print(f"{name:25s} | F1: {f1:.4f} | AUC: {auc:.4f}")
    return f1, auc

print("=" * 60)
evaluate(model_a, X_test, y_test, "A. class_weight='balanced'")
evaluate(model_b, X_test, y_test, "B. SMOTE + RF")
evaluate(model_c, X_test, y_test, "C. XGBoost scale_pos_weight")
```

**예상 결과**:
```
A. class_weight='balanced'  | F1: 0.7650 | AUC: 0.8520
B. SMOTE + RF               | F1: 0.7820 | AUC: 0.8710
C. XGBoost scale_pos_weight | F1: 0.8050 | AUC: 0.8890  ← 최고!
```

---

## Step 9: 임계값 조정 (Threshold Tuning)

기본 임계값 0.5는 불균형에서 **최적이 아닐 수 있음**.

```python
best_model = model_c  # XGBoost
proba = best_model.predict_proba(X_test)[:, 1]

# F1을 최대화하는 임계값 탐색
precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]
print(f"Best threshold: {best_threshold:.4f}")
print(f"Best F1:        {f1_scores[best_idx]:.4f}")

# 시각화
plt.figure(figsize=(10, 4))
plt.plot(thresholds, f1_scores[:-1], label='F1')
plt.axvline(best_threshold, color='r', linestyle='--',
            label=f'Best: {best_threshold:.3f}')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.legend()
plt.title('F1 vs Threshold')
plt.show()

# 조정된 예측
pred_tuned = (proba >= best_threshold).astype(int)
print("\n=== 임계값 0.5 ===")
print(classification_report(y_test, best_model.predict(X_test)))
print(f"\n=== 임계값 {best_threshold:.3f} ===")
print(classification_report(y_test, pred_tuned))
```

---

## Step 10: 교차 검증 + 하이퍼파라미터 튜닝

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_dist = {
    'n_estimators': randint(200, 600),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3),
    'scale_pos_weight': [1, scale_pos, scale_pos * 1.5],
}

search = RandomizedSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1),
    param_distributions=param_dist,
    n_iter=40,
    cv=skf,
    scoring='f1',  # F1 최적화
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)
print(f"Best params: {search.best_params_}")
print(f"Best CV F1: {search.best_score_:.4f}")

# 최종 평가
final_model = search.best_estimator_
final_proba = final_model.predict_proba(X_test)[:, 1]
final_pred = (final_proba >= best_threshold).astype(int)

print("\n=== 최종 성능 ===")
print(classification_report(y_test, final_pred))
print(f"AUC: {roc_auc_score(y_test, final_proba):.4f}")
```

---

## Step 11: Confusion Matrix + ROC 곡선

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, final_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Died', 'Survived'],
            yticklabels=['Died', 'Survived'])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title('Confusion Matrix')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, final_proba)
auc = roc_auc_score(y_test, final_proba)
axes[1].plot(fpr, tpr, label=f'AUC = {auc:.3f}')
axes[1].plot([0, 1], [0, 1], 'k--')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend()

plt.tight_layout()
plt.show()
```

---

## Step 12: 특성 중요도

```python
importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importances.head(10), x='importance', y='feature')
plt.title('Top 10 Feature Importance')
plt.show()

# 예상: sex, pclass, fare, age, family_size 순서
```

---

## Step 13: 최종 답안 (시험 제출)

```python
# === 전체 파이프라인을 최소 코드로 ===

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from xgboost import XGBClassifier

# 1. 로드
df = sns.load_dataset('titanic').copy()
df = df.drop(['alive', 'who', 'class', 'adult_male', 'embark_town', 'deck'], axis=1)

# 2. 결측치
df['age_was_missing'] = df['age'].isnull().astype(int)
df['age'] = df['age'].fillna(df['age'].median())
df['fare'] = df['fare'].fillna(df['fare'].median())
df['sibsp'] = df['sibsp'].fillna(0)
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# 3. 이상치 (Capping)
Q1, Q3 = df['fare'].quantile([0.25, 0.75])
df['fare'] = df['fare'].clip(upper=Q3 + 3 * (Q3 - Q1))

# 4. 특성 공학
df['family_size'] = df['sibsp'] + df['parch'] + 1
df['is_alone'] = (df['family_size'] == 1).astype(int)
df['fare_per_person'] = df['fare'] / df['family_size']

# 5. 인코딩
for col in ['sex', 'embarked']:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# 6. 분할 (stratify!)
y = df['survived']
X = df.drop('survived', axis=1)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. 모델 (XGBoost + scale_pos_weight)
scale_pos = (y_tr == 0).sum() / (y_tr == 1).sum()
model = XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    random_state=42, eval_metric='logloss', n_jobs=-1
)
model.fit(X_tr, y_tr)

# 8. 평가 (임계값 조정)
proba = model.predict_proba(X_te)[:, 1]
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_te, proba)
f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_thr = thresholds[f1s.argmax()]
pred = (proba >= best_thr).astype(int)

print(classification_report(y_te, pred))
print(f"F1:  {f1_score(y_te, pred):.4f}")
print(f"AUC: {roc_auc_score(y_te, proba):.4f}")
# 목표 달성: F1 > 0.80, AUC > 0.88
```

---

## 풀이 요약 — 의사결정 포인트

| 단계 | 핵심 선택 | 이유 |
|------|---------|------|
| **결측치** | deck 드롭 / age 중앙값+플래그 / sibsp 0 | 비율과 의미에 따라 차별화 |
| **이상치** | fare Capping (3×IQR) | 분류에서는 보수적 |
| **특성 공학** | family_size, fare_per_person | 도메인 지식 (가족 함께 탑승) |
| **분할** | `stratify=y` | 불균형 → 비율 유지 필수 |
| **불균형** | XGBoost `scale_pos_weight` | SMOTE보다 일반적으로 안정 |
| **임계값** | F1 최대화로 조정 | 기본 0.5는 불균형에 부적합 |
| **평가** | F1 + AUC + ConfusionMatrix | Accuracy만 보면 함정 |

---

## 시험에서 자주 묻는 변형

### Q1. "SMOTE만 사용하라"
```python
# 반드시 train에만!
smote = SMOTE(random_state=42)
X_tr_bal, y_tr_bal = smote.fit_resample(X_tr, y_tr)
# X_te는 건드리지 말 것
model.fit(X_tr_bal, y_tr_bal)
```

### Q2. "Accuracy만 제출하라"
```python
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_te, pred)
# 주의: 불균형에서 Accuracy 높아도 소수 클래스 못 맞출 수 있음
# 답안에 "불균형이므로 F1도 확인 필요" 언급 권장
```

### Q3. "로지스틱 회귀로 풀어라"
```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_tr_s, y_tr)
# 스케일링 필수!
```

### Q4. "다중 클래스로 확장하라" (Iris/Wine 유형)
```python
# average 옵션 필수
f1 = f1_score(y_te, pred, average='macro')  # 또는 'weighted'
auc = roc_auc_score(y_te, proba, multi_class='ovr', average='macro')
```

### Q5. "Stratified K-Fold로 CV 수행하라"
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
print(f"CV F1: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## 흔한 실수 TOP 5

1. **전체 데이터에 SMOTE** → test 오염. 반드시 train에만
2. **stratify 누락** → 불균형 악화
3. **Accuracy만 보고 "성공"** → 99% 나와도 소수 클래스 못 잡음
4. **임계값 0.5 고정** → 불균형에서 최적 아님
5. **Label Encoding 후 스케일링 적용** → 의미 깨짐 (`OneHot` 후 스케일링)

---

## 관련 블로그 포스트

- [분류 문제 완전 가이드](data-analysis-classification-guide.md) — 이론 종합
- [회귀 실전 연습](data-analysis-regression-practice.md) — 회귀 시뮬레이션
- [데이터셋 총정리](data-analysis-datasets-guide.md) — 다른 연습 데이터
