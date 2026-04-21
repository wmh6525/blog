---
title: "[시험 대비] 데이터 분석 분류 문제 완전 가이드 — 전처리, 라이브러리, 모델, 평가"
date: 2026-04-21
tags: ["데이터분석", "분류", "시험대비", "머신러닝", "sklearn"]
categories: ["ML/AI"]
summary: "데이터 분석 시험의 분류(Classification) 문제 대비. 결측치/이상치 처리, 클래스 불균형 해결(SMOTE), 인코딩, Logistic/RandomForest/XGBoost/LightGBM 모델, Accuracy/F1/AUC 평가까지 총정리."
math: true
toc: true
draft: false
---

## 분류 문제의 표준 파이프라인

```
데이터 로드 → EDA → 클래스 분포 확인 → 결측치 처리 → 이상치 처리
  → 인코딩 → 스케일링 → 불균형 처리 → 모델 학습
  → Stratified CV → 하이퍼파라미터 튜닝 → 평가(F1, AUC) → 예측
```

**회귀와의 핵심 차이**:
- 타겟이 **이산 클래스** (0/1, 또는 다중 클래스)
- **클래스 불균형** 처리 필수
- Stratified K-Fold 사용
- 평가 지표: F1, AUC, Confusion Matrix 중심

---

## 1. 필수 라이브러리

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 전처리
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)

# 불균형 처리
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

# 분류 모델
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, VotingClassifier, StackingClassifier
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# 부스팅
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 평가
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, precision_recall_curve, average_precision_score
)

# 검증
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, cross_validate,
    GridSearchCV, RandomizedSearchCV
)

# 파이프라인
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
```

설치:
```bash
pip install scikit-learn xgboost lightgbm catboost imbalanced-learn
```

---

## 2. EDA (분류 특화)

### 2.1 클래스 분포 확인 (필수!)

```python
# 빈도
df['target'].value_counts()
df['target'].value_counts(normalize=True)  # 비율

# 시각화
sns.countplot(x='target', data=df)
plt.title('Class Distribution')
plt.show()

# 불균형 비율
minority = df['target'].value_counts().min()
majority = df['target'].value_counts().max()
print(f"Imbalance Ratio: {majority / minority:.2f}")
```

### 2.2 불균형 판단 기준

| 비율 | 상태 | 조치 |
|------|------|------|
| 1:1 ~ 1:3 | 균형 | 특별 처리 불필요 |
| 1:3 ~ 1:10 | 경미한 불균형 | `class_weight='balanced'` |
| 1:10 ~ 1:100 | 심각한 불균형 | SMOTE, 언더샘플링 |
| 1:100 이상 | 극단적 불균형 | 이상 탐지 문제로 접근 |

### 2.3 특성과 타겟의 관계

```python
# 수치형 — 클래스별 분포
for col in num_cols:
    sns.boxplot(x='target', y=col, data=df)
    plt.show()

# 범주형 — 크로스탭
pd.crosstab(df['cat_col'], df['target'], normalize='index')

# 범주형 — stacked bar
pd.crosstab(df['cat_col'], df['target']).plot(kind='bar', stacked=True)
```

---

## 3. 결측치 처리

회귀와 동일하지만 주의사항:

### 3.1 기본 방법

```python
# 수치형: 중앙값
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# 범주형: 최빈값 또는 'Missing' 카테고리
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna('Missing')
```

### 3.2 클래스별 평균으로 대체 (고급)

```python
# 각 클래스별 중앙값으로 결측치 채움
for col in num_cols:
    df[col] = df.groupby('target')[col].transform(lambda x: x.fillna(x.median()))
```

**주의**: Train에만 적용. Test에는 전체 train의 값으로.

### 3.3 KNN Imputer

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df[num_cols] = imputer.fit_transform(df[num_cols])
```

### 3.4 결측 플래그 (중요할 때)

```python
# 결측 자체가 정보인 경우
df['col_is_missing'] = df['col'].isnull().astype(int)
df['col'].fillna(df['col'].median(), inplace=True)
```

---

## 4. 이상치 처리 (회귀와 동일)

```python
# IQR
Q1, Q3 = df['col'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df['col'] = df['col'].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

# 분류에서는 이상치가 중요한 경우 많음 (사기 탐지 등)
# → 함부로 제거하지 말고 플래그만 추가하는 것도 방법
df['col_outlier'] = ((df['col'] < lower) | (df['col'] > upper)).astype(int)
```

---

## 5. 인코딩

회귀와 동일하지만 **Target Encoding**에 주의:

### 5.1 Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['encoded'] = le.fit_transform(df['cat_col'])
```

### 5.2 One-Hot Encoding

```python
df_encoded = pd.get_dummies(df, columns=['cat_col'], drop_first=True)
```

### 5.3 Target Encoding (데이터 누수 주의!)

```python
from category_encoders import TargetEncoder

# 반드시 K-Fold 내에서 각 fold마다 별도 fit
te = TargetEncoder()

# Train에서만 fit
X_train['encoded'] = te.fit_transform(X_train['cat_col'], y_train)
X_val['encoded'] = te.transform(X_val['cat_col'])
```

**분류에서 TE가 효과적인 경우**: 카테고리 수가 매우 많을 때 (>50).

---

## 6. 스케일링 (회귀와 동일)

- 로지스틱 회귀, SVM, KNN, NN → **필수**
- 트리 모델 → 불필요

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # transform만!
```

---

## 7. 클래스 불균형 처리 (분류 핵심)

### 7.1 `class_weight='balanced'` (가장 간단)

```python
# 각 클래스 가중치를 자동으로 역비례
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model = LogisticRegression(class_weight='balanced')
model = SVC(class_weight='balanced')

# 수동 설정
model = RandomForestClassifier(class_weight={0: 1, 1: 10})
```

**원리**: 소수 클래스에 더 큰 가중치 부여 → 손실 함수에서 비중 증가.

### 7.2 SMOTE (Synthetic Minority Over-sampling)

```python
from imblearn.over_sampling import SMOTE

# 훈련 데이터에만 적용 (test에는 절대 사용 금지!)
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Before: {y_train.value_counts().to_dict()}")
print(f"After:  {y_train_balanced.value_counts().to_dict()}")
```

**원리**: 소수 클래스 샘플 사이에서 **보간하여 새 샘플 생성**.

### 7.3 SMOTE 변형들

```python
from imblearn.over_sampling import ADASYN, BorderlineSMOTE

# ADASYN: 분류가 어려운 경계 영역에 더 많이 생성
adasyn = ADASYN(random_state=42)

# BorderlineSMOTE: 결정 경계 근처만 오버샘플링
borderline = BorderlineSMOTE(random_state=42)
```

### 7.4 언더샘플링

```python
from imblearn.under_sampling import RandomUnderSampler, TomekLinks

# 무작위 다수 클래스 제거
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

# Tomek Links: 경계 근처 다수 샘플 제거
tomek = TomekLinks()
```

**주의**: 정보 손실 — 데이터가 많을 때만.

### 7.5 결합 방법 (추천)

```python
from imblearn.combine import SMOTETomek, SMOTEENN

# SMOTE + Tomek Links (오버 후 정리)
smt = SMOTETomek(random_state=42)
X_balanced, y_balanced = smt.fit_resample(X_train, y_train)

# SMOTE + ENN (더 강력한 정리)
sme = SMOTEENN(random_state=42)
```

### 7.6 중요한 함정: 파이프라인 내에서 적용

```python
from imblearn.pipeline import Pipeline as ImbPipeline

# ❌ 잘못된 방법 (데이터 누수)
# X_resampled, y_resampled = smote.fit_resample(X, y)  # 전체 데이터에!
# cross_val_score(model, X_resampled, y_resampled, cv=5)  # CV의 validation에도 SMOTE 적용됨

# ✅ 올바른 방법
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])
# CV의 각 fold에서만 SMOTE 적용됨 (validation에는 적용 X)
cross_val_score(pipeline, X, y, cv=5, scoring='f1')
```

### 7.7 임계값 조정 (Threshold Tuning)

```python
# 기본은 0.5 — 불균형에서는 부적합할 수 있음
y_proba = model.predict_proba(X_test)[:, 1]

# F1을 최대화하는 임계값 찾기
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_threshold = thresholds[f1_scores.argmax()]

y_pred = (y_proba >= best_threshold).astype(int)
```

---

## 8. 분류 모델

### 8.1 모델별 특징

| 모델 | 특징 | 언제 |
|------|------|------|
| **LogisticRegression** | 선형, 확률 출력, 해석 쉬움 | 베이스라인 |
| **KNN** | 간단, 느림 | 소규모 데이터 |
| **SVM** | 강력, 커널 트릭 | 중규모, 고차원 |
| **GaussianNB** | 초고속, 조건부 독립 가정 | 텍스트, 기준선 |
| **DecisionTree** | 해석 가능, 과적합 | 교육용 |
| **RandomForest** | 강건, 과적합 덜함 | **시험 필수** |
| **GradientBoosting** | sklearn 기본 부스팅 | 중간 성능 |
| **XGBoost** | 성능 탑, 결측 자동 처리 | **표준 선택** |
| **LightGBM** | 빠름, 대용량 | 대규모 데이터 |
| **CatBoost** | 범주형 자동 처리 | 범주 많을 때 |

### 8.2 기본 사용법

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Stratify 필수 (클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
```

### 8.3 주요 모델 코드

```python
# LogisticRegression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(
    C=1.0,
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

# SVM
from sklearn.svm import SVC
model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    probability=True,  # predict_proba 사용 위해
    random_state=42
)

# RandomForest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# XGBoost
from xgboost import XGBClassifier
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
    eval_metric='logloss',
    random_state=42,
    early_stopping_rounds=50,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# LightGBM
from lightgbm import LGBMClassifier
model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    class_weight='balanced',
    random_state=42,
)

# CatBoost
from catboost import CatBoostClassifier
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    class_weights=[1, 10],
    cat_features=cat_cols,  # 자동 인코딩!
    verbose=False,
    random_state=42
)
```

---

## 9. 평가 지표 (분류)

### 9.1 Confusion Matrix

```
                  예측
              0         1
실제 0    [TN]      [FP]
실제 1    [FN]      [TP]
```

- **TP (True Positive)**: 1을 1로 맞춤
- **TN (True Negative)**: 0을 0으로 맞춤
- **FP (False Positive)**: 0을 1로 오분류 (Type I Error)
- **FN (False Negative)**: 1을 0으로 오분류 (Type II Error)

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()

# 수치
tn, fp, fn, tp = cm.ravel()
```

### 9.2 주요 지표

| 지표 | 수식 | 의미 |
|------|------|------|
| **Accuracy** | $\frac{TP+TN}{전체}$ | 전체 중 맞춘 비율 (불균형에 부적합!) |
| **Precision** | $\frac{TP}{TP+FP}$ | 1 예측 중 실제 1 비율 |
| **Recall** (Sensitivity) | $\frac{TP}{TP+FN}$ | 실제 1 중 맞춘 비율 |
| **F1** | $\frac{2 \cdot P \cdot R}{P + R}$ | Precision & Recall 조화평균 |
| **Specificity** | $\frac{TN}{TN+FP}$ | 실제 0 중 맞춘 비율 |
| **AUC-ROC** | ROC 곡선 아래 면적 | 임계값 무관 성능, 0.5~1 |

### 9.3 어느 지표를 볼까?

| 상황 | 우선 지표 |
|------|---------|
| 클래스 균형 | Accuracy, F1 |
| 클래스 불균형 | **F1, AUC-ROC** |
| 극단적 불균형 (사기 탐지) | **PR-AUC, Recall** |
| 위양성이 치명적 (스팸) | **Precision** |
| 위음성이 치명적 (암 진단) | **Recall** |
| 전반적 비교 | **AUC-ROC** |

### 9.4 코드

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)

# 단일 지표
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1:        {f1_score(y_test, y_pred):.4f}")
print(f"AUC:       {roc_auc_score(y_test, y_proba):.4f}")

# 전체 리포트 (클래스별)
print(classification_report(y_test, y_pred))
```

### 9.5 다중 클래스

```python
# Macro: 클래스별 지표의 단순 평균 (균형)
# Micro: 전체 샘플 기준 (불균형 반영)
# Weighted: 클래스별 샘플 수로 가중 평균

f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

# 다중 클래스 AUC
auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
```

### 9.6 ROC / PR 곡선

```python
from sklearn.metrics import roc_curve, precision_recall_curve

# ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_proba):.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

# PR (불균형 때 더 유용)
precisions, recalls, _ = precision_recall_curve(y_test, y_proba)
plt.plot(recalls, precisions)
plt.xlabel('Recall')
plt.ylabel('Precision')
```

---

## 10. Stratified K-Fold (분류 필수)

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    model, X, y, cv=skf,
    scoring='f1',  # 또는 'roc_auc', 'accuracy'
    n_jobs=-1
)
print(f"F1: {scores.mean():.4f} ± {scores.std():.4f}")
```

**일반 K-Fold 대신 StratifiedKFold를 써야 하는 이유**:
- 각 fold의 클래스 비율을 유지
- 불균형 데이터에서 특히 중요

---

## 11. 하이퍼파라미터 튜닝

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1],
    'scale_pos_weight': [1, 5, 10],  # 불균형용
}

grid = GridSearchCV(
    XGBClassifier(random_state=42),
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1',  # 분류 불균형 시 F1 또는 roc_auc
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

print(f"Best: {grid.best_params_}")
print(f"Score: {grid.best_score_:.4f}")
```

---

## 12. 앙상블

### 12.1 Voting

```python
from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(estimators=[
    ('lr', LogisticRegression(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42)),
    ('xgb', XGBClassifier(random_state=42)),
], voting='soft')  # 확률 평균 (hard는 다수결)

voting.fit(X_train, y_train)
```

### 12.2 Stacking

```python
from sklearn.ensemble import StackingClassifier

stacking = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(random_state=42)),
        ('xgb', XGBClassifier(random_state=42)),
        ('lgb', LGBMClassifier(random_state=42)),
    ],
    final_estimator=LogisticRegression(),
    cv=5
)
```

---

## 13. 파이프라인 (완성형)

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer

num_cols = ['age', 'income']
cat_cols = ['city', 'gender']

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols),
])

# imblearn Pipeline 사용 (SMOTE 포함)
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(random_state=42))
])

# CV
scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1')
```

---

## 14. 시험 답안 템플릿

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE

# 1. 로드
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 2. 타겟 분리
y = train['target']
X = train.drop(['target', 'id'], axis=1)

# 3. 결측치
num_cols = X.select_dtypes(include=[np.number]).columns
cat_cols = X.select_dtypes(include=['object']).columns

X[num_cols] = X[num_cols].fillna(X[num_cols].median())
test[num_cols] = test[num_cols].fillna(X[num_cols].median())

X[cat_cols] = X[cat_cols].fillna('Missing')
test[cat_cols] = test[cat_cols].fillna('Missing')

# 4. 인코딩
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col].astype(str), test[col].astype(str)])
    le.fit(combined)
    X[col] = le.transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# 5. 분할 (stratify 필수)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. 불균형 처리 (train에만!)
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# 7. 모델
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_bal, y_train_bal)

# 8. 평가
y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]

print(classification_report(y_val, y_pred))
print(f"F1:  {f1_score(y_val, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_val, y_proba):.4f}")

# 9. 예측 + 제출
test_pred = model.predict(test)
submission = pd.DataFrame({'id': test['id'], 'target': test_pred})
submission.to_csv('submission.csv', index=False)
```

---

## 15. 회귀 vs 분류 비교표

| 항목 | 회귀 | 분류 |
|------|------|------|
| 타겟 | 연속값 | 이산 클래스 |
| 분할 | train_test_split | train_test_split + **stratify** |
| CV | KFold | **StratifiedKFold** |
| 불균형 처리 | 불필요 | **SMOTE, class_weight** |
| 평가 지표 | RMSE, MAE, R² | F1, AUC, Confusion Matrix |
| 출력 | predict() | predict() + **predict_proba()** |
| 임계값 조정 | 불가 | **가능 (proba 기반)** |
| 손실 함수 | MSE | CrossEntropy, LogLoss |

---

## 16. 체크리스트 (시험 전)

```
데이터 이해:
☐ info(), describe()
☐ 타겟 클래스 분포 (불균형 여부!)
☐ 결측치 비율
☐ 수치형/범주형 구분

전처리:
☐ 결측치 처리 (median / 'Missing')
☐ 이상치 확인 (Capping 또는 플래그)
☐ 범주형 인코딩 (Label/OneHot)
☐ 스케일링 (선형/KNN/SVM 사용 시)

불균형 처리:
☐ 비율 확인 후 전략 결정
☐ class_weight='balanced' 시도
☐ SMOTE는 **train에만** 적용 (Pipeline 내에서)
☐ stratify=y로 분할

모델링:
☐ StratifiedKFold 사용
☐ 베이스라인 (LogisticRegression)
☐ 강력한 모델 (RandomForest, XGBoost)
☐ 튜닝 (F1 또는 AUC 기준)

평가:
☐ classification_report 출력
☐ Confusion Matrix 시각화
☐ ROC/PR 곡선 (불균형 시 PR 중시)
☐ 임계값 조정 고려

제출:
☐ test 데이터 동일 전처리
☐ predict() 또는 predict_proba() 중 선택
☐ 파일 형식 맞추기
```

---

## 17. 자주 하는 실수

1. **전체 데이터에 SMOTE**: Test까지 오염됨 → 반드시 train에만
2. **stratify 누락**: Train/Test 클래스 비율 불균형
3. **Accuracy만 보기**: 불균형 시 99% 나와도 소수 클래스 못 맞춤
4. **predict_proba 미사용**: AUC, 임계값 조정 불가
5. **LogisticRegression max_iter 기본값**: 수렴 안 할 수 있음 (`max_iter=1000`)
6. **XGBoost eval_metric**: 분류는 `'logloss'` 또는 `'auc'`
7. **범주형 수치 인코딩 후 스케일링**: 의미 깨짐 → OneHot 후 스케일링
8. **CV의 각 fold에서 SMOTE 미적용**: `imblearn.Pipeline` 사용
9. **Threshold 0.5 고정**: 불균형 시 조정 필수
10. **다중 클래스에서 `average` 누락**: `average='macro'` 또는 `'weighted'` 명시

---

## 18. 관련 블로그 포스트

- [데이터 분석 회귀 문제 가이드](data-analysis-regression-guide.md)
