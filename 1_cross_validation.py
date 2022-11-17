# -*- coding: utf-8 -*-

# 1_cross_validation.py

# 머신러닝을 사용하여 데이터를 분석하는 과정
# 1. 데이터 셋 로딩
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)

# 2. 데이터의 전처리
# - 라벨 데이터 인코딩!
# - 원핫인코딩
# - 특성 확장
# ...

# 3. 데이터 셋의 분할
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=1)

# 2. 데이터의 전처리
# - 스케일 처리
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 머신러닝 모델의 학습
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(
            C=1.0,
            n_jobs=-1,
            random_state=1).fit(X_train_scaled,y_train)

# 5. 머신러닝 모델의 평가
# - 모델의 평가 기준(테스트 데이터 셋이 사용)
score = model.score(X_train_scaled, y_train)
print(f'(MODEL) TRAIN SCORE : {score}')

score = model.score(X_test_scaled, y_test)
print(f'(MODEL) TEST SCORE : {score}')










