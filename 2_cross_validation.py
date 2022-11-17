# -*- coding: utf-8 -*-

# 2_cross_validation.py

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

# 4. 머신러닝 모델의 생성
# - 학습 안된 모델의 생성!
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(
            C=1.0,
            n_jobs=-1,
            random_state=1)

# 5. 생성된 머신러닝 모델을 사용하여 성능을 예측
# - 학습 진행이 X
# - 교차 검증의 수행을 통해 전체 데이터 셋에 대한
#   머신러닝 모델의 성능을 예측

# 교차검증 기능을 제공하는 cross_val_score 함수
# - 사용되는 파라메터
#   cross_val_score(예측기 객체, 
#                   전체 X 데이터, 
#                   전체 y 데이터, 
#                   교차검증개수)
# - 반환되는 값
#   교차검증 개수에 정의된 크기의 예측기 객체가 생성되며
#   각 예측기의 평가 점수가 반환됨
#   (회귀 모델의 경우 R2 스코어가 반환
#   분류 모델의 경우 정확도가 반환됨)
from sklearn.model_selection import cross_val_score

# 교차검증 과정에서의 평가 방법 수정
# - scoring 매개변수
# - https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
cv_scores = cross_val_score(model,
                         X_train_scaled,
                         y_train,
                         scoring='recall',
                         cv=5,
                         n_jobs=-1)

# 6. 교차검증 결과의 확인 평가
# - 현재 데이터 셋에 대한 모델의 평가
print(f'(CV) scores : \n{cv_scores}')
print(f'(CV) scores mean : \n{cv_scores.mean()}')

# 7. 머신러닝 모델의 학습
model.fit(X_train_scaled, y_train)

# 8. 머신러닝 모델의 평가
# - 모델의 평가 기준(테스트 데이터 셋이 사용)
score = model.score(X_train_scaled, y_train)
print(f'(MODEL) TRAIN SCORE : {score}')

score = model.score(X_test_scaled, y_test)
print(f'(MODEL) TEST SCORE : {score}')










