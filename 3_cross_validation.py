# -*- coding: utf-8 -*-

# 3_cross_validation.py

from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# 전체 데이터에 대한 사전 평가
# - 교차검증
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            n_jobs=-1,
            random_state=1)

# cross_val_score 함수는 매개변수로 전달된
# 예측기 객체의 타입이 분류인 경우에는 
# 데이터의 서플 과정을 선행합니다.
# 반면, 예측기의 타입이 회귀인 경우에는 
# 데이터의 셔플 과정을 생략
from sklearn.model_selection import cross_val_score
cv_scores=cross_val_score(
            model, X, y,
            cv=5, scoring='accuracy', n_jobs=-1)

print(f'(CV) scores : \n{cv_scores}')
print(f'(CV) scores mean : \n{cv_scores.mean()}')

model.fit(X, y)

score = model.score(X, y)
print(f'(MODEL) TRAIN SCORE : {score}')

score = model.score(X, y)
print(f'(MODEL) TEST SCORE : {score}')


# 교차 검증을 위해서 사용되는 KFold 클래스
# - 데이터의 분할
from sklearn.model_selection import KFold

# 생성자의 매개변수 n_splits 값에 지정된 크기만큼 
# 데이터를 분할할 수 있는 기능을 제공
# (기본적으로 데이터를 셔플하지 않고 순차적으로 분할)
# KFold 타입의 객체를 cross_val_score 함수의 cv 매개변수로
# 사용할 수 있음
cv = KFold(n_splits=3)

cv_scores=cross_val_score(
            model, X, y,
            cv=cv, scoring='accuracy', n_jobs=-1)

print(f'(CV) scores : \n{cv_scores}')
print(f'(CV) scores mean : \n{cv_scores.mean()}')

# KFold 클래스의 객체를 생성할 때, shuffle 매개변수를 지정하지 
# 않는 경우 데이터를 순차적으로 분할하기 때문에 아래와 같이
# 라벨이 정렬된 데이터에는 잘못된 분석 결과가 나올 수 있습니다.

# KFold 클래스의 객체를 생성할 때, shuffle 매개변수의 값을
# True 로 지정하는 경우 정답 데이터(y)의 비율을 균등하게
# 포함하는 폴드들을 생성할 수 있습니다.
cv = KFold(n_splits=3, shuffle=True, 
           random_state=11)

cv_scores=cross_val_score(
            model, X, y,
            cv=cv, scoring='accuracy', n_jobs=-1)

print(f'(CV) scores : \n{cv_scores}')
print(f'(CV) scores mean : \n{cv_scores.mean()}')

from sklearn.model_selection import StratifiedKFold















