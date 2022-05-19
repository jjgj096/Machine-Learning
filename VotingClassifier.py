
# 앙상블 (Esemble)
# 다수계의 머신러닝 알고리즘 결합하여,
# 각 모델이 예측한 결과를 취합/부스팅 방법을 통해 예측을 수행하는 방법.

# 앙상블의 구현 방식 2가지
# 1. 취합 -> 독립적으로 수행
# 앙상블을 구성하는 서로 독립적인(서로 연관성 X) 모델이 예측한 결과값을 취합하여,
# 더 많은 쪽, 즉 다수결의 방식으로 수행한다. -> 더 많은 쪽으로 수행(분류 분석)
# 각각의 모델이 예측한 결과값에 대해서 평균을 취한다. (회귀 분석의 경우)

#*** 취합 방식에서는 각 모델이 성능이 너무 좋은것보다, 적절한 수준인게 더 좋음.
#       -> 각 모델이 너무 완벽하면, 취합하는 의미, 즉 다수결 의미가 없어짐. 

# 각 모델이 독립적이므로 병렬처리 가능 -> 학습 및 예측 속도가 빠름.

# Votiong, Bagging, RandomForest



# 2. 부스팅 -> 연관되어 수행
# 앙상블을 구성하는 모델들이 서로 선형으로 연결되어, 학습 및 예측을 수행하는 방법.
# 내부의 각 모델은 다음 모델에 영향을 준다.

#*** 부스팅 방식에서는 취합과 반대로, 각 모델들에게 강한 제약을 설정하여,
#       -> 점진적인 성능향상을 도모한다.

# 내부의 각 모델이 선형으로 연결되어(앞 모델이 학습 종료 후 뒷 모델 학습 수행)
#       -> 학습 및 예측 속도가 느림.

# AdaBoosting, GradientBoosting. XGBoost, LightGBM

import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

y.head()
y.tail()
y.value_counts()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                test_size=0.3,
                                                stratify=y,
                                                random_state=1)

# 앙상블 클래스 로딩
# VotingClassifier
# ** 하이퍼 파라메터 분석 **
# estimators : 사용할 데이터를 넣는 것
# voting : hard와 soft -> 디폴트는 hard
# n_jobs : 모든 코어를 사용하여 최대한 빠르게 학습 -> 병렬처리 가능의 증거

from sklearn.ensemble import VotingClassifier

# 앙상블을 구현하기 위한 내부 모델의 클래스 로딩
from sklearn.neighbors import KNeighborsClassifier # 최근접 이웃
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

m1 = KNeighborsClassifier(n_jobs=-1) 
m2 = LogisticRegression(n_jobs=-1, random_state=1)
m3 = DecisionTreeClassifier(random_state=1)

estimators = [('knn', m1), ('lr', m2), ('dt', m3)]

model = VotingClassifier(estimators=estimators,
                        voting='hard',n_jobs=-1)

model.fit(X_train, y_train)

score = model.score(X_train, y_train)
print(f'Score(Train) : {score}')

score = model.score(X_test, y_test)
print(f'Score(Test) : {score}')


# 앙상블 내부의 구성 모델 확인
print(model.estimators_[0]) # KNeighbors
print(model.estimators_[1]) # LogisticRegression
print(model.estimators_[2]) # DecisionTreeClassifier

# 앙상블 내부의 각 모델의 예측 값 확인
pred = model.estimators_[0].predict(X_test[50:51])
print(f'Predict (knn) : {pred}')

pred = model.estimators_[1].predict(X_test[50:51])
print(f'Predict (lr) : {pred}')

pred = model.estimators_[2].predict(X_test[50:51])
print(f'Predict (dt) : {pred}')
