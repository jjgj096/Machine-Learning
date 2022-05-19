import pandas as pd
pd.options.display.max_columns=100
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X.head()
y.head()
y.tail()
y.value_counts()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.3,
                                                 stratify=y,
                                                 random_state=1)

# 앙상블 클래스 로딩
#- 부스팅 계열 클래스
#- 앙상블을 구성하는 내부 각 모델들이 선형으로 연결되어 학습 및 예측을 수행.
# 부스팅 계열의 베이스 모델은 강한 제약을 설정해야함 ****중요
# -> 이거만 보고 어떻게 배우라고!! 라는 소리가 나올 정도로 설정해야함.
 
# 1. AdaBoost: 데이터 중심의 부스팅 방법론 구현
#           -> 데이터 중심: 직전 모델이 잘못 예측한 데이터에 가중치를 부여하는 방법
#       -> 제약이 심한 상태에서 맞췄다는건 맞추기 쉽다는 것이고,
#       -> 못 맞춘 것을 찾아서 해야하므로 직전 모델이 틀린거에 집중해야함.


# 2. GradientBoosting : 오차에 중심을 둔 부스팅 방법론 구현 - 얼마나 틀렸나?
# (데이터 중심은 틀린 여부에 집중을 두는 반면) 어느 만큼 틀렸는지에 중심을 둠.

# 각 학습 데이터에 대해 오차 범위가 큰 데이터에 가중치를 부여하여,
#    오차를 줄여나가는 방식

# 부스팅 계열의 데이터 예측 예시
# 1번째 모델의 예측값 * 가중치(1번 모델의 가중치) +
# 2번째 모델의 예측값 * 가중치(2번 모델의 가중치) +
# ...
# n번째 모델의 예측값 * 가중치(N번 모델의 가중치)

# AdaBoost 클래스
from sklearn.ensemble import AdaBoostClassifier

# 앙상블을 구현하기 위한 기본 클래스 로딩
from sklearn.linear_model import LogisticRegression

# C의 값을 매우 작게 설정 -> 제약을 매우 크게 설정
base_estimator = LogisticRegression(C=0.001,
                                    class_weight='balanced',
                                    n_jobs= -1,
                                    random_state=1)

model = AdaBoostClassifier(base_estimator = base_estimator,
                           n_estimators=150,
                           learning_rate=1.0,
                           random_state=1)

model.fit(X_train, y_train)

score = model.score(X_train, y_train)   # max samples와 features 1.0이면 100%
print(f'Score(Train) : {score}')

score = model.score(X_test, y_test)
print(f'Score(Test) : {score}')
