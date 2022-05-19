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

# 앙상블 클래스 로딩 - BagginClassifier
# Bagging : 특정 머신러닝 알고리즘을 기반으로 (base),
#           데이터의 무작위 추출을 사용하여 
#           서로 다른 데이터를 학습하며 앙상블을 구현하는 방법

from sklearn.ensemble import BaggingClassifier

# BagginfClassifier 하이퍼 파라메터 분석
# base_estimator : 하나의 방법으로 고정하는 것. -> lr dt knn 여러 방법 말고
# n_estimators : base와 다른 방법은 다른 estimators이니까 다르게 분류해야함.
# Bootstrapping : 무작위 추출할 때 중복 sampling을 허락하는지 안하는지. 디폴트는 트루.
# max_features : 원본 데이터의 특성 중, 참고할 수 있는 특성 비율? 
#               -> 1.0일수록 모든 것을 참고하므로 성능이 올라감.

from sklearn.tree import DecisionTreeClassifier

# 학습의 성능을 줄이는법, max_depth, max samples, max_features 등 설정
base_estimator = DecisionTreeClassifier(max_depth=3, random_state=1)

model = BaggingClassifier(base_estimator=base_estimator,
                          n_estimators=50,
                          max_samples=0.3,
                          max_features=0.3,
                          n_jobs=-1,
                          random_state=1)


model.fit(X_train, y_train)

score = model.score(X_train, y_train)   # max samples와 features 1.0이면 100%
print(f'Score(Train) : {score}')

score = model.score(X_test, y_test)
print(f'Score(Test) : {score}')
