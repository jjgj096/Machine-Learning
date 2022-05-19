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
# GradientBoosting 클래스
# GradientBoosting은 기본 모델이 결정트리로 고정되어 있음. Ada와는 다른 점.
# -> 랜덤포레스트의 부스팅 버전!!!
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators=200,
                                   learning_rate=0.1,
                                   max_depth=1,
                                   subsample=0.3,
                                   max_features=0.3,
                                   random_state=1,
                                   verbose=3)

model.fit(X_train, y_train)

score = model.score(X_train, y_train)   # max samples와 features 1.0이면 100%
print(f'Score(Train) : {score}')

score = model.score(X_test, y_test)
print(f'Score(Test) : {score}')
