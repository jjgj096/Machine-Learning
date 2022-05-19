
# 선형모델 -> 분류하기 

import pandas as pd
pd.options.display.max_columns = 100
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns = data.feature_names)
y = pd.Series(data.target)

X.info()
X.isnull().sum() # 결집데이터 개수 확인 
X.describe(include = 'all')
 # 30개니까 처음엔 짤림, 그래서 pd.options.display.max_columns 값 설정

y.head()
y.value_counts()    # 데이터 개수 보기
y.value_counts() / len(y)   # 데이터 개수를 비율로 보기

from sklearn.model_selection import train_test_split

splits = train_test_split(X, y, test_size=0.3,
                          random_state=10,
                          stratify=y)


X_train=splits[0]
X_test=splits[1]
y_train=splits[2]
y_test=splits[-1]

X_train.head()
X_test.head()

X_train.shape   #398,30
X_test.shape    #171,30

y_train.value_counts() / len(y_train)
y_test.value_counts() / len(y_test)

from sklearn.linear_model import LogisticRegression #CV를 붙이면 교차검증임.
model = LogisticRegression()

# LogisticRegeression의 파라매터 분석

# Logistic에서는 제약조건 alpha 대신 C를 씀. 
# 알파가 커질수록 제약이 커져, 가중치가 작아진다.
# 하지만 C는 반대, C 가 커질수록 제약이 작아진다.
# class_weight -> 디폴트는 1, 즉, 맞췄을 때 보상이 1.
#              -> 보통 balanced를 사용. 
# solver -> 대용량 데이터인 경우 saga 사용
# max_iter -> 디폴트 100
# n_jobs -> 병렬처리, 최대의 효율을 뽑을 수 있도록 설정. 사용은 -1 값.
# verbose -> 실행결과를 출력하면서 볼 수 있음. 양수 값으로 설정.

# *** C랑 alpha 분류하는거 시험에 내신다고 하심.

model = LogisticRegression(penalty='l2',
                           C=1.0,
                           class_weight='balanced',
                           solver = 'lbfgs',
                           max_iter=10000,
                           n_jobs=-1,
                           random_state=5)

model.fit(X_train, y_train)

model.score(X_train, y_train)   #0.9522 -> 분류형이니까 정확도로 리턴임.
model.score(X_test,y_test)      #0.9415 -> 이 정도면 믿고 쓸 수 잇음.

# 가중치 값 확인 -> coef_
print(f'coef_ : {model.coef_}')
# 절편 값 확인 -> intercept_
print(f'intercept_ : {model.intercept_}') #34.234
# 확률 값 반환 -> predict_proba
proba = model.predict_proba(X_train[:5])
proba   #왼쪽이 0, 오른쪽이 1을 맞출 확률 -> 거의 무조건 1로 맞춤.

pred = model.predict(X_train[:5])
pred    # 모두 1로 예측, [-10:]하면 0으로 맞추는 것도 나옴.

df = model.decision_function(X_train[:5])
df # 모두 양수로 나옴 -> 모두 1로 예측
# [-10:]으로 하면 양수 음수 나뉨. 0을 기준으로 양수는 1, 음수는 0을 의미.

y_train[:5]

# **********************************************************************
# 분류 모델의 평가 방법
# 1. 정확도 -> 전체 중 정답을 맞춘 비율 -> score를 사용
#   -> 분류하고자 하는 클래스의 비율이 동일한 경우에만 사용.
#   -> 100개 중 하나만 틀린 경우에는 정확도 평가 방법이 좋지 않음.

# 2. 정밀도 -> 집합에서 각 클래스 별 정답 비율
# 집합 -> 머신러닝 모델이 예측한 결과
# 위 집합에서 각각의 클래스 별 정답 비율

# 3. 재현율
# 집합 -> 실제 데이터 셋 : 내가 학습하고 있는 데이터 셋
# 위 집합에서 머신러닝 모델이 예측한 정답 비율

# 정밀도와 재현율의 차이점 알기. 보통은 반비례 관계임.
# 내가 가진 100개 중 실제로 100개에 가까움. -> 재현율이 높음.
# 이렇게 되면 거의 다 정답으로 생각 -> 정밀도가 낮아짐.

# 혼동행렬
from sklearn.metrics import confusion_matrix

pred = model.predict(X_train)
cm = confusion_matrix(y_train, pred)
cm
# 실제 0인 데이터 [[141,   7], -> 148개가 0인 데이터 0을 기본적으로 141개 맞춤
# 실제 1인 데이터   [5, 245]]  -> 250개가 1인 데이터 1을 기본적으로 245개 맞춤

#class_weight의 비중을 조절하면서 재현율을 조절할 수 있음.
# balanced로 바꿀 경우, 좀더 0에 가깝게 된다. [142.6] 확인
# 0의 비중치를 올리면 어떻게든 재현율을 1.0을 만들 수 있음.

y_train.value_counts() # 실제로 확인 가능

# 0데이터에 대한 정밀도 : 141 / (141+5)
# 0데이터에 대한 재현율 : 141 / (141+7)

# 정확도
from sklearn.metrics import accuracy_score
# 정밀도
from sklearn.metrics import precision_score
# 재현율
from sklearn.metrics import recall_score


pred = model.predict(X_train)

ps = precision_score(y_train, pred, pos_label=0)
ps  #0.965

rs = recall_score(y_train, pred, pos_label=0)
rs  #0.952
