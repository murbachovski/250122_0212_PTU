from keras.api.models import Sequential
from keras.api.layers import Dense
from sklearn.model_selection import train_test_split
# pip install scikit-learn==1.6.1
from sklearn.metrics import accuracy_score # 모델 정확도 계산
import matplotlib.pyplot as plt # 시각화
from sklearn.datasets import load_breast_cancer # 데이터셋
import numpy as np
import warnings
warnings.filterwarnings(action='ignore')

# 1. 데이터 로드 및 확인
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target # 0 : 양성, 1: 악성 => 이진 분류

# print(dataset.DESCR)
# print("특징 이름:", dataset.feature_names)

# 2. 데이터셋 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.4
)

# 3. 모델 구성
model = Sequential([
    Dense(10, input_dim=x.shape[1], activation='relu'),
    Dense(8),
    Dense(1, activation='sigmoid') # 시그모이드 함수(이진 분류용)
])

# 4. 모델 컴파일
model.compile(
    loss='binary_crossentropy', # 이진 분류 문제
    optimizer='adam',
    metrics=['accuracy']
)

# 5. 모델 학습
model.fit(
    x_train,
    y_train,
    epochs=10,
    verbose=2
)

# 6. 모델 평가 및 예측
# 테스트 데이터를 사용하여 모델을 평가하고 결과 출력
results = model.evaluate(x_test, y_test)
print("테스트 데이터 평가 결과:", results)

# 7. 모델 정확도 계산
y_predict = np.round(model.predict(x_test)) # 예측 결과를 0 또는 1로 반올림
accuracy = accuracy_score(y_test, y_predict)
print("테스트 데이터 정확도:", accuracy)

# accuracy 0.9이상 만들어보세요!!