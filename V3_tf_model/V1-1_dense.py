# V4_tf_model 폴더 생성
# V1-1_dense.py 파일 생성
# pip install tensorflow==2.17.0 설치!!! => 가상환경에서!!!(py39)
import tensorflow as tf
print(tf.__version__)
import keras
print(keras.__version__)
# 2.17.0
# 3.8.0
from keras.api.models import Sequential
from keras.api.layers import Dense
import numpy as np # => 수치 계산을 위한 

# 1. 데이터셋 생성
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 2. 모델 정의
model = Sequential([
    # 첫 번째 Dense 레이어 64개의 노드, 활성화 함수는 ReLU 사용
    Dense(64, activation='relu', input_dim=1),
    # 두 번째 Dense 레이어(히든 레이어) 32개의 노드
    # Dense(32, activation='relu'),
    Dense(128),
    Dense(64),
    Dense(32),
    # 출력 레이어 노드 1개(회귀 문제는 활성화 함수 없음)
    Dense(1)
])

# 3. 모델 컴파일
model.compile(
    optimizer='adam', # => 대체적으로 효율적
    loss='mean_squared_error' # => 회귀 문제에 적합
)

# 4. 모델 훈련
model.fit(
    X, # 입력 데이터
    y, # 정답 레이블
    epochs=10, # 훈련 반복 횟수
    verbose=2 # 훈련 과정 출력(1: 간략, 2: 상세)
)

# 5. 모델 예측
predictions = model.predict(X)
print("훈련 데이터에 대한 예측 값 : ")
print(predictions)

# 6. 원하는 숫자에 대한 예측 수행
input_data = np.array([5])
pred = model.predict(input_data)
print(f"{input_data}에 대한 예측 값 : {pred}")