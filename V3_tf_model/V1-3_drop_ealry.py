from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout #
from keras.api.callbacks import EarlyStopping # 
import numpy as np

# 1. 데이터 셋 생성
X = np.array([10, 20, 30, 40, 50])
y = np.array([1, 2, 3, 4, 5])

# 2. 모델 정의
model = Sequential([
    Dense(64, activation='relu', input_dim=1),
    Dropout(0.5), # => 과적합을 방지하기 위해
    Dense(32),
    Dense(1)
])

# 3. 모델 컴파일
model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# 4. 얼리스탑
early_stopping = EarlyStopping(
    monitor='loss',
    patience=5,
    verbose=2
)

# 5. 모델 훈련
model.fit(
    X,
    y,
    epochs=100000000000000,
    verbose=2,
    callbacks=[early_stopping]
)

# 6. 모델 예측
predictions = model.predict(X)
print("훈련 데이터에 대한 예측 값:")
print(predictions)