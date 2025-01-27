from keras.api import layers, models
from keras.api.datasets import mnist
from keras.api.utils import to_categorical

# 1. 데이터셋 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 데이터 차원 변경
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# 3. 원핫인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 4. 모델 정의
model = models.Sequential([
    layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)),
    layers.Conv2D(64, (3, 3)),
    layers.Flatten(),
    layers.Dense(32),
    layers.Dense(10, activation='softmax')
])

# 5. 모델 컴파일
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 6. 모델 훈련
model.fit(x_train, y_train, epochs=2)

# 7. 모델 저장
model.save('./V4_tf_model/mnist.h5')
