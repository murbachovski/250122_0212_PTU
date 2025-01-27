import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.api.datasets import mnist
from keras import layers, models
from keras.api.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# 1. 데이터셋 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 데이터셋 크기 출력
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# (60000, 28, 28) (60000,)
# (10000, 28, 28) (10000,)

# 3. CNN모델 훈련을 위한 데이터 차원 변경
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# 4. 원핫 인코딩
# 레이블 예 7 => [0, 0, 0, 0, 0, 1, 0, 0, 0]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 5. 모델 정의
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.Conv2D(64, (3, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(64),
    layers.Dense(10, activation='softmax')
])

# 6. 모델 컴파일
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 7. 모델 구조 출력
model.summary()

# 8. 모델 학습
model.fit(x_train, y_train, epochs=1)

# 9. 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy : {test_acc}")
# Test accuracy : 0.9697999954223633

predictions = model.predict(x_test)

# 10. 예측 데이터 시각화
num_images_to_display = 2
random_sample_indices = np.random.choice(len(x_test), num_images_to_display)

plt.figure(figsize=(12, 4))

# 첫 번째 이미지
plt.subplot(1, num_images_to_display, 1)
plt.imshow(x_test[random_sample_indices[0]].reshape(28, 28), cmap='gray')
plt.axis('off')
true_label_1 = np.argmax(y_test[random_sample_indices[0]])
predicted_label_1 = np.argmax(predictions[random_sample_indices[0]])
plt.title(f"True: {true_label_1}\nPred: {predicted_label_1}")

# 두 번째 이미지
plt.subplot(1, num_images_to_display, 2)
plt.imshow(x_test[random_sample_indices[1]].reshape(28, 28), cmap='gray')
plt.axis('off')
true_label_2 = np.argmax(y_test[random_sample_indices[1]])
predicted_label_2 = np.argmax(predictions[random_sample_indices[1]])
plt.title(f"True: {true_label_2}\nPred: {predicted_label_2}")

plt.tight_layout()
plt.show()