import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# mnist 데이터셋 로드
from keras.api.datasets import mnist

# 1. mnist 데이터셋 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 데이터 형태 확인
print(x_train.shape, y_train.shape)
# (60000, 28, 28) (60000,)
# 훈련 데이터는 6만 개, 각 이미지는 28x28픽셀
print(x_test.shape, y_test.shape)
# (10000, 28, 28) (10000,)
# 테스트 데이터는 1만 개, 각 이미지는 28x28픽셀
# 테스트 데이터 레이블은 1만 개

# 3. 데이터 시각화
import matplotlib.pyplot as plt
# plt.imshow(x_train[0])
# plt.title(f"Label: {y_train[0]}")
# plt.axis('off')
# plt.show()

for i in range(10):
    plt.imshow(x_train[i])
    plt.title(f"LabeL : {y_train[i]}")
    plt.axis('off')
    plt.show()