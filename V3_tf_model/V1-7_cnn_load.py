from keras.api.datasets import mnist
from keras.api.utils import to_categorical
from keras.api.models import load_model
import numpy as np

# 1. 데이터셋 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 이미지 차원 변경
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# 3. 원핫인코딩
y_test = to_categorical(y_test)

# 4. load_model
loaded_model = load_model('V4_tf_model/mnist.h5')

# 5. 모델 평가
test_loss, test_acc = loaded_model.evaluate(x_test, y_test)
print(f"평가 : {test_acc}")
# 평가 : 0.892300009727478