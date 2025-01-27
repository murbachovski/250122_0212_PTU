# 분류 모델 구현
from ultralytics import YOLO
import cv2

# 1. 모델 로드
model = YOLO("yolo11x-cls.pt")

# 2. 모델 예측
results = model(
    "./V6_yolo/input.PNG"
)

# 3. 이미지 저장
image = results[0].plot()
cv2.imwrite("./result_image2.jpg", image)