from ultralytics import YOLO
import cv2

# 모델 로드
model = YOLO("yolo11n-seg.pt")

# 모델 예측
results = model(
    "./V6_yolo/input_2.jpg",
)

# 결과 저장
image = results[0].plot()
cv2.imwrite("./result.jpg", image)