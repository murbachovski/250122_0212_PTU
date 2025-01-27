from ultralytics import YOLO

# 1. YOLO 모델 로드
model = YOLO("yolo11n.pt")

# 2. 모델 예측
results = model("V5_yolo/input.jpg", save=True)