from ultralytics import YOLO

# 1. 모델 로드
model = YOLO("yolo11n.pt")

# 2. 모델 훈련
model.train(
    data = 'V5_yolo/coco8.yaml',
    epochs=10,
    imgsz=320,
    device='cpu'
)