from ultralytics import YOLO

# 1. 모델 로드
model = YOLO('./runs/detect/tune2/weights/best.pt')

# 2. 모델 학습
model.train(
    data='V5_yolo\coco8.yaml',
    epochs=2,
    imgsz=640,
    project='V5_yolo/fine_tune',
    name='V5_yolo/new_training',
)
