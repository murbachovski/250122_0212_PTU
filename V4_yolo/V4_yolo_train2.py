from ultralytics import YOLO

# 1. 모델 로드
model = YOLO('yolo11n.pt')

# 2. 하이퍼파라미터 탐색 공간 정의
search_space = {
    "degrees" : (0.0, 45.0), # 이미지 회전 각도 (0도에서 45도 사이)
    "mosaic" : (0.0, 1.0), # 모자이크 증강 활성화 확률 (0 ~ 100% 사이)
    "mixup" : (0.0, 1.0), # MixUp 증강 활성화 확률 (0 ~ 100% 사이)
    "fliplr" : (0.0,)
}

# 3. 튜닝 실행
model.tune(
    data = 'V5_yolo/coco8.yaml',
    epochs = 2,
    iterations = 2,
    space = search_space,
    device = 'cpu'
)