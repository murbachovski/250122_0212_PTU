# YOLO 버전별 속도 및 정확도 비교
from ultralytics import YOLO
import cv2 
import time

# 1. 비교할 YOLO 모델 목록 정의
# n => s => m => l => x 순서
# nano, small, medium, large, xlarge
model_paths = [
    # "yolov5nu.pt"
    # "yolov8n.pt"
    # "yolov9t.pt",
    # "yolo11n.pt",
    # "yolo11s.pt",
    # "yolo11m.pt",
    # "yolo11l.pt",
    # "yolo11x.pt"
]

# 2. 테스트 입력 이미지 경로
image_path = "./V5_yolo/input.jpg"

# 3. 각 모델에 대해 추론 속도 및 결과 비교
for model_path in model_paths:
    # 3-1. 모델 로드
    model = YOLO(model_path)
    
    # 3-2. 추론 시작 시간
    start_time = time.time()
    
    # 3-3. 이미지에 대한 예측 수행
    results = model(image_path, save=True)
    
    # 3-4. 추론 종료 시간 기록
    end_time = time.time()
    
    # 3-5. 추론 시간 계산
    inference_time = end_time - start_time
    
    # 3-6. 추론 결과 시각화
    image = results[0].plot
    
    # 3-7. 결과 이미지 저장
    # result_image_path = f"./result_{model_path.split('.')[0]}.jpg"
    # cv2.imwrite(result_image_path, image)
    
    # 3-8. 모델 정보 출력
    print(f"모델 : {model_path}")
    print(f"추론 시간 : {inference_time}")
    # print(f"결과 이미지 저장 경로 : {result_image_path}")
    print("-" * 40)
    