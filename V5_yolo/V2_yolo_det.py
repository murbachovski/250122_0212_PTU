from ultralytics import YOLO
import cv2

# 1. 모델 로드
model = YOLO('yolo11x.pt')

# 2. 모델 예측
results = model(
    "./V6_yolo/input_2.jpg",
    conf=0.9, # => 임계치
    # max_det 이미지당 허용되는 최대 감지 횟수. 모델이 한 번의 추론에서 감지할 수 있는 총 오브젝트 수를 제한하여 밀집된 장면에서 과도한 출력을 방지합니다.
)

# 3. 이미지 저장
image = results[0].plot()
cv2.imwrite("./result_image4.jpg", image)