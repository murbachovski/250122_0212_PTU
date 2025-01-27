from ultralytics import YOLO
import cv2

# 1. 모델 로드
model = YOLO('yolo11x.pt')

# 2. 모델 예측
results = model(
    "./V6_yolo/input_2.jpg",
    # max_det 이미지당 허용되는 최대 감지 횟수. 모델이 한 번의 추론에서 감지할 수 있는 총 오브젝트 수를 제한하여 밀집된 장면에서 과도한 출력을 방지합니다.
)

# 3. 이미지 저장
image = results[0].plot()

# 4. 상태 정의
number = len(results[0])
print(number)
if number <=5:
    status = "Normal"
elif 6 <= number <=10:
    status = "Warning"
else:
    status = "Danger"
    
# 5. 상태 출력
print(f"Detected Number : {number}, status : {status}")
cv2.putText(image, f"Detected Number : {number}, Status : {status}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 255) ,2)
cv2.imwrite("./result_image_number.jpg", image)