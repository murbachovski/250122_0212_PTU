from ultralytics import YOLO
import cv2

# 모델 로드
model = YOLO("yolo11n-pose.pt")

# 웹캠 가져오기
cap =cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라 확인해주세요.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 확인 바람")
        break
    
    # YOLO 모델로 예측
    results = model(frame)
    
    # 결과 시각화
    annotated_frame = results[0].plot()
    
    # 화면 출력
    cv2.imshow("YOLO REAL_TIME", annotated_frame)
    
    # q 키를 눌러서 끄기
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()