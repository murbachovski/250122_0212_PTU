from ultralytics import YOLO
import cv2

# 1. 비디오 경로 설정
cap = cv2.VideoCapture(0)

# 2. 카메라 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
fps = cap.get(cv2.CAP_PROP_FPS)

# 3. 모델 로드
model = YOLO("yolo11n.pt")

# 4. 비디오 프레임 처리
while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        results = model(frame)
        print(fps)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO_REALTIME", annotated_frame)
        
        # 'q' 키 눌러서 나가기
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
cap.release()
cv2.destroyAllWindows()
