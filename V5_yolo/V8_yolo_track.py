from ultralytics import YOLO
import cv2

# 모델 로드
model = YOLO("yolo11n.pt")

# 비디오 생성
cap = cv2.VideoCapture(0)

# 프레임 처리
while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:
        results = model.track(frame, persist=False)
        
        # 시각화
        annotated_frame = results[0].plot()
        
        # 화면 표시
        cv2.imshow("YOLO_TRACKING", annotated_frame)
        
        # q 키 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

# tracking 모델 사용하는 이유는?
