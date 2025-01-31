from ultralytics import solutions
import cv2

# 비디오 읽기
# cap = cv2.VideoCapture("./V6_yolo/input2.mp4")
cap = cv2.VideoCapture(0)

# 모델 로드
distance = solutions.DistanceCalculation(model='yolo11n.pt', show=True)

# 비디오 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("프레임 확인")
        break
    
    # 거리 계산 함수 호출
    frame, pixels_distancs = distance.calculate(frame)
    print(pixels_distancs)
    
# 비디오 객체 해제
cap.release()
cv2.destroyAllWindows()
    