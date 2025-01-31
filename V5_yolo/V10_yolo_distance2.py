from ultralytics import solutions
import cv2

DISTANCE = 10

# 비디오 읽기
cap = cv2.VideoCapture("./V6_yolo/input2.mp4")
# cap = cv2.VideoCapture(0)

# 모델 로드
distance = solutions.DistanceCalculation(model='yolo11n.pt', show=True)

# 비디오 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("비디오 종료 또는 프레임 확인 불가")
        break
    
    # 거리 계산 함수 호출
    frame, b = distance.calculate(frame)
    status = "=>"
    color = (0, 0, 0)
    print(b)
    
    # 위험 감지 상태 설정
    if b is not None and b >= 200:
        status += "Safe"
        color = (0, 0, 0)
    elif b is not None and b >= 100:
        status += "Warning"
        color = (0, 255, 0)
    elif b is None:
        status += "None"
        color = (255, 0, 0)
    else:
        status += "Danger"
        color = (0, 0, 255)
    
    # 상태를 프레임에 출력
    cv2.putText(frame, status, (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # 비디오 출력 창에 표시
    # cv2.imshow("Distance Detection", frame)
    # solutions.display_output(frame)
    solutions.DistanceCalculation.display_output(frame)
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 객체 해제
cap.release()
cv2.destroyAllWindows()