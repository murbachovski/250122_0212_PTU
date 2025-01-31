from ultralytics import solutions
import cv2

# 비디오 열기
cap = cv2.VideoCapture(0)

# Heatmap 설정
heatmap = solutions.Heatmap(
    colormap=cv2.COLORMAP_PARULA,
    model="./yolo11n.pt"
)

# 비디오 실행
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 히트맵 생성
    heat_frame = heatmap.generate_heatmap(frame)
    
    # 결과 시각화
    cv2.imshow("HEATMAP", heat_frame)
    
    # q 눌러서 종료
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

# 비디오 해제
cap.release()
cv2.destroyAllWindows()