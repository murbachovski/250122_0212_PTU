from ultralytics import solutions
import cv2

# 비디오 경로
cap = cv2.VideoCapture(0)

# 특정 좌표 설정
region_points = {
    "region-01" : [(200, 100), (600, 100), (100, 550)],
    "region-02" : [(10, 100), (60, 10), (10, 50)],
}

# 구역 설정
region = solutions.RegionCounter(
    show=True,
    region=region_points,
    model='yolo11n.pt'
)

# 비디오 처리
while cap.isOpened():
    ret, im0 = cap.read()
    if not ret:
        print("프레임 확인 요청")
        break
    
    # 특정 구역 계산
    im0 = region.count(im0)
    
# 비디오 해제
cap.release()
cv2.destroyAllWindows()