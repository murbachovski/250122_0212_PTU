from ultralytics import solutions
import cv2

cap = cv2.VideoCapture("rtsp://210.99.70.120:1935/live/cctv027.stream")

region_points = {
    "region-01": [(102, 117), (56, 357), (5, 400), (18, 473), (167, 439), (209, 354), (142, 111)]
}

region = solutions.RegionCounter(
    region=region_points,
    model='./yolo11n.pt',
    show=True
)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("프레임 확인바람")
        break
    
    frame = region.count(frame)

cap.release()
cv2.destroyAllWindows()