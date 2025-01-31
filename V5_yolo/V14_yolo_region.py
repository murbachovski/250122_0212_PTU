from ultralytics import solutions
import cv2
import os

# 이미지 파일 열기
im0 = cv2.imread("input1.jpeg")
im0 = cv2.resize(im0, (640, 480))

# 좌표 설정
region_points = {
    "region-01" : [(276, 252), (59, 467), (596, 468), (397, 250)]
}
# CLICKED : 276, 252
# CLICKED : 59, 467
# CLICKED : 596, 468
# CLICKED : 397, 250

# 구역 설정
region = solutions.RegionCounter(
    show=True,
    region=region_points,
    model="./yolo11n.pt"
)

# 구역에서 객체 수 계산
im0 = region.count(im0)

# 결과 이미지 저장
file_name = os.path.join("./result.jpg")
cv2.imwrite(file_name, im0)