# pip install ultralytics
from ultralytics import YOLO

# 1. 모델 로드
model = YOLO("yolo11n.pt")

# 2. 모델 예측
results = model("./V5_yolo/input.jpg")

# 3. 예측 결과 출력
# print(results)

# print(dir(results[0]))
# print(results[0].boxes.xywh)

# 4. 예측 결과 출력2
# num = 0
# for box in results[0].boxes.xywh:
#     num += 1
#     print(f"Bounding box_{num}: {box}")

# print("FINISH")

# 5. 모델의 클래스 이름 가져오기
# class_names = model.names
# # print(class_names)

# for one in results[0].boxes.cls:
#     class_name = class_names[int(one)]
#     print(f"Detected object class : {class_name}")
# print("FINISH")

# 6. confidence score 가져오기
# print(dir(results[0].boxes))
print(results[0].boxes.conf)