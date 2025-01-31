# pip install sahi
from sahi.utils.file import download_from_url
from sahi.utils.ultralytics import download_yolo11n_model
from sahi.predict import get_prediction
from sahi import AutoDetectionModel

# 모델 다운로드
model_path = './yolo11n.pt'
download_yolo11n_model(model_path)

# 테스트 이미지 다운로드
download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg",
    "demo_data/small-vehicles1.jpeg"
)

# 모델 불러와서 예측
detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=model_path
)

# 기본 예측
results = get_prediction("./demo_data/small-vehicles1.jpeg", detection_model)

# 예측 결과 저장
results.export_visuals(export_dir="./demo_data/default")