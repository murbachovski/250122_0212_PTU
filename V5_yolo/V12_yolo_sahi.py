# from sahi.utils.ultralytics import download_yolo11n_model
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

# 모델 경로 지정
model_path = "./yolo11n.pt"

# 모델 예측 준비
detect_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model_path,
    confidence_threshold=0.4
)

# SAHI 예측
results = get_sliced_prediction(
    "demo_data/small-vehicles1.jpeg",
    detect_model,
    slice_height=256, # => (이미지를 256x256 크기의 조각으로 분할)
    slice_width=256,
    overlap_height_ratio=0.2, # => 슬라이스 간 높이 오버랩 비율(20%)
    overlap_width_ratio=0.2,
    verbose=2
)

# 예측 결과 시각화
results.export_visuals(export_dir="demo_data/sahi")