from ultralytics import YOLO

# 1. 모델 로드
model = YOLO('yolo11n.pt')

# 2. 모델 내보내기
model.export(format='onnx')

# pip install onnxruntime
# pip install onnxslim
# pip install onnx

# 구글에 yolo tutorial colab 검색 후 export 확인!