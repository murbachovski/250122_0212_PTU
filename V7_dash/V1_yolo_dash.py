from ultralytics import YOLO
import cv2
import base64 # => 이미지를 인코딩하기 위한
import plotly.graph_objs as go # => 그래프 시각화
from dash import Dash, html, dcc # => 대시보드 웹을 만들기 위한
from dash.dependencies import Input, Output # => 콜백을 통해 업데이트
import time # => FPS 시간 측정
from datetime import datetime # => 현재 시간 가져오기

# 모델 로드
model = YOLO("yolo11n.pt")

# 데이터 저장할 리스트
x_data = []
y_data = []

# Dash 애플리케이션 초기화
app = Dash(__name__)

# 대시보드 레이아웃 정의
app.layout = html.Div([
    html.H1("YOLO 실시간 탐지 대시보드",
            style={"textAlign": "center"}), # => 대시보드 제목
    html.Div([
        html.Img(id="live-detection-image",
                 style={"width": "80%", "margin": "auto", "display": "block"}) # => 실시간 탐지 이미지 공간
    ]),
    # 그래프
    dcc.Graph(id="real-time-graph", style={"height": "400px"}),
    # Interval 컴포넌트
    dcc.Interval(
        id="update-interval",
        interval=1000, # => 1초 마다 업데이트
        n_intervals=0 # => 초기 실행 횟수
    )
])

# 비디오 열기
cap = cv2.VideoCapture(0)

# 콜백 함수 정의 => 화면, 그래프 업데이트를 위한
# Dash는 특정 입력이 발생하면 출력을 자동으로 업데이트해줌
@app.callback(
    [
        Output("live-detection-image", 'src'),
        Output("real-time-graph", 'figure'),
    ],
    [
        Input("update-interval", "n_intervals")
    ]
)

def update_frame(n_intervals):
    global cap
    
    # FPS 계산
    start_time = time.time()
    
    # 프레임 읽기
    success, frame = cap.read()
    if not success:
        print("프레임 확인")
        return None, go.Figure(), "FPS: N/A"
    
    # 객체 탐지 수행
    results = model(frame)
    
    # 객체 탐지 수 계산
    num = len(results[0].boxes)
    
    # 탐지 결과를 이미지에 표시
    annotated_frame = results[0].plot()
    
    # FPS 계산
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    
    # FPS 정보를 영상에 추가
    fps_text = f"FPS:{fps:.2f}"
    cv2.putText(
        annotated_frame,
        fps_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1
    )
    
    # 이미지를 Base64 형식으로 변환
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    frame_base64 = base64.b64encode(buffer).decode()
    
    # 현재 시간 가져오기
    current_time = datetime.now().strftime('%Y%m%d_%H:%M:%S')
    
    # 시간과 탐지된 객체 수를 누적 저장
    x_data.append(current_time)
    y_data.append(num)
    
    # 실시간 그래프 데이터 생성
    figure = {
        'data': [
            go.Scatter(
                x=x_data, # => 시간 데이터
                y=y_data, # => 탐지된 객체 수 데이터
                mode='lines+markers', # => 선과 마커로 표시
                marker={'color': 'red'}
            )
        ],
        'layout': go.Layout(
            title="실시간 탐지 수 변화", # 그래프 제목
            xaxis={'title': '시간'}, # x축
            yaxis={'title': '탐지 수', 'dtick': 1}, # y축
            template='plotly_white' # 그래프 스타일
        )
    }
    
    return f"data:image/jpeg;base64,{frame_base64}", figure

# 앱 실행
if __name__ == "__main__":
    app.run_server() # => Dash 서버 실행