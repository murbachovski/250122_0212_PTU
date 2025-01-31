# moviepy 라이브러리 임포트
# moviepy는 비디오 편집을 위한 파이썬 라이브러리로, 비디오 파일을 처리하고 다양한 변환을 할 수 있습니다.
from moviepy import VideoFileClip, CompositeVideoClip

# 비디오 파일 로드 및 특정 구간 잘라내기
clip = (
    VideoFileClip("input_video.mp4")  # 'input_video.mp4' 파일을 비디오 클립으로 로드합니다.
    .subclipped(10, 20)  # 비디오의 10초에서 20초 구간만 추출합니다.
    # .subclipped(start_time, end_time) 함수는 비디오의 특정 구간을 잘라내는 데 사용됩니다.
    # 여기서는 10초부터 20초까지의 구간을 잘라낸 클립을 생성합니다.
)

# 비디오 편집
final_video = CompositeVideoClip([clip])  # 여러 클립을 합성하여 하나의 비디오로 만듭니다.
# CompositeVideoClip은 여러 클립을 동시에 화면에 배치하거나 합성할 때 사용됩니다.
# 이 코드에서는 단일 클립만 사용하므로, 결과는 잘라낸 부분만 포함된 비디오입니다.

# 결과 비디오 파일로 저장
final_video.write_videofile("result.mp4")  # 잘라낸 클립을 'result.mp4' 파일로 저장합니다.
# 이 함수는 최종 편집된 비디오를 지정된 파일 이름으로 출력합니다.
