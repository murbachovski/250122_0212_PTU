import cv2
import os
from datetime import datetime # => pip install datetime
import schedule # => pip install schedule
import time

# 0. 이미지 수집 시간 간격 설정
interval_time = 10

# 1. 저장 디렉토리 설정
SAVE_DIR = "captured_images" # 캡처된 이미지를 저장할 폴더 이름
os.makedirs(SAVE_DIR, exist_ok=True) # 디렉토리가 없으면 생성, 있으면 무시합니다.

def capture_image():
    # 2. 카메라 불러오기
    cap = cv2.VideoCapture(0)

    # 3. 카메라 확인
    if not cap.isOpened():
        raise RuntimeError("카메라 연결 안됨")

    print("카메라 연결 됐습니다.")

    # 4. 카메라 프레임 읽기
    success, frame = cap.read()
    if success:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(SAVE_DIR, f"photo_{timestamp}.jpg")
        
        # 5. 이미지 저장
        cv2.imwrite(file_path, frame)
        print(f"사진이 저장 됐습니다. {file_path}")
    else:
        print("프레임을 못 읽습니다.")

    # 6. 카메라 해제
    cap.release()
    cv2.destroyAllWindows()

capture_image()

# 스케쥴러 실행 설정
schedule.every(interval_time).second.do(capture_image)
print(f"{interval_time}초 간격으로 사진을 찍습니다.")

while True:
    schedule.run_pending()
    time.sleep(1)