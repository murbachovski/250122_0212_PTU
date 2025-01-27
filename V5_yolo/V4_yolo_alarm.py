import cv2
from ultralytics import solutions

# 인풋 데이터 로드
cap = cv2.VideoCapture("./V6_yolo/input.mp4")

# 데이터 속성 정보 가져오기
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
# print(w, h, fps)
# 1920 1080 25

# 이메일 인증 정보 설정(발신자, 비밀번호, 수신자)
from_email = "ai.murbachovski@gmail.com"
password = "hqsu venx bbba zokx"
to_email = "ai.murbachovski@gmail.com"

# 보안 알림 시스템 객체 생성
security = solutions.SecurityAlarm(
    model='yolo11n.pt',
    record=1 # 감지된 객체 수가 1 이상일 때 이메일 전송!!
)

# 이메일 인증
security.authenticate(from_email, password, to_email)

# 데이터 처리 시작
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("데이터 문제있어요.")
        break
    result_data = security.monitor(im0)
    
# 데이터 해제
cap.release()
cv2.destroyAllWindows()

# pip install shapely
# pip install lap