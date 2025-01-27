def meters_to_feet(meters):
    feet = meters * 3.28084 # 1미터는 약 3.28피트
    return feet

# 사용자 입력 받기
user_input = input("미터 값을 입력하세요 :")
try:
    meters = float(user_input)
    feet = meters_to_feet(meters)
    print(f"{meters}m는 {feet:.2f}ft입니다.")
except ValueError:
    print("숫자를 입력해주세요.")