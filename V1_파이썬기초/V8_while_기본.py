def meters_to_feet(meters):
    feet = meters * 3.28084
    return feet

while True:
    user_input = input("미터 값을 입력하세요 : ")
    try:
        meters = float(user_input)
        feet = meters_to_feet(meters)
        print(f"{meters}m는 {feet:.2f}ft입니다.")
        break
    except ValueError:
        print("잘못된 입력입니다. 숫자를 입력해주세요.")