# 1 킬로그램(kg) = 2.20462 파운드(lb)
# kg to lb변환 함수를 만들어주세요~
# (while, try, except를 활용해서)
def kg_to_pounds(kg):
    pounds = kg * 2.20462
    return pounds

while True:
    user_input = input("킬로그램 값을 입력하세요. : ")
    try:
        kg = float(user_input)
        pounds = kg_to_pounds(kg)
        print(f"{kg}kg은 {pounds:.2f}lb입니다.")
        break
    except ValueError:
        print("잘못된 입력입니다. 숫자를 입력해주세요.")
        