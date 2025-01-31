import cv2
import matplotlib.pyplot as plt

# 이미지 로드
img = cv2.imread('V3_데이터전처리/input_image.jpg')  # 이미지 파일을 불러옵니다.

# BGR에서 RGB로 변환 (OpenCV는 기본적으로 BGR 형식을 사용)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV는 기본적으로 BGR을 사용하므로, RGB 형식으로 변환합니다.

# 1. 이미지 회전
height, width = img.shape[:2]  # 이미지의 높이와 너비를 구합니다.
center = (width // 2, height // 2)  # 회전의 중심은 이미지의 중앙으로 설정합니다.
rotation_matrix = cv2.getRotationMatrix2D(center, 180, 1)  # 180도 회전하는 행렬을 생성합니다.
img_rotated = cv2.warpAffine(img, rotation_matrix, (width, height))  # 이미지를 회전시킵니다.

# 2. 이미지 밝기 조정
img_brightened = cv2.convertScaleAbs(img, alpha=1.5, beta=0)  # alpha=1.5는 밝기를 1.5배로 증가시킵니다. beta는 밝기 추가값을 설정하는데, 0으로 설정해 밝기만 증가합니다.

# 3. 이미지 크기 조정
img_resized = cv2.resize(img, (width // 2, height // 2))  # 이미지의 크기를 절반으로 줄입니다. (너비와 높이를 절반으로 설정)

# 4. 이미지 반전 (수평 반전)
img_flipped = cv2.flip(img, 1)  # flip 함수에서 1은 수평 반전을 의미합니다. 0은 수직 반전, -1은 수평+수직 반전입니다.

# 5. 이미지 대비 조정
img_contrast = cv2.convertScaleAbs(img, alpha=2.0, beta=0)  # alpha=2.0은 대비를 두 배로 증가시킵니다. beta는 밝기 추가값을 설정하는데, 0으로 설정해 대비만 증가합니다.

# 6. 이미지 색상 조정 (HSV 변환 후 색상 조정)
img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # RGB 이미지를 HSV 색상 공간으로 변환합니다.
h, s, v = cv2.split(img_hsv)  # HSV 채널을 분리합니다. h: 색상, s: 채도, v: 명도
s = cv2.add(s, 50)  # 채도(s)를 50만큼 증가시킵니다.
s = cv2.min(s, 255)  # 채도가 255를 넘지 않도록 제한합니다.
img_hsv_new = cv2.merge([h, s, v])  # 수정된 채도 값을 포함한 새로운 HSV 이미지를 만듭니다.
img_color_adjusted = cv2.cvtColor(img_hsv_new, cv2.COLOR_HSV2RGB)  # 수정된 HSV 이미지를 다시 RGB로 변환합니다.

# 결과 확인
fig, ax = plt.subplots(2, 3, figsize=(15, 10))  # 2x3 격자 형태로 서브 플롯을 생성합니다.

ax[0, 0].imshow(img_rotated)  # 회전된 이미지를 첫 번째 서브 플롯에 표시
ax[0, 0].axis('off')  # 축을 숨깁니다.
ax[0, 0].set_title('Rotated Image')  # 제목을 설정합니다.

ax[0, 1].imshow(img_brightened)  # 밝기가 조정된 이미지를 두 번째 서브 플롯에 표시
ax[0, 1].axis('off')
ax[0, 1].set_title('Brightened Image')

ax[0, 2].imshow(img_resized)  # 크기가 조정된 이미지를 세 번째 서브 플롯에 표시
ax[0, 2].axis('off')
ax[0, 2].set_title('Resized Image')

ax[1, 0].imshow(img_flipped)  # 반전된 이미지를 네 번째 서브 플롯에 표시
ax[1, 0].axis('off')
ax[1, 0].set_title('Flipped Image')

ax[1, 1].imshow(img_contrast)  # 대비가 조정된 이미지를 다섯 번째 서브 플롯에 표시
ax[1, 1].axis('off')
ax[1, 1].set_title('Contrast Adjusted')

ax[1, 2].imshow(img_color_adjusted)  # 색상이 조정된 이미지를 여섯 번째 서브 플롯에 표시
ax[1, 2].axis('off')
ax[1, 2].set_title('Color Adjusted')

plt.show()  # 모든 이미지를 화면에 표시합니다.
