import albumentations as A  # Albumentations 라이브러리 임포트
import cv2  # OpenCV 라이브러리 임포트 (이미지 처리에 사용)
import matplotlib.pyplot as plt  # matplotlib을 사용해 이미지를 표시할 수 있도록 임포트

# 이미지 로드
img = cv2.imread('V3_데이터전처리/input_image.jpg')  # OpenCV로 이미지를 읽어옵니다.
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV는 이미지를 BGR 형식으로 읽어오므로, RGB 형식으로 변환합니다.
# RGB로 변환하는 이유는 matplotlib이 기본적으로 RGB 형식을 사용하기 때문입니다.

# Albumentations 설정
transform = A.Compose([
    A.Rotate(limit=40, p=1.0),  # 회전 변환: 이미지를 -40도에서 40도 사이로 랜덤하게 회전시킵니다.
    # `limit=40`은 회전 범위를 -40도에서 40도까지로 지정하며, `p=1.0`은 이 변환이 항상 적용되도록 설정합니다.
    
    A.RandomBrightnessContrast(p=1.0),  # 밝기와 대비를 랜덤하게 조정합니다.
    # `p=1.0`은 변환이 항상 적용되도록 설정하며, 밝기와 대비는 이미지의 픽셀 값을 조정하여 변화를 줍니다.
])

# 이미지 변환
augmented = transform(image=img)  # 정의한 변환(transform)을 이미지에 적용합니다.
img_augmented = augmented['image']  # 변환된 이미지는 'image' 키로 접근할 수 있습니다.

# 결과 확인
plt.imshow(img_augmented)  # 변환된 이미지를 matplotlib로 표시합니다.
plt.axis('off')  # 축을 숨깁니다.
plt.title('Augmented Image')  # 이미지의 제목을 설정합니다.
plt.show()  # 이미지를 화면에 표시합니다.
