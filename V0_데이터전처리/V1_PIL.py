from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

# 이미지 로드
img = Image.open('V3_데이터전처리/input_image.jpg')

# 이미지 회전
img_rotated = img.rotate(180)

# 이미지 밝기 조정
enhancer = ImageEnhance.Brightness(img)
img_brightened = enhancer.enhance(1.5)  # 1.5배 밝기 증가

# 결과 확인
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img_rotated)
ax[0].axis('off')
ax[0].set_title('Rotated Image')

ax[1].imshow(img_brightened)
ax[1].axis('off')
ax[1].set_title('Brightened Image')

img_rotated.save('./V3_데이터전처리/rotated.jpg')

plt.show()
