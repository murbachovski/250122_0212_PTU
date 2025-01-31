from samgeo.text_sam import LangSAM
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
import numpy as np

str_time = time.time()

sam = LangSAM(model_type='vit_b')
print(sam)

# 이미지 불러오기
image = Image.open("./V8_text_sam/segment-geospatial-main/input_image.jpg")

# text prompt 설정
text_prompt = '사람'

# 예측
result = sam.predict(image, text_prompt, box_threshold=0.24, text_threshold=0.24)

# output 경로 설정
output_folder = './'

# 결과 이미지 설정
output_file_path = os.path.join(output_folder, 'result_image.jpg')

# 이미지에 탐지 표시
sam.show_anns(box_color='red')
plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
plt.clf()

print(f"여기에 결과 이미지 저장됐습니다 => {output_file_path}")

end_time = time.time()
all_time = end_time - str_time
print(np.round(all_time))
# One-shot learning => Few-shot learning => Zero-shot Learning