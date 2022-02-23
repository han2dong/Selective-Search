import selectivesearch
import cv2
import matplotlib.pyplot as plt
import random

 
default_dir = '/content/DLCV'
img = cv2.imread('parrot.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print('img shape:', img.shape)

plt.figure(figsize=(8, 8))
plt.subplot(1,2,1)
plt.imshow(img_rgb)
#plt.show()

_, regions = selectivesearch.selective_search(img_rgb, scale=1000, min_size=5000)

#img_rgb: 이미지의 rgb값
#scale: 알고리즘이 선택하는 오브젝트 크기를 조정하는 값->알고리즘 조정
#min_size: 추천되는 값 중에 최소 이 값 이상은 선택하겠다(가로x세로값)->선택값 조정

print(type(regions), len(regions)) #박스 개수 

#rect는 x값, y값, width값, height값을 가지며 이 값이 Detected Object 후보를 나타내는 Bounding box
#size는 Bounding box의 크기
#labels는 해당 rect로 지정된 Bounding Box내에 있는 오브젝트들의 고유 ID
print(regions)

#사각형 정보만 출력하기
rect_size = [i['rect']for i in regions]
#print(rect_size)

img_rgb_copy = img_rgb.copy()
for rect in rect_size: 
    green_rgb = (random.randint(0,255),random.randint(0,255),random.randint(0,255)) #박스 색깔   
    left = rect[0]
    top = rect[1]
    right = left + rect[2] #오른쪽=왼쪽+너비
    bottom = top + rect[3] #바닥=탑+높이
    
    img_rgb_copy = cv2.rectangle(img_rgb_copy, (left, top), (right, bottom), color=green_rgb, thickness=2)

plt.subplot(1,2,2)
plt.imshow(img_rgb_copy)
plt.show()