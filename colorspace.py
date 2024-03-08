import cv2
import numpy as np

cielab_range = np.array([[170, 122, 152], [233, 255, 255]])

im1 = np.zeros((210-122,255-152,3),dtype=np.uint8)
im2 = np.zeros((210-122,255-152,3),dtype=np.uint8)
im3 = np.zeros((210-122,255-152,3),dtype=np.uint8)

for y in range(im1.shape[0]):
    for x in range(im1.shape[1]):
        im1[y,x] = (170,122+y,152+x)

for y in range(im1.shape[0]):
    for x in range(im1.shape[1]):
        im2[y,x] = (202,122+y,152+x)

for y in range(im1.shape[0]):
    for x in range(im1.shape[1]):
        im3[y,x] = (233,122+y,152+x)

print(im1[-1,-1])

cv2.imwrite(f"output/color_space1.JPG",
    cv2.cvtColor(im1,cv2.COLOR_Lab2BGR))
cv2.imwrite(f"output/color_space2.JPG",
    cv2.cvtColor(im2,cv2.COLOR_Lab2BGR))
cv2.imwrite(f"output/color_space3.JPG",
    cv2.cvtColor(im3,cv2.COLOR_Lab2BGR))