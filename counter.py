import cv2
from matplotlib import pyplot as plt 
import numpy as np

img_name = "EB-02-660_0595_0482.JPG"

img = cv2.imread(f"Images for first miniproject/{img_name}")
assert img is not None, "Failed to load image."


# create cropped image
H,W,_ = img.shape
img2 = img[H-H//8:,W-W//8:]
#cv2.imwrite('output/sliced.JPG', img2)

pumkin_mask = cv2.imread(f"output/sliced_mask2.JPG")
assert pumkin_mask is not None, "Failed to load image."
pumkin_mask = pumkin_mask.reshape((-1))

img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

pixels = img2.reshape((-1))

annotated_pixels = pixels[pumkin_mask == 255].reshape((-1,3))

avg = np.average(annotated_pixels, axis=0)
cov = np.cov(annotated_pixels.transpose())

print(avg,cov)
cov = cov / 10
# Calculate the euclidean distance to the reference_color annotated color.
pixels = img2.reshape((-1,3))
shape = pixels.shape
diff = pixels - np.repeat([avg], shape[0], axis=0)
inv_cov = np.linalg.inv(cov)
moddotproduct = diff * (diff @ inv_cov)
mahalanobis_dist = np.sum(moddotproduct, 
    axis=1)

print(f"{mahalanobis_dist.shape=}")

mahalanobis_distance_image = np.reshape(
    mahalanobis_dist, 
    (img2.shape[0],
        img2.shape[1]))



print(mahalanobis_distance_image.shape)

# Scale the distance image and export it.
mahalanobis_distance_image_remapped = 255 * mahalanobis_distance_image / np.max(mahalanobis_distance_image)
cv2.imwrite("output/mahalanobis.JPG",
        255 - mahalanobis_distance_image_remapped)

annotated_mahalanobis_threshold = img2.copy()
annotated_mahalanobis_threshold[mahalanobis_distance_image_remapped > 30] = 0

print(f"{annotated_mahalanobis_threshold.shape=}")

annotated_mahalanobis_threshold = np.reshape(
    annotated_mahalanobis_threshold, 
    img2.shape)

# annotated_mahalanobis_threshold = cv2.cvtColor(annotated_mahalanobis_threshold, cv2.COLOR_HSV2BGR)
# cv2.imwrite("output/mahalanobis_threshold.JPG",
#         annotated_mahalanobis_threshold)

# define
min_pixels = 15 # TODO base it on the GSD and physical pumpkin size

# annotated_mahalanobis_threshold = cv2.cvtColor(annotated_mahalanobis_threshold, cv2.COLOR_HSV2BGR)
# res1,res2 = cv2.findContours(annotated_mahalanobis_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# print(res1,res2)