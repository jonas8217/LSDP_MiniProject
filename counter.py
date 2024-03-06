import cv2
import numpy as np

directory = "Images for first miniproject/"

def load_image_name(file_name : str, path : str = directory):
    img = cv2.imread(path+file_name)
    assert img is not None, f"Failed to load image: {file_name}"
    return img

def load_image_num(num : str):
    return load_image_name(f"EB-02-660_0595_{num}.JPG")

def generate_mean_and_cov_from_mask(img, mask):
    pixels = img.reshape((-1))
    mask = mask.reshape((-1))
    annotated_pixels = pixels[mask == 255].reshape((-1,3))

    mean = np.average(annotated_pixels, axis=0)
    cov = np.cov(annotated_pixels.transpose())

    return mean,cov

def get_max_possible_mahalanobis_distance(mean,cov):
    HSV_max = [180,255,255]
    max_dist = 0
    test_img = np.zeros((8,3))
    for i in range(1,8):
        test_img[i] = np.multiply(HSV_max,[i%2,(i//2)%2,i//4])
    return np.max(get_mahalanobis_image(test_img,mean,cov))

def get_mahalanobis_image(img,mean,cov):
    # Calculate the euclidean distance to the reference_color annotated color.
    pixels = img.reshape((-1,3))
    shape_pixels = pixels.shape
    diff = pixels - np.repeat([mean], shape_pixels[0], axis=0)
    inv_cov = np.linalg.inv(cov)
    moddotproduct = diff * (diff @ inv_cov)
    mahalanobis_dist = np.sum(moddotproduct, 
        axis=1)
    
    mahalanobis_distance_image = np.reshape(
        mahalanobis_dist, 
        img.shape[:-1])
    
    return mahalanobis_distance_image

def generate_mahalanobis_mask(img,mean,cov,max_dist):
    mahalanobis_distance_image = get_mahalanobis_image(img,mean,cov)

    # Norm the image by the maximum possible distance for consistancy across images
    mahalanobis_distance_image_norm = mahalanobis_distance_image / max_dist
    
    annotated_mahalanobis_threshold = np.zeros(mahalanobis_distance_image.shape,dtype=img.dtype)
    annotated_mahalanobis_threshold[mahalanobis_distance_image_norm < 0.006] = 1
    
    mahalanobis_mask = np.reshape(
        annotated_mahalanobis_threshold, 
        (img.shape[0],
         img.shape[1]))
    
    return mahalanobis_mask


num = 482
num = str(num).rjust(4,'0')
num2 = 52
num2 = str(num2).rjust(4,'0')

img = load_image_num(num)
mask = load_image_name(f"{num}_sliced_mask.JPG","masks/")

# create cropped image (bottom right 1/8 by 1/8 of the image)
H,W,_ = img.shape
img_sliced = img[H-H//8:,W-W//8:]
cv2.imwrite(f'output/{num}_sliced.JPG', img_sliced)

img_sliced = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2HSV)

mean,cov = generate_mean_and_cov_from_mask(img_sliced, mask)

print(mean)
print(cov)
max_dist_mahalanobis = get_max_possible_mahalanobis_distance(mean,cov)

mahalanobis_mask = generate_mahalanobis_mask(img_sliced,mean,cov,max_dist_mahalanobis)

annotated_image = img_sliced.copy()
annotated_image[mahalanobis_mask == 0] = 0
annotated_image_inv = img_sliced.copy()
annotated_image_inv[mahalanobis_mask == 1] = 0

annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_HSV2BGR)
cv2.imwrite(f"output/{num}_annotated.JPG",
        annotated_image)
annotated_image_inv = cv2.cvtColor(annotated_image_inv, cv2.COLOR_HSV2BGR)
cv2.imwrite(f"output/{num}_annotated_inv.JPG",
        annotated_image_inv)



contours, hierarchy = cv2.findContours(mahalanobis_mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
print(hierarchy[0][0:4])
for i,h in enumerate(hierarchy[0]):
    if h[2] != -1 or h[3] != -1:
        print(h,contours[i][0])

# h[0] and h[1] not useful
# h[2] != -1 means it is there is another contour within it
# h[3] != -1 means it is a contour within another contour

# define 
min_pixels = 15 # TODO base it on the GSD and physical pumpkin size



# TODO use standard cv2 functions to determine the area and circularity of the areas for classifying if it is in fact a pumpkin
# maybe use a clustering alorithm to get more circular result of multiple pumpkins with overlapping regions

# maybe just count pixels and find out how many pixels per average pumpkin??

# probably use another criteria, also think about pumpkins that go into the same pixel cluster
# https://stackoverflow.com/questions/58182631/opencv-counting-overlapping-circles-using-morphological-operation
# https://answers.opencv.org/question/43195/detecting-overlapping-circles/
# k-means clustering?
# https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html



