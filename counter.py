import cv2
import numpy as np
import os

directory = "rasterized/"


def load_image_name(file_name : str, path : str = directory):
    img = cv2.imread(path+file_name)
    assert img is not None, f"Failed to load image: {file_name}"
    return img

def load_image_num(num : str): # only the original images (not rasterized from orthomosaic)
    return load_image_name(f"EB-02-660_0595_{num}.JPG", "Images for first miniproject/")

def generate_mean_and_cov_from_mask(img, mask):
    pixels = img.reshape((-1))
    mask = mask.reshape((-1))
    annotated_pixels = pixels[mask == 255].reshape((-1,3))

    mean = np.average(annotated_pixels, axis=0)
    cov = np.cov(annotated_pixels.transpose())

    return mean,cov

def get_max_possible_mahalanobis_hsv_distance(mean,cov):
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
    img_temp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mahalanobis_distance_image = get_mahalanobis_image(img_temp,mean,cov)

    # Norm the image by the maximum possible distance for consistancy across images
    mahalanobis_distance_image_norm = mahalanobis_distance_image / max_dist
    
    annotated_mahalanobis_threshold = np.zeros(mahalanobis_distance_image.shape,dtype=img_temp.dtype)
    annotated_mahalanobis_threshold[mahalanobis_distance_image_norm < 0.01] = 1
    
    mahalanobis_mask = np.reshape(
        annotated_mahalanobis_threshold, 
        (img_temp.shape[0],
         img_temp.shape[1]))
    
    return mahalanobis_mask

def generate_onetime_mahalanobis_hsv_values():
    num = 482
    num = str(num).rjust(4,'0')

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
    max_dist_mahalanobis = get_max_possible_mahalanobis_hsv_distance(mean,cov)
    
    return mean,cov,max_dist_mahalanobis


def generate_onetime_CIELAB_values():
    # cielab range ((0,-127,-127),(100,128,128))
    img = load_image_name("window_0.jpg","rasterized/")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    pixels = img.reshape((-1))
    mask = load_image_name("window_0_better_res_mask.JPG","masks/")
    mask = mask.reshape((-1))
    annotated_pixels = pixels[mask == 255].reshape((-1,3))
    range = np.array([np.min(annotated_pixels,axis=0),np.max(annotated_pixels,axis=0)])
    return range

def generate_cielab_inrange_mask(img,range=None):
    img_temp = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    mask = cv2.inRange(img_temp,range[0],range[1])
    return mask

def get_sub_contours(idx, contours, hierarchy):
    res = []
    i = hierarchy[idx][2]
    while i < len(hierarchy) and hierarchy[i][3] != -1:
        res.append(contours[i])
        i += 1
    return res


# img_ = np.array([[[int(255*0.230), int(255*0.292), int(255*0.357)]]], dtype=np.uint8) # BGR
# img_ = np.array([[[134, 149, 165]]], dtype=np.uint8) # BGR
# print(cv2.cvtColor(img_, cv2.COLOR_BGR2Lab))

DEBUG = True
MAHALANOBIS = False
CIELAB = True
TEST = False

#print(generate_onetime_mahalanobis_hsv_values())
mean = np.array([ 15.12      , 180.26666667, 225.36      ])
cov = np.array([[   1.45837838,    6.33243243,   -3.57081081],
                [   6.33243243,  103.79279279, -178.17837838],
                [  -3.57081081, -178.17837838,  825.53081081]])
max_dist_mahalanobis = 35554.24373277427

cielab_range = generate_onetime_CIELAB_values()
cielab_range[0][0] = 170
cielab_range[1][1] = 255
cielab_range[1][2] = 255
print(cielab_range)
#cielab_range = np.array([[102, 132, 143],[226, 157, 190]])


if TEST:
    num = "0482"

    img = load_image_num(num)
    H,W,_ = img.shape
    img = img[H-H//8:,W-W//8:]
    img_blurred = cv2.GaussianBlur(img,(3,3),0,borderType=cv2.BORDER_REPLICATE)

    mask = generate_cielab_inrange_mask(img_blurred,cielab_range)
        
    if DEBUG:
        print(np.unique(mask))
        num_str = str(num).rjust(4,'0')
        folder = f"output/{num_str}"
        if not os.path.exists(folder):
            os.mkdir(folder)
        annotated_image = img_blurred.copy()
        annotated_image[mask == 0] = 0
        annotated_image_inv = img_blurred.copy()
        annotated_image_inv[mask == 255] = 0
        cv2.imwrite(f"{folder}/{num_str}_annotated.JPG",
                annotated_image)
        cv2.imwrite(f"{folder}/{num_str}_annotated_inv.JPG",
                annotated_image_inv)
        cv2.imwrite(f"{folder}/{num_str}_mask.JPG",
                mask)
    exit()

overlap_px,rows,cols = list(map(int,open("rasterized/rasta_meta.csv","r").read().split("\n")[1].split(",")))


total_pumpkins = 0
for num in range(rows*cols):
    col,row = num%cols,num//cols
    img = load_image_name(f"window_{num}.jpg","rasterized/")
    H,W,_ = img.shape
    
    #img_blurred = cv2.GaussianBlur(img,(3,3),0,borderType=cv2.BORDER_REPLICATE)
    
    if MAHALANOBIS:
        mask = generate_mahalanobis_mask(img,mean,cov,max_dist_mahalanobis)
        mask = cv2.medianBlur(mask,3)

    elif CIELAB:
        mask = generate_cielab_inrange_mask(img,cielab_range)
        mask = cv2.medianBlur(mask,3)
        

    if DEBUG:
        num_str = str(num).rjust(2,'0')
        folder = f"output/{num_str}"
        if not os.path.exists(folder):
            os.mkdir(folder)
        annotated_image = img.copy()
        annotated_image[mask == 0] = 0
        annotated_image_inv = img.copy()
        annotated_image_inv[mask != 0] = 0
        cv2.imwrite(f"{folder}/{num_str}_annotated.JPG",
                annotated_image)
        cv2.imwrite(f"{folder}/{num_str}_annotated_inv.JPG",
                annotated_image_inv)
        cv2.imwrite(f"{folder}/{num_str}_mask.JPG",
                mask)
        cv2.imwrite(f"{folder}/{num_str}_blurred.JPG",
                img)

    contours, hierarchy = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0]
    #print(hierarchy[0][0:4])

    average_pumpkin_area = 64*4

    small = 35
    big = 235
    cluster = 158
    
    

    pumpkins = 0
    centers = []
    print(num,row,col)
    for i,c,h in zip(list(range(len(contours))),contours,hierarchy):
        if h[3] == -1:
            area = cv2.contourArea(c)
            if area < 15: # get rid of small areas
                continue
            if h[2] != -1: # ignore "holes" in patches of multiple pumpkins forming a ring
                inner_contours = get_sub_contours(i, contours, hierarchy)
                for inner in inner_contours:
                    inner_area = cv2.contourArea(inner)
                    if inner_area > 5:
                        area -= inner_area
            # check if the contour is touching the left or top edge, in which case ignore it, since it is "has been" counted in the overlap (unless first row or column)
            test = False
            if col != 0:
                if c[0,0][0] < overlap_px: # using for extra purpose, equivelent idea
                    # print(c[0,0][0])
                    for coord in c:
                        if coord[0][0] == 0:
                            test = True
                            break
            if test:
                continue
            
            test = False
            if row != 0:
                if c[0,0][1] < overlap_px: # using for extra purpose, equivelent idea
                    for coord in c:
                        if coord[0][1] == 0:
                            test = True
                            break
            if test:
                continue
            
            # now dont count all countours which do not thouch the boundery on the right and bottom side of the original image, but which are otherwise outside the image
            # cases
            if col != cols-1 and row == rows-1: # case one (last row)
                if c[0,0][0] > W - overlap_px:
                    for coord in c:
                        if coord[0][0] <= W - overlap_px + 1:
                            break
                    else:
                        continue
            
            if col == cols-1 and row != rows-1: # case two (last column)
                if c[0,0][1] > H - overlap_px:
                    for coord in c:
                        if coord[0][1] <= H - overlap_px + 1:
                            break
                    else:
                        continue
            
            if col != cols-1 and row != rows-1: # case three (everything else)
                if c[0,0][0] > W - overlap_px or c[0,0][1] > H - overlap_px:
                    for coord in c:
                        if coord[0][0] <= W - overlap_px + 1 and coord[0][1] <= H - overlap_px + 1:
                            break
                    else:
                        continue
            
            amount = max(1,round(area/cluster))
            
            if DEBUG:
                m = list(map(int,list(np.rint(np.mean(np.reshape(c,(-1,2)),axis=0)))))
                if amount == 1:
                    centers.append((m,-1))
                else:
                    centers.append((m,amount))
                    
            # is arbitrarely dependant on the calibration of the defined color range
            pumpkins += amount #max(1,round(area/average_pumpkin_area))

    total_pumpkins += pumpkins
    print(f"{num}: {pumpkins=}")
    
    if DEBUG:
        circle_annotated_image = img.copy()
        for center in centers:
            m,c = center
            if c == -1:
                cv2.circle(circle_annotated_image,m,2,(255,0,0),thickness=cv2.FILLED)
            else:
                cv2.putText(circle_annotated_image,str(c),(m[0]-8,m[1]+6),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.7,color=(255,0,0))
        cv2.line(circle_annotated_image,(0, H - overlap_px),(W - overlap_px, H - overlap_px),(0,0,0),2)
        cv2.line(circle_annotated_image,(W - overlap_px, 0),(W - overlap_px, H - overlap_px),(0,0,0),2)
        cv2.imwrite(f"{folder}/{num_str}_circles.JPG",
                circle_annotated_image)

print(total_pumpkins)

# h[0] and h[1] not useful
# h[2] != -1 means there is another contour within it
# h[3] != -1 means it is a contour within another contour


# Possible methods
# use standard cv2 functions to determine the area and circularity of the areas for classifying if it is in fact a pumpkin
# maybe use a clustering alorithm to get more circular result of multiple pumpkins with overlapping regions
# maybe just count pixels and find out how many pixels per average pumpkin??
# https://stackoverflow.com/questions/58182631/opencv-counting-overlapping-circles-using-morphological-operation
# https://answers.opencv.org/question/43195/detecting-overlapping-circles/
# k-means clustering?
# https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html



