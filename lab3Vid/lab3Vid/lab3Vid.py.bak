import numpy as np
import cv2 as cv
from collections import deque

detector = cv.SIFT_create()
MIN_MATCH_COUNT = 30


def merge_images(img1, img2):
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) <= MIN_MATCH_COUNT:
        return None

   
    right_pts = list(
        filter(lambda el: kp1[el.queryIdx].pt[0] > img1.shape[1] // 2, good))
    more_right_pts = True if len(right_pts) > (len(good) // 2) else False
    left_img = img1 if more_right_pts else img2
    right_img = img2 if more_right_pts else img1
    

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    if more_right_pts:
        M, _ = cv.findHomography(pts2, pts1, cv.RANSAC, 5.0)
    else:
        M, _ = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0)
    result = cv.warpPerspective(
        right_img, M, (right_img.shape[1] + left_img.shape[1], right_img.shape[0] + left_img.shape[0]))
    res_cpy = result.copy()
    result[0:left_img.shape[0], 0:left_img.shape[1]] = left_img
    result = np.where(res_cpy == [0, 0, 0], result, res_cpy)
    non_black_pixels = np.where(result != [0, 0, 0])

    
    result = result[0:np.max(non_black_pixels[0]), 0:np.max(non_black_pixels[1])]
    return result


image1 = cv.imread('11.jpg')
image2 = cv.imread('22.jpg')
image3 = cv.imread('33.jpg')

stack = deque()
helper_stack = deque()

curr_image = image1
stack.append(image2)
stack.append(image3)

 
while len(stack) > 0:
    second_image = stack.pop()
    merged = merge_images(curr_image, second_image)
    if merged is not None:
        curr_image = merged
        while len(helper_stack) > 0:
            stack.append(helper_stack.pop())
    else:
        helper_stack.append(second_image)

if curr_image is not None:
    cv.imshow("Rezultat", curr_image)
    cv.imwrite('rezultat.jpg', curr_image)
cv.waitKey(0)
