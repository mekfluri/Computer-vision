import numpy as np
import cv2 as cv
from collections import deque

# Kreiranje SIFT detektora
detector = cv.SIFT_create()

# Minimalni broj podudaranja potrebnih za uspešno spajanje slika
MIN_MATCH_COUNT = 30


def merge_images(img1, img2):
    # Pronalaženje ključnih tačaka i opisa za obe slike
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    # Konfiguracija FLANN Matcher-a
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # Pronalaženje najboljih podudaranja
    matches = flann.knnMatch(des1, des2, k=2)

    # Filtriranje dobrih podudaranja na osnovu distanci
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]

    # Provera minimalnog broja dobrih podudaranja
    if len(good) <= MIN_MATCH_COUNT:
        return None

    # Određivanje orijentacije i redosleda slika
    right_pts = [kp1[el.queryIdx].pt[0] > img1.shape[1] // 2 for el in good]
    more_right_pts = len([pt for pt in right_pts if pt]) > (len(good) // 2)
    left_img, right_img = (img1, img2) if more_right_pts else (img2, img1)

    # Priprema tačaka za određivanje homografije
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Pronalaženje homografije pomoću RANSAC algoritma
    if more_right_pts:
        M, _ = cv.findHomography(pts2, pts1, cv.RANSAC, 5.0)
    else:
        M, _ = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0)

    # Spajanje slika pomoću homografije
    result = cv.warpPerspective(
        right_img,
        M,
        (right_img.shape[1] + left_img.shape[1], right_img.shape[0] + left_img.shape[0]),
    )

    # Kopiranje rezultujuće slike
    res_cpy = result.copy()

    # Postavljanje leve slike na odgovarajuće mesto
    result[0 : left_img.shape[0], 0 : left_img.shape[1]] = left_img

    # Zamena crnih piksela u rezultujućoj slici sa odgovarajućim pikselima iz kopije
    result = np.where(res_cpy == [0, 0, 0], result, res_cpy)

    # Uklanjanje viška ivica u rezultatu
    non_black_pixels = np.nonzero(result != [0, 0, 0])
    result = result[0 : np.max(non_black_pixels[0]), 0 : np.max(non_black_pixels[1])]

    return result


# Učitavanje slika
image1 = cv.imread("11.jpg")
image2 = cv.imread("22.jpg")
image3 = cv.imread("33.jpg")

# Inicijalizacija steka i pomoćnog steka
stack = deque()
helper_stack = deque()

# Postavljanje trenutne slike na prvu učitanu sliku
curr_image = image1
# Dodavanje preostalih slika na stek
stack.append(image2)
stack.append(image3)

# Spajanje slika sve dok ima slika na steku
while stack:
    second_image = stack.pop()
    merged = merge_images(curr_image, second_image)
    if merged is not None:
        # Ako je uspešno spojeno, postavljanje trenutne slike na spojeni rezultat
        curr_image = merged
        # Dodavanje slika sa pomoćnog steka na stek
        stack.extend(reversed(helper_stack))
        helper_stack.clear()
    else:
        # Ako nije uspešno spojeno, dodavanje slike na pomoćni stek
        helper_stack.append(second_image)

# Prikazivanje rezultata
if curr_image is not None:
    cv.imshow("Rezultat", curr_image)
    cv.imwrite("rezultat.jpg", curr_image)
    cv.waitKey(0)
