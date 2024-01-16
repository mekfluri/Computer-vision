import cv2
import numpy as np
import matplotlib.pyplot as plt

# Funkcija za primenu Canny detektora ivica
def Anja_Canny(img, th1, th2):
    imgEdge = cv2.Canny(img, th1, th2, apertureSize=3, L2gradient=True)
    return imgEdge

# Funkcija za prikazivanje slike
def plot_img(img, title, order, cmap='gray'):
    plt.subplot(2, 2, order)
    plt.title(title)
    plt.imshow(img, cmap=cmap)

if __name__ == '__main__':
    # Učitavanje originalne slike i konverzija u sivu sliku
    img_org = cv2.imread("coins.png")
    img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)

    # Prikazivanje originalne slike
    plot_img(img_gray, "Original", 1)

    # Filtriranje sive slike da bi se izdvojili određeni delovi
    mask_circles = cv2.inRange(img_gray, 0, 190)
    mask_filtered = cv2.morphologyEx(mask_circles, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    plot_img(mask_filtered, "Morph", 2)

    # Konverzija originalne slike u HSV prostor boja i filtriranje po zasićenju
    img_hsv = cv2.cvtColor(img_org, cv2.COLOR_BGR2HSV)
    img_sat = img_hsv[:, :, 1]
    img_sat_filtered = cv2.inRange(img_sat, 40, 255)
    img_sat_filtered = cv2.morphologyEx(img_sat_filtered, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    plot_img(img_sat_filtered, "Sat", 3)

    # Rekonstrukcija slike kombinovanjem filtriranih delova
    reconstructed = cv2.bitwise_and(mask_filtered, img_sat_filtered)
    plot_img(reconstructed, "Final", 4)

    # Prikazivanje rezultata
    plt.show()

    # Dodatno prikazivanje preklapanja originalne i rekonstruisane slike
   # plt.imshow(reconstructed, alpha=1)
   # plt.imshow(img_org, alpha=0.25)
    #plt.show()
