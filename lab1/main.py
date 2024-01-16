import cv2
import numpy as np
import matplotlib.pyplot as plt

def transformacija_fft(slika):
    frekventni_dom_slike = np.fft.fft2(slika)  # Sliku pretvaramo u frekventni domen pomoću FFT (Brza Furijeova Transformacija), fft2 jer je u 2 dimenzije
    frekventni_dom_slike = np.fft.fftshift(frekventni_dom_slike)  # Pomeramo koordinatni početak u centar slike
    return frekventni_dom_slike

# Od frekvencije do prostornog domena
def inverzna_transformacija_fft(magnituda_log, kompleksni_modul):
    frekventni_dom_slike = kompleksni_modul * np.exp(magnituda_log)  # Vraćanje magnituda iz logaritma i množenje sa kompleksnim brojevima na slici
    filtrirana_slika = np.abs(np.fft.ifft2(frekventni_dom_slike))  # Funkcija ifft2 vraća sliku iz frekventnog u prostorni domen. Nije potrebno raditi ifftshift jer se to implicitno izvršava.
                                                                  # Rezultat ifft2 je opet kompleksna slika, ali nas zanima samo moduo, zato opet treba np.abs()

    return filtrirana_slika

def pronadji_i_ukloni_sumu(slika, centar):
    frekventni_dom_slike = transformacija_fft(slika)
    magnituda_slike = np.abs(frekventni_dom_slike)  # amplituda kompleksnog broja (slika u frek dom)
    kompleksni_modul_1 = frekventni_dom_slike / magnituda_slike  # cuvanje kompleksnih brojeva sa jediničnim modulom (1)
    magnituda_log = np.log(magnituda_slike)

    plt.imshow(magnituda_log, cmap="gray")
    plt.show()

    kernel = np.full((3, 3), 1)
    kernel[1][1] = -1

    magnituda_log[281, 281] = 0
    magnituda_log[231, 231] = 0
    magnituda_log[156, 356] = 0
    magnituda_log[356, 156] = 0

    plt.imshow(magnituda_log, cmap="gray")
    plt.show()

    uklonjena_periodicna_suma = inverzna_transformacija_fft(magnituda_log, kompleksni_modul_1)
    plt.imshow(uklonjena_periodicna_suma, cmap="gray")
    plt.show()

if __name__ == '__main__':
    slika = cv2.imread("slika_3.png")  # OpenCV učitava sliku u formatu BGR
    slika = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)  # Pretvaranje slike u nijanse sive

    plt.imshow(slika, cmap='gray')
    plt.show()

    centar = (256, 256)  # Slika je 512x512, pa će centar biti na poziciji (256, 256)
    poluprecnik = 50  # Poluprečnik idealnog low ili high pass filtra

    slika_sa_sumom = pronadji_i_ukloni_sumu(slika, centar)  # Dodavanje periodičnog šuma
