import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import imutils
import time
def load_image(path):
    img=cv.imread(path)
    return img

def load_and_crop_image(path):
    img= load_image(path)

    img_crop=img[162:883,192:1634]
    plt.imshow(img_crop)
    plt.show()
    return img_crop

def pyramid(img,scale=1.5,min_size=(30,30)):
    yield img
    while True:
        w=int(img.shape[1]/scale)
        img=imutils.resize(img,width=w)
        if img.shape[0]<min_size[1] or img.shape[1]<min_size[0]:
            break
        yield img

def sliding_window(img,stepSize, windowSize):
    for y in range(0,img.shape[0],stepSize):
        for x in range(0,img.shape[1],stepSize):
            yield(x,y,img[y:y+windowSize[1],x:x+windowSize[0]])


if __name__=="__main__":
    img=load_and_crop_image('download.png')
    winW,winH=(180,180)
    rows = open("synset_words.txt").read().strip().split("\n")
    classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
    step_size=180
    net=cv.dnn.readNetFromCaffe('bvlc_googlenet.prototxt','bvlc_googlenet.caffemodel')
    num_resize=-1
    scale=2
    for resized in pyramid(img,scale=scale,min_size=(180,180)):
        num_resize+=1
        for(x,y,window) in sliding_window(resized, stepSize=step_size,windowSize=(winW,winH)):
            if window.shape[0]!=winH or window.shape[1] !=winW:
                continue
            blob = cv.dnn.blobFromImage(window, 1, (224, 224), (104, 117, 123))
            net.setInput(blob)
            start = time.time()
            preds = net.forward()
            end = time.time()
            print("[INFO] classification took {:.5} seconds".format(end - start))
            idxs = np.argsort(preds[0])[::-1][:5]
            for (i, idx) in enumerate(idxs):
                if i == 0 and preds[0][idx]>0.4:
                    if "cat" in classes[idx]:
                        cv.putText(img, f'CAT - {"%.3f" % (preds[0][idx] * 100)}%', (x*(scale**num_resize) + 5, y*(scale**num_resize) + 15),cv.FONT_HERSHEY_COMPLEX,0.5, (0, 0,255), 2)
                        cv.rectangle(img,(x*(scale**num_resize), y*(scale**num_resize)), (x*(scale**num_resize) + (scale**num_resize)*winW, y*(scale**num_resize) +(scale**num_resize)* winH),(0, 0,255 ), 2)
                    elif "dog" in classes[idx]:
                        cv.putText(img, f'DOG - {"%.3f" % (preds[0][idx] * 100)}%', (x*(scale**num_resize) + 5, y*(scale**num_resize) + 15),cv.FONT_HERSHEY_COMPLEX,0.5, (0, 255, 255), 2)
                        cv.rectangle(img,(x*(scale**num_resize), y*(scale**num_resize)), (x*(scale**num_resize) + (scale**num_resize)*winW, y*(scale**num_resize) +(scale**num_resize)* winH),(0, 255, 255), 2)

            clone = resized.copy()
            cv.rectangle(clone, (x, y), (x +winW, y + winH), (0, 255, 0), 2)
            cv.imshow("Window", clone)
            cv.waitKey(1)

            # time.sleep(1.025)


cv.imshow("result",img)
cv.waitKey(0)
cv.imwrite('output.jpg',img)