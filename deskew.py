import numpy as np
import argparse
import cv2
import os
from PIL import Image
import pytesseract
from pytesseract import Output
from time import sleep

def main():
    #gets imagepath, textpath, and finalplace for text files
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
	   help="path to input image file")
    ap.add_argument("-t", "--textfile", required=True,
	   help="path to text file")
    ap.add_argument("-f", "--finalplace", required=True,
	   help="path to final place")
    args = vars(ap.parse_args())
    f = args["textfile"]
    img = cv2.imread(args["image"])
    imageswritten = 0
    #run cropping and write images to files
    print(imageswritten)
    thresh = processimage(img)
    angle = getangle(thresh)
    threshrotated = deskewimage(angle, thresh)
    i = deskewimage(angle, img)
    imageswritten = averageline(threshrotated, i, imageswritten, angle, f, args["finalplace"])


def processimage(image):
    #converts image to grayscale to be able to thresh
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    #threshes image, makes text white and background black
    thresher = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return thresher

def averageline(thresh, image, imageswritten, angle, file, finalplace):
    averageline = []
    h, w = thresh.shape
    added = 0
    count = 0
    #determines if line contains text or is blank based on black and white pixels
    for row in thresh:
        for pixel in row:
            if (pixel == 255):
                averageline.append(1)
                break
        #if (added >= 3):
            #averageline.append(1)
        else:
            averageline.append(0)
        added = 0
        count += 1
    imageswritten = crop(imageswritten, averageline, image, angle, file, finalplace)
    return imageswritten

#draws lines on images
def draw(imageswritten, averageline, image, angle):
    h, w, t = image.shape
    #line = image[0:x]
    #cv2.imwrite("%d.jpg" % imageswritten, line)
    #return image[x:h]

    for i in range(0, len(averageline)):
        if (averageline[i] == 0):
            cv2.line(image, (0, i), (w, i), (0, 255, 0), 1)
    image = deskewimage((-1*angle), image)

#gets rotation angle for thresh
def getangle(thresh):
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
	       angle = -(90 + angle)
    else:
	       angle = -angle
    return angle

#rotates image
def deskewimage(angle, image):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
	   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#crops images and runs tesseract to get text
def crop(imageswritten, averageline, image, angle, file, finalplace):
    h, w, t = image.shape
    cropend = 0
    line = []
    textpath = file
    for i, l in enumerate(averageline):
        if ((l == 1) and (i > cropend)):
            cropstart = i
            try:
                index = averageline[i:].index(0)
                cropend = i + index
                line = image[cropstart:cropend, 0:w]
                cv2.imwrite(finalplace + "/%d.jpg" % imageswritten, line)
                imageswritten += 1
                rgb = cv2.cvtColor(line, cv2.COLOR_BGR2RGB)
                temp = pytesseract.image_to_string(Image.open(finalplace + "/%d.jpg" % imageswritten))
                if (temp != ""):
                    file = open(textpath, "a")
                    file.write(temp + "\n")
                    file.close()
                past = temp
            except ValueError:
                index=-1
    return(imageswritten)


main()
