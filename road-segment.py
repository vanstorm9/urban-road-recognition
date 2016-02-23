import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import imutils

def invert_img(img):
    img = (255-img)
    return img

def demi_img_fun(img):
    img_height = img.shape[0]
    img_width = img.shape[1]
    
    #img_demi = img[(img_height/2):img_height , 0:img_width]
    #img_demi = img[(img_height/2):img_height , (img_width/4):3*(img_width/4)]
    #img_demi = img[4*(img_height/5):img_height , 0:img_width]   # Don't use this, it fails on some test cases
    img_demi = img[4*(img_height/5):img_height , (img_width/4):3*(img_width/4)]

    return img_demi

def histogram_backprojection(img, img_demi):
    hsvt = cv2.cvtColor(img_demi,cv2.COLOR_BGR2HSV)
    hsvt_f = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # calculating object histogram
    roihist = cv2.calcHist([hsvt],[0, 1], None, [180, 256], [0, 180, 0, 256] )
     
    # normalize histogram and apply backprojection
    cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([hsvt_f],[0,1],roihist,[0,180,0,256],1)
     
    # Now convolute with circular disc
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cv2.filter2D(dst,-1,disc,dst)
     
    # threshold and binary AND
    ret,thresh = cv2.threshold(dst,50,255,0)
    thresh_one = thresh.copy()
    thresh = cv2.merge((thresh,thresh,thresh))
    #res = cv2.bitwise_and(img,thresh)

    return thresh

def morph_trans(img):
    # Implementing morphological erosion & dilation
    kernel = np.ones((2,2),np.uint8)  # (6,6) to get more contours (9,9) to reduce noise
    img = cv2.erode(img, kernel, iterations = 4) # Shrink to remove noise
    img = cv2.dilate(img, kernel, iterations=4)  # Grow to combine stray blobs

    return img

def watershed(img, thresh):
    # noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 4)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    #sure_bg = cv2.morphologyEx(sure_bg, cv2.MORPH_TOPHAT, kernel)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    '''
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgray = cv2.GaussianBlur(imgray, (5, 5), 0)
    img = cv2.Canny(imgray,200,500)
    '''
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]

    return sure_bg, sure_fg


#img = cv2.imread('images/coins_clustered.jpg')


img = cv2.imread('images/road_4.jpg')
#img = cv2.imread('images/road_0.bmp')

img = imutils.resize(img, height = 300)

img_height = img.shape[0]
img_width = img.shape[1]

# Getting the lower part of the image
img_demi = demi_img_fun(img)
thresh = histogram_backprojection(img, img_demi)
#thresh = morph_trans(thresh)



cv2.imshow('original', img)
cv2.imshow('result', thresh)
cv2.waitKey(0)

















'''

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


thresh = invert_img(thresh)

sure_bg, sure_fg = watershed(img, thresh)


cv2.imshow('background',sure_bg)
cv2.imshow('foreground',sure_fg)
cv2.imshow('threshold',thresh)
cv2.imshow('result',img)


cv2.waitKey(0)
'''
