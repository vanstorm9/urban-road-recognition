import cv2
import numpy as np

def color_seg(choice):
    if choice == 'blue':
        lower_hue = np.array([110,50,50])
        upper_hue = np.array([130,128,255])
    elif choice == 'white':
        lower_hue = np.array([0,0,0])
        upper_hue = np.array([0,0,255])
    elif choice == 'black':
        lower_hue = np.array([0,0,0])
        upper_hue = np.array([0,0,100])
    return lower_hue, upper_hue


cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # define range of a color in HSV



    lower_hue, upper_hue = color_seg('black')
    

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_hue, upper_hue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
