import cv2 as cv
import numpy as np

#blank img
blank = np.zeros((500,500,3), dtype='uint8')
cv.imshow('Blank img', blank)
# img = cv.imread('/Users/bandaruvinaykumar/Desktop/opencv/opencv-course/Resources/Photos/cat.jpg')
# cv.imshow('cat', img)

# 1. paint the img certain colour

# blank[:] = 0,255,0
# cv.imshow('Green', blank)

# 2.Drawing the Rectangle

# cv.rectangle(blank, (0,0), (250,500), (0,0,255), thickness=-1)
cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (0,255,0), thickness=-1)
cv.imshow('Rectangle', blank)

#3. Drawing the circle
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0,0,255), thickness=-1)
cv.imshow('Circle', blank)

#4 Draw a line 
cv.line(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (255,255,150), thickness=3)
cv.imshow('Line', blank)

#5 Writing the text
cv.putText(blank, 'Vinay', (250,250), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255,0,0), 2)
cv.imshow('Text ', blank)

cv.waitKey(0)
