import cv2 as cv
import numpy as np


blank = np.zeros((500,500,3), dtype='unint8')
cv.imshow('Blank', blank)