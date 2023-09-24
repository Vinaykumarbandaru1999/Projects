import cv2 as cv
img = cv.imread('/Users/bandaruvinaykumar/Desktop/opencv/opencv-course/Resources/Photos/cat_large.jpg')
img1 = cv.imread('/Users/bandaruvinaykumar/Desktop/opencv/opencv-course/Resources/Photos/cat.jpg')
cv.imshow('This is a cat', img)
cv.imshow('cat small', img1)
cv.waitKey(2000)


capture = cv.VideoCapture('/Users/bandaruvinaykumar/Desktop/opencv/opencv-course/Resources/Videos/dog.mp4')

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)
    
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()

cv.waitKey(2000)