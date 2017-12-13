import cv2
import DetectMouth as mouth

testImgs = []
testImgs.append(cv2.imread('iran1.JPG'))
mouth.getMouthPoints(testImgs)
cv2.imshow('Test',testImgs[0])
cv2.waitKey(0)