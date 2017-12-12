import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import DetectFace as dFace
import FacialFeatures as dFeatures
import OverlayMask as mask

video_capture = cv2.VideoCapture(2)

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    #**********OUR PART**********
    onlyFaces = dFace.getFaceRegion(frame)
    featurePoints = dFeatures.getFeaturePoints(onlyFaces,frame) #frame is optional for easily debugging but your code should work if it is nil
    #mask.overlayMask(onlyFaces, featurePoints)
    #****************************

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

    #eyebrows.getEyebrowsPoints(onlyFaces,frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()