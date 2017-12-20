import cv2
from time import sleep
import DetectFace as dFace
import FacialFeatures as dFeatures
import OverlayMask as mask
import Preprocess as pre
import DetectFace as dF
import sys

video_capture = cv2.VideoCapture('http://192.168.137.123:4747/mjpegfeed')
video_capture.set(cv2.CAP_PROP_BUFFERSIZE,0)

#Load Masks Once
mouthMask_number=input("Choose your mask (1,2):")
if(mouthMask_number=='1'):
    mouthMask = cv2.imread('mouthMask.jpg')
else:
   mouthMask = cv2.imread('mouthMask2.jpeg')

eyebrowMask = cv2.imread('eyebrowMask.png')
eyeMask = cv2.imread('eyeMask.jpeg')

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    
    #**********OUR PART**********
    #Temp error handling
    try:
        #frame = pre.PreProcessing(frame)
        #dF.getFaceRegions(frame)
        onlyFaces = dF.getFaceRegions(frame)
        featurePoints = dFeatures.getFeaturePoints(onlyFaces,frame) #frame is optional for easily debugging but your code should work if it is nil
        mask.overlayMasks(onlyFaces, featurePoints, mouthMask, eyebrowMask, eyeMask,mouthMask_number)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        pass
    #****************************

    # Display the resulting frame
    cv2.imshow('Welcome to DigiFace', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()