import cv2
from time import sleep
import DetectFace as dFace
import FacialFeatures as dFeatures
import OverlayMask as mask
import Preprocess as pre
import DetectFaceTrial as dF

video_capture = cv2.VideoCapture('http://192.168.1.5:4747/mjpegfeed')
#video_capture = cv2.VideoCapture(0)

#Load Masks Once
mouthMask = cv2.imread('mouthMask.jpg')
eyebrowMask = cv2.imread('eyebrowMask.png')

frameCounter = 0
featurePoints = []
onlyFaces = []
while True:
    frameCounter+=1
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
        onlyFaces = dFace.getFaceRegions(frame)
        featurePoints = dFeatures.getFeaturePoints(onlyFaces,frame) #frame is optional for easily debugging but your code should work if it is nil
        mask.overlayMask(onlyFaces,featurePoints, mouthMask, eyebrowMask)
    except Exception:
        pass
    #****************************

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()