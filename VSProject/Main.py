import cv2
from time import sleep
import DetectFace as dFace
import FacialFeatures as dFeatures
import OverlayMask as mask

#video_capture = cv2.VideoCapture('http://192.168.1.8:4747/mjpegfeed')
video_capture = cv2.VideoCapture(0)

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
        onlyFaces = dFace.getFaceRegion(frame)
        featurePoints = dFeatures.getFeaturePoints(onlyFaces,frame) #frame is optional for easily debugging but your code should work if it is nil
        mask.overlayMask(onlyFaces,featurePoints)
    except:
        pass
    #****************************

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()