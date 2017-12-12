#Here there should be a call to a each function that detects a feature points set (ex: detectEyes)  
import DetectMouth as mouth
import DetectEyebrows as eyebrows
import DetectEye as eye
import matplotlib.pyplot as plt
import cv2

def getFeaturePoints(onlyFaces, frame = None):
    featurePoints = []
    if len(onlyFaces) > 0 and len(onlyFaces) < 5:
        #call all feature points and concatenate        
        #featurePoints.append(eyebrows.getEyebrowsPoints(onlyFaces, frame))
        #plt.imshow(onlyFaces[0])
        plt.imshow(cv2.cvtColor(onlyFaces[0], cv2.COLOR_BGR2RGB))
        corners = mouth.getMouthPoints(onlyFaces,frame)
        centers = eyebrows.getEyebrowsPoints(onlyFaces, frame)

        #mwidth = corners[1][0] - corners[0][0]

        circles = eye.getEyeFeatures(onlyFaces,centers[0],centers[1],0)
        # draw the outer circle
        for i in circles:
            cv2.circle(onlyFaces[0], (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(onlyFaces[0], (i[0], i[1]), 2, (0, 0, 255), 3)

        #plt.show()

    return featurePoints