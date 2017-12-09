import cv2
import numpy as np
import math

def overlayMask(onlyFaces, featurePoints):
    if len(featurePoints) < 1 or len(onlyFaces) < 1:
        return
    
    face1 = onlyFaces[0]
    #Add mouth Mask
    
    #Get mask
    leftCorner = featurePoints[0][0]
    rightCorner = featurePoints[0][1]
    mouthMask = cv2.imread('mouthMask.png')

    #Resize mask
    width = rightCorner[1] - leftCorner[1]
    height = math.floor(mouthMask.shape[0]/5)
    mouthMask = cv2.resize(mouthMask, (width,height))
    #Get coordinates
    coord = [0,0]
    coord[0] = leftCorner[0] - math.floor(height/2)
    coord[1] = leftCorner[1]

    #Overlay
    mouth = face1[coord[0]:coord[0]+height,coord[1]:coord[1]+width]
    mouth[np.where((mouthMask!=[255,255,255]).all(axis=2))] = mouthMask[np.where((mouthMask!=[255,255,255]).all(axis=2))]

    return


