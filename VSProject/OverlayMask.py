import cv2
import numpy as np
import math
import Utilities as ut

#THINGS SURELY NEED TO BE MODULAR*****************************************************************************
def overlayMask(onlyFaces, featurePoints, mouthMask, eyebrowMask):
    if len(featurePoints) < 2 or len(onlyFaces) < 1:
        return
    
    #Get Rotation Angle    
    angle = ut.getRotationFrom2Pts(featurePoints[1][0],featurePoints[1][1])

    face1 = onlyFaces[0]

    #Add mouth Mask
    
    #Get mask
    mouthL = featurePoints[0][0]
    mouthR = featurePoints[0][1]
    mouthWidth = mouthR[1]-mouthL[1]
    mouthHeight = math.floor(mouthWidth*0.35)

    #Mask Specific
    h,w,c = mouthMask.shape
    backBGR = [25, 25, 25]
    maskL = [67, 17]
    maskR = [67, 509]
    heightFactor = 2

    #Resize mask
    width = math.floor((mouthWidth/(maskR[1]-maskL[1]))*w)
    height = mouthHeight * heightFactor
    mouthMask = cv2.resize(mouthMask, (width,height))

    #Get Rotation Angle from two points and rotate mask
    Trans = cv2.getRotationMatrix2D((math.floor(width/2),math.floor(height/2)),angle,1)
    mouthMask = cv2.warpAffine(mouthMask,Trans,(width,height))

    #Get coordinates
    coord = [0.0,0.0]
    coord[0] = mouthL[0] - math.ceil(maskL[0]*height/h) 
    coord[1] = mouthL[1] - math.ceil(maskL[1]*width/w) 

    #Overlay
    B = mouthMask[0:height,0:width,0]
    G = mouthMask[0:height,0:width,1]
    R = mouthMask[0:height,0:width,2]

    indicesForMask = np.where( (B!=backBGR[0]) & (G!=backBGR[1]) & (R!=backBGR[2]) & (B!=0) & (G!=0) & (R!=0))
    maskRegion = mouthMask[indicesForMask]
    indicesForMaskOffset = list(indicesForMask)
    indicesForMaskOffset[0] += coord[0]
    indicesForMaskOffset[1] += coord[1]
    face1[indicesForMaskOffset] = maskRegion


    #Get mask
    eyebrowL = featurePoints[1][0]
    eyebrowR = featurePoints[1][1]

    #Mask Specific
    h,w,c = eyebrowMask.shape
    backBGR = [255, 255, 255]
    maskL = [411, 651]
    
    #Resize mask
    width = math.floor((eyebrowL[0]/maskL[1])*w)
    height = eyebrowL[1]
    eyebrowMask = cv2.resize(eyebrowMask, (width,height))

    #Get Rotation Angle from two points and rotate mask
    Trans = cv2.getRotationMatrix2D((math.floor(width/2),math.floor(height/2)),angle,1)
    eyebrowMask = cv2.warpAffine(eyebrowMask,Trans,(width,height))

    #Get coordinates
    coord = [0.0,0.0]
    coord[0] = eyebrowL[1] - math.ceil(maskL[0]*height/h) 
    coord[1] = eyebrowL[0] - math.ceil(maskL[1]*width/w) 

    #Overlay
    B = eyebrowMask[0:height,0:width,0]
    G = eyebrowMask[0:height,0:width,1]
    R = eyebrowMask[0:height,0:width,2]

    indicesForMask = np.where( (B!=backBGR[0]) & (G!=backBGR[1]) & (R!=backBGR[2]))
    maskRegion = eyebrowMask[indicesForMask]
    indicesForMaskOffset = list(indicesForMask)
    indicesForMaskOffset[0] += coord[0]
    indicesForMaskOffset[1] += coord[1]
    face1[indicesForMaskOffset] = maskRegion

    return

