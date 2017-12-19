import cv2
import numpy as np
import math
import Utilities as ut
import sys

def overlayMasks(onlyFaces, featurePoints, mouthMask, eyebrowMask, eyeMask):
    if len(featurePoints) < 2 or len(onlyFaces) < 1:
        return
   

    #Add mouth Mask********************************
    try:
        #Get mouth parameters
        mouthL = featurePoints[0][0]
        mouthR = featurePoints[0][1]
        mouthWidth = mouthR[1]-mouthL[1]
        mouthHeight = math.floor(mouthWidth*0.35)
        #Set mask specific parameters
        backBGR = [25, 25, 25]
        maskL = [67, 17]
        maskR = [67, 509]
        #Overlay
        overlayMask(onlyFaces[0],mouthMask,mouthR,mouthL,backBGR,maskL,maskR,actualHeight=mouthHeight) 
    except:
        print("Unexpected error:", sys.exc_info()[0])
        pass


    #Add eyebrow mask********************************
    try:
        #Get parameters
        eyebrowL = featurePoints[1][0]
        eyebrowR = featurePoints[1][1]
        #Inverse to max i = rows j = columns
        temp = eyebrowL[0]
        eyebrowL[0] = eyebrowL[1]
        eyebrowL[1] = temp
        temp = eyebrowR[0]
        eyebrowR[0] = eyebrowR[1]
        eyebrowR[1] = temp
        #Set Mask Specific Parameter
        imaskR = [411, 651]
        imaskL = [0,0]
        #Provide rotation cuz single point
        angle = ut.getRotationFrom2Pts(eyebrowL,eyebrowR)
        #Overlay
        overlayMask(onlyFaces[0],eyebrowMask,eyebrowL,maskL=imaskL,maskR=imaskR,rotAngle=angle)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        pass

    #Add eye mask********************************
    try:
        eyeR = featurePoints[2][1][0:2]
        temp = eyeR[0]
        eyeR[0] = eyeR[1]
        eyeR[1] = temp
        radiusR = featurePoints[2][1][2]
        h,w,c = eyeMask.shape
        center = [0,0]
        center[0] = math.floor(h/2)
        center[1] = math.floor(w/2)
        overlayMask(onlyFaces[0],eyeMask,eyeR,eyeR,maskL=center,maskR=center,actualHeight=radiusR*3,actualWidth=radiusR*3,rotAngle=0)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        pass


def overlayMask(face,mask,ptR,ptL=[0,0],backBGR=[255,255,255],maskL=[-1,-1],maskR=[-1,-1],rotAngle=-500,actualWidth = 0, actualHeight = 0):
    
    intialMaskHeight,intialMaskWidth,intialMaskChannels = mask.shape
    
    #Provide mask centers and rotation angle if not provided
    if maskL == [-1,-1]:
        maskL[0] = math.floor(intialMaskHeight/2)
        maskL[1] = 0
    if maskR == [-1,-1]:
        maskR[0] = math.floor(intialMaskHeight/2)
        maskR[1] = intialMaskWidth

    #Resize mask
    if actualWidth == 0:
        actualWidth = math.floor(((ptR[1]-ptL[1])/(maskR[1]-maskL[1]))*intialMaskWidth)

    if actualHeight == 0:
        actualHeight = ptR[0]

    mask = cv2.resize(mask, (actualWidth,actualHeight))

    #Rotate Mask
    if rotAngle == -500:
        rotAngle = ut.getRotationFrom2Pts(ptL,ptR)

    transformation = cv2.getRotationMatrix2D((math.floor(actualWidth/2),math.floor(actualHeight/2)),rotAngle,1)
    mask = cv2.warpAffine(mask,transformation,(actualWidth,actualHeight))

    #Get coordinates offset
    coord = [0.0,0.0]
    coord[0] = ptR[0] - math.ceil(maskR[0]*actualHeight/intialMaskHeight) 
    coord[1] = ptR[1] - math.ceil(maskR[1]*actualWidth/intialMaskWidth) 

    #Overlay
    B = mask[0:actualHeight,0:actualWidth,0]
    G = mask[0:actualHeight,0:actualWidth,1]
    R = mask[0:actualHeight,0:actualWidth,2]

    indicesForMask = np.where(  ((B!=backBGR[0]) | (G!=backBGR[1]) | (R!=backBGR[2])) & ( (B!=0) | (G!=0) | (R!=0) ) )
    maskRegion = mask[indicesForMask]
    indicesForMaskOffset = list(indicesForMask)
    indicesForMaskOffset[0] += coord[0]
    indicesForMaskOffset[1] += coord[1]
    face[indicesForMaskOffset] = maskRegion