#Here I am going to attempt to detect the mouth point and return a list of it
import cv2
import numpy as np
import sys
import os
import math
import Utilities as ut

def getMouthPoints(onlyFaces, frame = None):

    #Get ROI which is the bottom half of the face
    nRows = len(onlyFaces[0]) 
    halfnRows = math.floor(nRows/2)
    nColumns = len(onlyFaces[0][0])
    bgrFace = (onlyFaces[0])[halfnRows:nRows,0:nColumns]
    #Get mouth region

    #First get mask of regions possible to be a mouth by color
    hsvFace = cv2.cvtColor(bgrFace, cv2.COLOR_BGR2HSV)    
    lower = np.array([0,90,0])
    upper = np.array([7,200,255])
    mask = cv2.inRange(hsvFace,lower,upper)
    #We have to do it twice because red is at 0 angle
    lower = np.array([175,90,0])
    upper = np.array([179,200,255])
    mask += cv2.inRange(hsvFace,lower,upper)
    components = mask

    imC = cv2.applyColorMap(components, cv2.COLORMAP_JET)
    cv2.imshow('areas',imC)

    #Get mouth as the biggest connected component in range of the color threshold
    regionAnalysis = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    nLabels = regionAnalysis[0]
    labels = regionAnalysis[1]
    statsPerRegion = regionAnalysis[2]

    if len(statsPerRegion) > 1: 

        cv2.normalize(labels, components, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

        #Get max area
        maxArea = statsPerRegion[1, cv2.CC_STAT_AREA]
        maxLabel = 1
        for i in range(1,nLabels):
            if statsPerRegion[i,cv2.CC_STAT_AREA] > maxArea:
                maxArea = statsPerRegion[i,cv2.CC_STAT_AREA]
                maxLabel = i

        i = maxLabel
        boundingCoord = [0,0]
        boundingCoord[1] = statsPerRegion[i,cv2.CC_STAT_LEFT]
        boundingCoord[0] = statsPerRegion[i,cv2.CC_STAT_TOP ]
        mouthWidth = statsPerRegion[i,cv2.CC_STAT_WIDTH ]
        mouthHeight = statsPerRegion[i,cv2.CC_STAT_HEIGHT ]

        #Convert face to grayscale and apply contrast stretching
        gryFace = cv2.cvtColor(bgrFace, cv2.COLOR_BGR2GRAY)
        gryFace = cv2.equalizeHist(gryFace)

        #Get intial mouth corners then iterate
        corners = []
        corners.append(iterateForMouthPoint(-1, maxLabel, bgrFace, boundingCoord, nColumns, gryFace, mouthHeight, mouthWidth, labels))
        corners.append(iterateForMouthPoint(1, maxLabel, bgrFace, boundingCoord, nColumns, gryFace, mouthHeight, mouthWidth, labels))

        #Neutralize effect of halving face for coord
        corners[0][0] += halfnRows
        corners[1][0] += halfnRows
        
    return corners

def iterateForMouthPoint(direction, maxLabel, bgrFace, boundingCoord, faceWidth, gryFace, mouthHeight, mouthWidth, labels):
    
    #Get intial mouth corner
    minIntensity = 255
    nMinPixels = 0
    minIndex = 0
    if direction == 1:
        x = boundingCoord[1]+mouthWidth-1
    else:
        x = boundingCoord[1]+1     
    
    for y in range(boundingCoord[0],boundingCoord[0]+mouthHeight):
        if labels[y,x] == maxLabel:
            if gryFace[y,x] < minIntensity:
                minIntensity = gryFace[y,x]
                nMinPixels = 1
                minIndex = y
            elif  gryFace[y,x] == minIntensity:
                nMinPixels+=1               

    #Get midpoint of min intensity
    corner = [0,0]
    minMid = minIndex + math.floor(nMinPixels/2)
    corner[0] = minMid
    corner[1] = x


    #Iterate to get best corner

    #First for the right corner
    iterationArr = gryFace[boundingCoord[0]:boundingCoord[0]+mouthHeight, corner[1]]
    i=0
    max = 0
    maxIndex = corner
    allVariances = []
    variance = cv2.Laplacian(iterationArr,cv2.CV_64F).var()
    while abs(variance) > 9 and abs(i) < faceWidth*0.1:
        
        for j in range(boundingCoord[0],boundingCoord[0]+mouthHeight):
            candidate = [j,corner[1]+i]
            dist = ut.getEuclideanDist(candidate, corner)
            invGray = gryFace[candidate[0], candidate[1]]
            if dist < 20 and dist > 0 and invGray > 0: 
                func = 1/invGray + 1/dist
                if max < func:
                    max = func
                    maxIndex = [candidate[0],candidate[1]]
        allVariances.append(variance)
        i+=direction
        iterationArr = gryFace[boundingCoord[0]:boundingCoord[0]+mouthHeight, corner[1]+i]
        gradient = cv2.Laplacian(iterationArr,cv2.CV_64F)
        variance = gradient.var()
    cv2.circle(bgrFace,(maxIndex[1],maxIndex[0]),2,(0,255,0))

    return corner
        