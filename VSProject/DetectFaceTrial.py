import cv2 
import numpy as np
import scipy.io as io
import math


def getFaceRegions(frame):
    frameBGR = frame
    #Convert Frame to HSV
    frameHSV = cv2.cvtColor(frameBGR,cv2.COLOR_BGR2HSV)
    (h,w,channels) = frameHSV.shape

    #*****Luminance Enhancement*****
    #Divide the image into  regions and iterate over them
    yRegions = 4
    xRegions = 4
    regionHeight = math.floor(h/yRegions)
    regionWidth = math.floor(w/xRegions)
    regionSize = regionHeight*regionWidth
    for y in range(0,yRegions):
        for x in range(0,xRegions):
            yStart = y*regionHeight
            yEnd = yStart+regionHeight
            xStart = x*regionWidth
            xEnd = xStart+regionWidth
            #compuet L
            v = frameHSV[yStart:yEnd, xStart:xEnd, 2]
            cdf = np.sort(v, kind='mergesort', axis=None)
            #0.1 maps to 0.1*size
            L = cdf[math.floor(0.1*regionSize)]
            #compute z
            if L <= 50:
                z = 0
            elif L < 150:
                z = (L-50)/100
            else:
                z = 1

            frameHSV[yStart:yEnd, xStart:xEnd, 2] = 255*(  (frameHSV[yStart:yEnd, xStart:xEnd, 2]/255)**(0.75*z+0.25) + 0.4*(1-z)*(1-(frameHSV[yStart:yEnd, xStart:xEnd, 2]/255)) + (frameHSV[yStart:yEnd, xStart:xEnd, 2]/255)**(2-z)   )/2
          

    frameBGR = cv2.cvtColor(frameHSV,cv2.COLOR_HSV2BGR)
    skinRegionsBGR = np.zeros((h,w))
    B = frameBGR[0:h,0:w,0].astype(int)
    G = frameBGR[0:h,0:w,1].astype(int)
    R = frameBGR[0:h,0:w,2].astype(int)
    skinRegionsBGR[np.where( (R>95) & (G>40) & (B>20) & (R>G) & (R>B) & (abs(G-R)>15) & ((np.max(frameBGR,axis=-1)-np.min(frameBGR,axis=-1))>15) )] = 1

    frameYCrCb = cv2.cvtColor(frameBGR,cv2.COLOR_BGR2YCrCb)
    skinRegionsYCrCb = np.zeros((h,w))
    lowerBound = np.array([0,133,77])
    upperBound = np.array([256,173,127])
    skinRegionsYCrCb = cv2.inRange(frameYCrCb,lowerBound,upperBound)

    skinRegions = np.logical_and(skinRegionsBGR,skinRegionsYCrCb)

    frameGray = cv2.cvtColor(frameBGR,cv2.COLOR_BGR2GRAY)
    mean = np.mean(frameGray)
    edges = cv2.Canny(frameGray,0.15*mean,0.2*mean)
    edgesInv = np.ones((h,w))
    edgesInv[np.where( (edges>0) )] = 0
    seperateSkinRegions = np.logical_and(edgesInv,skinRegions).astype(np.uint8)

    candidates = cv2.connectedComponentsWithStats(seperateSkinRegions,stats=cv2.CV_32S)
    statsPerRegion = candidates[2]
    nRegions = candidates[0]
    centroids = candidates[3]

    onlyFaces=[]
    smallestArea = w*h*0.0065
    for r in range(0,nRegions):

        if statsPerRegion[r,cv2.CC_STAT_AREA] > smallestArea:
            width = statsPerRegion[r,cv2.CC_STAT_WIDTH]
            height = statsPerRegion[r,cv2.CC_STAT_HEIGHT]
            ratio = width/height

            if ratio > 0.5 and ratio < 1.1:
                xStart = statsPerRegion[r,cv2.CC_STAT_LEFT]
                yStart = statsPerRegion[r,cv2.CC_STAT_TOP]
                xCent = centroids[r,1] - xStart
                yCent = centroids[r,0] - yStart  

                #if xCent > 0.4*width and xCent < 0.6*width and yCent > 0.4*height and yCent < 0.6*height:
                if 0<1:
                    boxArea = width*height
                    area = statsPerRegion[r,cv2.CC_STAT_AREA]
                    extent = area/boxArea

                    if extent > 0.45 and extent < 0.9:
                        widthMargin = math.floor(width*0.15)
                        xStart -= widthMargin
                        width += 2*widthMargin
                        onlyFaces.append(frame[yStart:yStart+height,xStart:xStart+width,0:3])
                        cv2.rectangle(frame,(xStart,yStart),(xStart+width, yStart+height),(255,0,0),4)

    return onlyFaces

