#Here I am going to attempt to detect the mouth point and return a list of it
from cv2 import *
import numpy as np
import sys
import os

def getMouthPoints(onlyFaces, frame = None):
    #*************************OLD CODE I WILL UPDATE TO THE STRUCTURE AND DETECT ALL MOUTH POINTS
    #img = cv2.imread('iran1.jpg')
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #lower = np.array([0,50,50])
    #upper = np.array([7,255,255])
    #mask = cv2.inRange(hsv,lower,upper)
    #components = mask

    #regions = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

    #cv2.normalize(regions[1], components, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

    #imC = cv2.applyColorMap(components, cv2.COLORMAP_JET)
    #cv2.imshow('areas',imC)

    #nLabels = regions[0]
    #stats = regions[2]

    #max = stats[1, cv2.CC_STAT_AREA]
    #maxL = 1
    #for i in range(1,nLabels):
    #    if stats[i,cv2.CC_STAT_AREA] > max:
    #        max = stats[i,cv2.CC_STAT_AREA]
    #        maxL = i

    #i = maxL
    #print("%d %d %d %d %d" % (stats[i,cv2.CC_STAT_LEFT],stats[i,cv2.CC_STAT_TOP ],stats[i,cv2.CC_STAT_WIDTH ],stats[i,cv2.CC_STAT_HEIGHT ],stats[i,cv2.CC_STAT_AREA ]))

    #res = cv2.bitwise_and(img,img, mask= mask)

    #cv2.imshow('image',img)
    #cv2.imshow('mask',mask)
    #cv2.imshow('res',regions[1])
    #waitKey()
    return []
        