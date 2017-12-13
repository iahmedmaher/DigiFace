import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def getEyeFeatures(cimg, EyeBrows1, EyeBrows2, Dm):
    
    """
    Get Eye pupil circle

    Inputs Parameters:
    cimg:       Color image of the full face only (detected by openCV or any other means)
    EyeBrows1:  (x,y) of the first EyeBrow Center in cimg
    EyeBrows2:  (x,y) of the second EyeBrow Center in cimg
    Dm:         Mouth Width

    Return:
    A list of 3 elements arrays (x,y,R)
    (x,y) is the center of the detected eye pupils relative to the face image
    R is the radius

    Usage Example:
    circles = DetectEye.getEyeFeatures(faceimg,(460,135),(680,130),180)
    for i in circles:
    # draw the outer circle
    cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
    """

    #De = Distance between the 2 EyeBrows centers
    De = math.sqrt(((EyeBrows1[0] - EyeBrows2[0]) ** 2) + ((EyeBrows1[1] - EyeBrows2[1]) ** 2))
    
    #width of the Eye region
    M = int(math.ceil(De + 0.4 * max(Dm,De)))
    #height of the Eye region
    N = int(math.ceil(0.7 * max(Dm,De)))

    #grayscale image (img)
    img = cv2.cvtColor(cimg[0],cv2.COLOR_BGR2GRAY)

    H, W = np.shape(img)

    x = np.zeros((2),int)
    y = np.zeros((2),int)

    #(X[0].y[0]) is the starting position of the Left Eye region
    x[0] = math.floor(EyeBrows1[0] - M / 2)
    y[0] = EyeBrows1[1]

    #(X[1].y[1]) is the starting position of the Left Eye region
    x[1] = math.floor(EyeBrows2[0] - M / 2)
    y[1] = EyeBrows2[1]

    #list containg found circles
    allcircles = []

    #if width or height is zero
    #or if the calculation results in something out of bounds
    #then there is nothing to do...
    #THIS SHOULD NOT HAPPEN anyway
    if N == 0 or M == 0 or min(x[0],x[1])<0 or (max(x[0],x[1]) + M)>=W:
        return allcircles

    #Loop for 2 eyes
    for j in range(0,2):

        #start of current Eye region
        xstart = x[j]
        ystart = y[j]
        #Extract eye from gray image to do the calculation on it
        EyeRegion = img[ystart:ystart + N, xstart:xstart + M].copy()
        
        #Use unsharp mask to sharpen the edges 
        #https://en.wikipedia.org/wiki/Unsharp_masking
        gaussian_3 = cv2.GaussianBlur(EyeRegion, ksize=(15,15), sigmaX=19.0, sigmaY=19.0)
        unsharp_image = cv2.addWeighted(EyeRegion, 1.5, gaussian_3, -0.5, 0, EyeRegion)

        #variance of each row
        vararrROW = np.zeros((N))
        #variance of each column
        vararrCOL = np.zeros((M))
        #(Approximated) variance of each pixel
        vararrALL = np.zeros((N,M))

        #calculate variance for each ROW
        for i in range(0, N):
            vararrROW[i] = np.var(unsharp_image[i,0:M])

        #calculate variance for each COLUMN
        for i in range(0, M):
            vararrCOL[i] = np.var(unsharp_image[0:N,i])

        #Estimate a value for the variance of pixel i,j
        for i in range(0, N):
            for k in range(0, M):
                vararrALL[i, k] = min(vararrCOL[k],vararrROW[i])

        #maximum estimated variance
        varmax = np.max(vararrALL)

        #Thersholding at t=0.5.
        #This is the mask to mark the eye region
        vararrALL[vararrALL < 0.50 * varmax] = 0
        vararrALL[vararrALL >= 0.50 * varmax] = varmax
        
        #Must convert to uint8 , values from 0 to 255 to be able to use opencv functions and ploting the variance 
        vararrALL = vararrALL * 255 / varmax 
        vararrALL = np.uint8(np.around(vararrALL))
        vararrALL = cv2.GaussianBlur(vararrALL,ksize=(9,9),sigmaX=10,sigmaY=10)


        #Try between the following:
        #Use adaptive theshold technique
        EyeRegion = cv2.adaptiveThreshold(unsharp_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        #Or Histogram Equalization or any other contrast streching technique if know one
        #EyeRegion = cv2.equalizeHist(unsharp_image)
        #Or maybe some sort of Thersholding the region surrounding the pupil (white part of the Eye) will work the best ?
        #but how to know the exact value
        #Or simply nothing
        #EyeRegion = unsharp_image


        #Multiply EyeRegion with Mask, Mask value should be between 0 and 1 to avoid overflow
        EyeRegion = EyeRegion * (vararrALL / 255)
        #Again we must convert to uint8 (removing decimals here) for the same reasons
        EyeRegion = np.uint8(np.around(EyeRegion))



        ##For Debugging, Shows the Mask, and the Eye masked
        #cv2.imshow('Image Processing Project', vararrALL)
        #cv2.waitKey(0)

        #cv2.imshow('Image Processing Project', EyeRegion)
        #cv2.waitKey(0)


        #Use hough transform to detect Eyes, 
        #Parameters:
        #accumulator size ration (1)
        #Minimum dist between 2 circles (40)
        #param1 : higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller).
        #param2 : accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles
                   
        circles = cv2.HoughCircles(EyeRegion,cv2.HOUGH_GRADIENT,1,40, param1=70,param2=30,minRadius=5,maxRadius=60)
        if circles is None:
            continue

        i = np.uint16(np.around(circles[0][0]))
        
        cv2.circle(cimg[0], (i[0], i[1]), 1, (0, 0, 255), 2)
        i[0] = i[0] + xstart
        i[1] = i[1] + ystart
        allcircles.append(i)

    return allcircles
