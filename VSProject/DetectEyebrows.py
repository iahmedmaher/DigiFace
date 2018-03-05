import numpy as np
import cv2
import scipy
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage
import pylab as p

def getEyebrowsPoints(onlyFace, frame = None):
    
    """
    getEyebrowsPoints
    
    Input Parameters:
    onlyface    : BGR image containing the face detected
    frame       : frame captured, used for debugging
    
    Returns:
    centers     : 2d numpy array
    first row contains the x and y coordinate of the center of the left eyebrow
    second row contains the x and y coordinate of the center of the right eyebrow
    
    """
    
    height = np.size(onlyFace[0],1)
    height_half = height/2
    width = np.size(onlyFace[0],0)
    
    # converting to grayscale
    face_gray = cv2.cvtColor(onlyFace[0],cv2.COLOR_BGR2GRAY)

    #pre-processing, noise reduction using gaussian
    gauss_filtered = cv2.GaussianBlur(face_gray,(11, 11),3)

    #Applying thresholding to make it binarized
    threshold = cv2.adaptiveThreshold(gauss_filtered,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
 
    # horizontal edge detection using Sobel
    edges_horizontal_sobel = ndimage.sobel(threshold,0)
   
    # Eroding the edges to reduce noise
    eroded_image = cv2.erode(edges_horizontal_sobel,(5,5),iterations = 1)

    #getting the connected components
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded_image)

    #creating margins in order to eliminate outliers
    margin_left = int(width * 0.20)
    margin_right = int(width * 0.80)
    margin_center_left = int(width * 0.45)
    margin_center_right = int(width * 0.55)
    margin_up = int(height * 0.20)
    margin_down = int(height * 0.40)

    min_index_l = 0
    min_index_r = 0
    ymin_l = height
    ymin_r = height

    #conditions on which centroids to take
    iterations = 10
    delta_y_l = 0
    delta_y_r = 0

    for i in range(len(centroids)):
        if stats[i,cv2.CC_STAT_AREA] > 50:
            if centroids[i,1] >margin_up and centroids[i,1] < margin_down and centroids[i,0] > margin_left and centroids[i,0] <margin_right:
                if centroids[i, 0] < margin_center_left:
                    if centroids[i,1] < ymin_l:
                        ymin_l = centroids[i,1]
                        min_index_l = i
                elif centroids[i,0] > margin_center_right:
                    if centroids[i, 1] < ymin_r:
                        ymin_r = centroids[i, 1]
                        min_index_r = i

    #getting left and right coordinates
    x_l = np.int(centroids[min_index_l,0])
    y_l = np.int(centroids[min_index_l, 1])
    x_r = np.int(centroids[min_index_r,0])
    y_r = np.int(centroids[min_index_r, 1])

    # un/comment to check margings locations
    #plt.imshow(onlyFace[0])
    # plt.plot(x_l, y_l, "ro")
    # plt.plot(x_r, y_r, "ro")
    # plt.hlines(margin_down,margin_center_left,margin_center_right,)
    # plt.hlines(margin_up,margin_left,margin_right)
    # plt.vlines(margin_center_right,0,height_half)
    # plt.vlines(margin_center_left,0,height_half)
    #plt.show()

    # drawing 2 circles around the centers calculated
    cv2.circle(onlyFace[0], (x_l, y_l),1, (255, 0, 0),2)
    cv2.circle(onlyFace[0],(x_r, y_r),1, (255, 0, 0),2)

    centers = np.array([[x_l,y_l],[x_r,y_r]])
    return centers
