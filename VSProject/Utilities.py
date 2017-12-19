import math

def getEuclideanDist(a,b):
    diff1 = (a[0]-b[0])**2
    diff2 = (a[1]-b[1])**2
    return (diff1+diff2)**(.5)

def getRotationFrom2Pts(pt1, pt2):
    hyp = getEuclideanDist(pt1,pt2)
    base = pt2[1] - pt1[1]
    angle = math.degrees(math.acos(base/hyp))
    if pt1[0] > pt2[0]:
        return angle
    else:
        return angle*-1




