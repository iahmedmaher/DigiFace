#Here there should be a call to a each function that detects a feature points set (ex: detectEyes)  
import DetectMouth as mouth
import DetectEyebrows as eyebrows

def getFeaturePoints(onlyFaces, frame = None):
    featurePoints = []
    if len(onlyFaces) > 0 and len(onlyFaces) < 5:
        #call all feature points and concatenate        
        featurePoints.append(mouth.getMouthPoints(onlyFaces, frame))
        featurePoints.appped(eyebrows.getEyebrowsPoints(onlyFaces, frame))
    return featurePoints
