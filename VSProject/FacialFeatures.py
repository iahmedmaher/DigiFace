#Here there should be a call to a each function that detects a feature points set (ex: detectEyes)  
import DetectMouth as mouth
import DetectEyebrows as eyebrows
import DetectEye as eyes

def getFeaturePoints(onlyFaces, frame = None):
    featurePoints = [[],[],[]]
    if len(onlyFaces) > 0 and len(onlyFaces) < 5:
        #call all feature points and concatenate        
        featurePoints[0] = mouth.getMouthPoints(onlyFaces, frame)

        featurePoints[1] = eyebrows.getEyebrowsPoints(onlyFaces, frame)

        mouthWidth = featurePoints[0][0][1] - featurePoints[0][1][1]
        featurePoints[2] = eyes.getEyeFeatures(onlyFaces,featurePoints[1][0],featurePoints[1][1],mouthWidth)

    return featurePoints
