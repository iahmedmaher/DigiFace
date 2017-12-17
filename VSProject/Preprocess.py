import cv2
import numpy as np

##*def ColorHistogramEnhace(frame):
#    levels=255
#    rows, columns, dims = frame.shape
#    I_out=np.zeros(frame.shape)
#    if(dims>1):
#        I_grayScaleimage=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#    else:
#        I_grayScaleimage=frame
#    H=np.zeros(levels)
#    for i in range(0,columns):
#        for j  in range(0,rows):
#            val=I_grayScaleimage[j,i]
#            H[val]=H[val]+1
            
#    H_c= np.copy(H)
#    for s in range(1,H_c.shape):
#        H_c[i]=H_c[i]+H_c[i-1]
#    for b in range(0,columns):
#        for n in range(0,rows):
#            for m in range(0,dims):
#                I_out[i,j,k]=round((levels-1)*H_c(frame[i,j,k]+1)/(columns*rows))
#    return I_out*#
def PreProcessing(frame):
    equilized_image=frame;
    blue_channel, green_channel, red_channel = cv2.split(equilized_image)
    #green_channel=equilized_image( :, :, 2)
    #blue_channel=equilized_image( :, :, 3)
    red_channel_filtered= cv2.medianBlur(red_channel,3)
    green_channel_filtered=cv2.medianBlur(green_channel,3)
    blue_channel_filtered=cv2.medianBlur(blue_channel,3)
    recontructed_image= cv2.merge((blue_channel_filtered,green_channel_filtered,red_channel_filtered))
    return recontructed_image
        


