import cv2
import time
import imutils
import numpy as np
from sklearn.metrics import pairwise
import time
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import glob

global loadedModel
size = 300

# resize the frame to required dimensions and predict
def predict_pothole(currentFrame):

    currentFrame = cv2.resize(currentFrame,(size,size))
    currentFrame = currentFrame.reshape(1,size,size,1).astype('float')
    currentFrame = currentFrame/255
    prob = loadedModel.predict_proba(currentFrame)
    max_prob = max(prob[0])
    if(max_prob>.90):
        return loadedModel.predict_classes(currentFrame) , max_prob
    return "none",0

# main function
if __name__ == '__main__':

    loadedModel = load_model('E:/Major 8sem/Model Files/full_model.h5')

    camera = cv2.VideoCapture(0)

    show_pred = False
    # loop until interrupted
    while (True):

        (grabbed,frame) = camera.read()
        frame = imutils.resize(frame,width = 700)
        frame = cv2.flip(frame,1)
        
        clone = frame.copy()
        
        (height,width) = frame.shape[:2]

        grayClone = cv2.cvtColor(clone,cv2.COLOR_BGR2GRAY)

        pothole,prob = predict_pothole(grayClone)

        keypress_toshow = cv2.waitKey(1)
        
        if(keypress_toshow == ord("e")):
            show_pred = not show_pred
        
        if(show_pred):
            cv2.putText(clone , str(pothole)+' '+str(prob*100)+'%' , (30,30) , cv2.FONT_HERSHEY_DUPLEX , 1 , (0,255,0) , 1)

        cv2.imshow("GrayClone",grayClone)

        cv2.imshow("Video Feed",clone)

        keypress = cv2.waitKey(1) & 0xFF

        if(keypress == ord("q")):
            break

camera.release()

cv2.destroyAllWindows()