import numpy as np
import cv2
import pickle

cap = cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)

pickle_id = open("model_trained_10.p","rb")
model = pickle.load(pickle_id)

def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    return img
    
while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img,(320,320))
    img = preProcessing(img)
    cv2.imshow("Processed Image",img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
