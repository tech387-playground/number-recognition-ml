import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

# Configuration
path='data'

images = []
classNo = []
myList = os.listdir(path)
print("Total number of classes detected:",len(myList))
noofClasses = len(myList)
print("Importing classes...")
for x in range(0,noofClasses):
    myPickList = os.listdir(path+'/'+str(x))
    for y in myPickList:
        curImg = cv2.imread(path+'/'+str(x)+'/'+y)
        curImg = cv2.resize(curImg,(32,32))
        images.append(curImg)
        classNo.append(x)
    print(x,end=" ")
print("")

images = np.array(images)
classNo = np.array(classNo)
print(images.shape)

# Spliting the data
X_train, X_test, Y_train, Y_test = train_test_split(images,classNo,test_size=0.2)
X_train,X_validation,Y_train,Y_validation = train_test_split(X_train,Y_train,test_size=0.2)

print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

print(np.where(Y_train==0))