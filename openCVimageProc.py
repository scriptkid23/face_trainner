import sys
import cv2,os
import numpy as np
import pickle

imagePath   = "imgTest\\avt.jpg"

faceCascade = cv2.CascadeClassifier("data\haarcascade_frontalface_default.xml") 
#faceCascade = cv2.CascadeClassifier("venv\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml")
#faceCascade = cv2.CascadeClassifier("ML_python\\cascardes\\data\\haarcascade_profileface.xml")
image = cv2.imread(imagePath)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("database\\trainner.yml")
labels = {"person_name": 1}

with open("database\\labels.pickle",'rb') as f:

    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5) # chỉnh scaleFactor ảnh hưởng đến nhận diện gương mặt 

for(x,y,w,h) in faces:
    roi_gray = gray[y:y+h,x:x+w]
    id_,conf  = recognizer.predict(roi_gray)
    #print(id_)
    #print(conf)
    if conf >=4 and conf <=170:

        font = cv2.FONT_HERSHEY_SIMPLEX
        name = labels[id_]
        color = (255,253,255) # color for name
        stroke = 2
        cv2.putText(image,name,(x,y),font,0.8,color,stroke,cv2.LINE_AA)

    color = (255,0,0)  #color for rectangle
    stroke = 2
    end_cord_x = x + w
    end_cord_y = y + h

    cv2.rectangle(image,(x,y),(end_cord_x,end_cord_y),color,stroke)
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#D:\code\image_recognition\database\labels.pickle
#D:\code\image_recognition\database\trainner.yml
