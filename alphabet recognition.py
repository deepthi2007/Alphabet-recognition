import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import PIL.ImageOps
import os,ssl,time

if (not os.environ.get('PYTHONHTTPSVERIFY','') and getattr(ssl,'_create_verified_context',None)):
    ssl._create_default_https_context = ssl._create_unverified_context
X = np.load("image.npz")['arr_0']
y = pd.read_csv("labels.csv")['labels']
classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses = len(classes)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

x_train_scaled = x_train/255
x_test_scaled = x_test/255

clf = LogisticRegression(solver="saga",multi_class='multinomial').fit(x_train_scaled,y_train)
predict = clf.predict(x_test_scaled)
acc = accuracy_score(predict,y_test)
print(acc)

cap = cv2.VideoCapture(0)
while True :
    try:
        ret , frame = cap.read()
        gray_scale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width = cv2.shape()
        upper_left = (int(width/2-56),int(height/2-56))
        bottom_right = (int(width/2+56),int(height/2+56))
        cv2.rectangle(gray_scale,upper_left,bottom_right,(0,255),2)
        roi = gray_scale[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
        impil = Image.fromarray(roi)
        image_bw = impil.convert('L')
        image_bw_resize = image_bw.resize((22,30),Image.ANTIALIAS)
        image_bw_resize_inverted = PIL.ImageOps.invert(image_bw_resize)
        pixelfilter = 20
        min_pixel = np.percentile(image_bw_resize_inverted,pixelfilter)
        image_bw_resize_inverted_scaled = np.clip(image_bw_resize_inverted-min_pixel,0,255)
        max_pixel = np.max(image_bw_resize_inverted)
        image_bw_resize_inverted_scaled = np.asarray(image_bw_resize_inverted_scaled)/max_pixel
        test_sample = np.array(image_bw_resize_inverted_scaled).reshape(1,660)

        predict = clf.predict(test_sample)
        print(predict)

        cv2.imshow("frame",gray_scale)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()