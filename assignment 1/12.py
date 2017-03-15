import numpy as np
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier

#category descrpition
category = {0:"Scenery", 1:"Night mode", 2:"Portrait"}

#path selection
path = './images'
image_paths = [os.path.join(path,f) for f in os.listdir(path) if not f.startswith('t')]

feats = []
labels = []

#################################TRAINING PHASE

#iterating throgh the whole training dataset
for image_path in image_paths:
    img = cv2.imread(image_path)
    l = int(os.path.split(image_path)[1].split(".")[0])
    img = cv2.resize(img, (200,200))
    z = img.reshape((-1,3))
    z = np.float32(z)
    #applying the kmeans algorithm
    cri = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
    k = 4
    ret, label, center = cv2.kmeans(z,k,None,cri, 10, cv2.KMEANS_PP_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
   
    #splitting the image to L,a,b
    res2  = cv2.cvtColor(res2, cv2.COLOR_BGR2Lab)
    h,s,v = cv2.split(res2)
    #converting the 2D component (i.e. a*) to a vector
    s = s.flatten()
    #appending all the images as vector to a matrix and similarly the labels
    feats.append(s)
    labels.append(l)

feats = np.array(feats)
labels = np.array(labels)

#defining the nearest neighbour model
model = KNeighborsClassifier(n_neighbors=35, n_jobs=-1)

#fitting the model with the training images
model.fit(feats,labels)

###################TESTING PHASE
image_paths = [os.path.join(path,f) for f in os.listdir(path) if f.startswith('t')]

#iterating through the test dataset
for image in image_paths:
    print image
    test_img = cv2.imread(image)
    timg = np.copy(test_img)
    test_img = cv2.resize(test_img, (200,200))
    z = test_img.reshape((-1,3))
    z = np.float32(z)
    cri = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
    k = 4
    ret, label, center = cv2.kmeans(z,k,None,cri, 10, cv2.KMEANS_PP_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((test_img.shape))
    res2  = cv2.cvtColor(res2, cv2.COLOR_BGR2Lab)
    h,s,v = cv2.split(res2)
    s = s.flatten()

    dest = './res/'+category[int(model.predict(s))]+'.'+os.path.split(image)[1]+'.png'
    cv2.imwrite(dest, timg)

