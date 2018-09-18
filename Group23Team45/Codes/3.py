import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler


detector = cv2.SIFT()        #initialize detector

clas = [ 'Bikes',  'Horses']   
path = 'images\\' 

feat = []
yt = [] 
n = 0

for i in range(0, len(clas) ):
	#Getting images of class (i+1)
	print "Training image of class " + str(i+1) + clas[i]
	Images = [ f for f in listdir(path + clas[i]) if isfile(join(path + clas[i],f)) ]    #store all image of class (i+1)
	for j in range(0, len(Images)):
		#print join( path+clas[i], Images[n])
		n = n + 1
		tempImage = cv2.imread( join( path+clas[i], Images[j]) )
		kp, des = detector.detectAndCompute(tempImage,None)     #detect features 
		feat.append(des)
		yt.append(i+1)

descriptor = feat[0]
y = np.array(yt)
for i in range(0, len(feat)):
	descriptor = np.vstack((descriptor, feat[i]))
	#print descriptor.shape
#print descriptor.shape, y.shape, n


#Perform Kmeans on descriptor array
print "K Means clustering"
k = 100
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
retval, bestLabels, centers= cv2.kmeans(descriptor,k,criteria,10,0)


#Creating Histogram of clustered features
print "Creating Histogram"
X = np.zeros((n, k),dtype = np.float32)
for i in xrange(n):
	print feat[i]
	words, distance = vq(feat[i],centers)       #vector quantization
	for w in words:
		X[i][w] += 1          #bag-of-visual-words representation[]
		
		
#knn classification
print "Training K Nearest Neighbor classifier"
knn = cv2.KNearest()
knn.train(X,y)



#-------------TESTING PART----------------

print "Loading Test Images"
testPath='images/Test'
test=[]
#Getting Test Images
testImages = [ f for f in listdir(testPath) if isfile(join(testPath,f)) ]
testLength = len(testImages)

#Creating descriptors of Test Images
print "Creating Test Image descriptors"
for n in range(0, testLength):
  tempImage = cv2.imread( join(testPath,testImages[n]) )
  kp, des = detector.detectAndCompute(tempImage,None)
  test.append(des)

print "Creating Histogram for test images"
#Creating Histogram
test_features = np.zeros((testLength, k),dtype=np.float32)
for i in xrange(testLength):
    words, distance = vq(test[i],centers)
    for w in words:
        test_features[i][w] += 1

#Scaling Features
#test_features = stdSlr.transform(test_features)
  
 
#Prediction of Test Data
print "Calculating knn distances with k = 3"
ret, results, neighbours ,dist = knn.find_nearest(test_features, 3)

#Calculating Accuracy
print "Results are"
success=0.0
for i in xrange(0,testLength):
    print "Image name: ",testImages[i]
    print "Predicted class: ", results[i]
    a = 0
    b = 0
    if "horse" in testImages[i] :
       a = 0
    else:
       a=1 
    
    if int(results[i])==2:      #ie result predicts bike
        b=0
    else:
        b=1
  
    if a==b:
        success=success+1
        print "Correct"
    else:
        print "Incorrect"    

print "Toatl:"+str(testLength)+" success"+str(success)+" Accuracy: "
print (success/testLength)*100


