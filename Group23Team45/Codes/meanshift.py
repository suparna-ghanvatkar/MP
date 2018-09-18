import cv2
import numpy as np
from matplotlib import pyplot as plt


for i in range(1,6):
	img = cv2.imread('dataset/'+ str(i) +'.jpg',1)   #read image in colored mode
	spatialRad = 15			# mean shift parameters  
	colorRad = 15
	maxPyrLevel = 5    #default=0, pyramid level

	#performing meanshift for different parameter values
	res1 = cv2.pyrMeanShiftFiltering(img,spatialRad,colorRad,maxPyrLevel)    
	res2 = cv2.pyrMeanShiftFiltering(img,30,30,20)
	res3 = cv2.pyrMeanShiftFiltering(img,40,40,5)
	res4 = cv2.pyrMeanShiftFiltering(img,25,25,10) 

	cv2.imshow('Original',img)
	cv2.imshow('Result1',res1)
	cv2.imshow('Result2',res2)
	cv2.imshow('Result3',res3)
	cv2.imshow('Result4',res4)
	edges = cv2.Canny(res4, 50,150)
	thresh,contours = cv2.threshold(edges, 128, 255, cv2.THRESH_BINARY)
	cv2.imwrite('out/'+str(i)+str(i)+'kmeans.jpg',contours)
	cv2.waitKey(0)
	cv2.destroyAllWindows()