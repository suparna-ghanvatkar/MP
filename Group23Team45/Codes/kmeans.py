import cv2
import numpy as np

#performing kmeans on basis of two features- intensity and color
for i in range(1,6):

	img = cv2.imread('dataset/' + str(i) + '.jpg',1)
	blur = cv2.medianBlur(img,3)
	blur_temp = blur.copy()
	x = blur.reshape((-1,3))  #taking R,G,B in different columns

	img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	y = img2.reshape((-1,1))    #taking intensity into a column
	
	z = np.column_stack((x,y))   #stacking the 4 columns together-color and intensity
	z = np.float32(z)      #conversion to float32, kmeans func requires float32 input type
	
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)    #criteria to stop the iterations
	k = 2      #no. of clusters

	ret,label,center=cv2.kmeans(z,k,criteria,10,cv2.KMEANS_RANDOM_CENTERS)     #apply kmeans func. on basis of both intensity and color
	

	# Now convert back into uint8, and make original image
	center = np.uint8(center)
	res = center[label.flatten()]

	resNew=res[:,[0,1,2]] 				# to pick up all rows and only R,G,B columns
	res2 = resNew.reshape((img.shape))     #to reshape into that of original image
	
	#med2 = cv2.medianBlur(res2,3)
	cv2.imshow('Original',img) 
	#img2  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)     #image to grayscale to get intensity
	#h,s,v = cv2.split(img2)
	#s = s.flatten()
	edges = cv2.Canny(res2, 50,150)
	thresh,contours = cv2.threshold(edges, 128, 255, cv2.THRESH_BINARY)
	cv2.imshow('kmeans',contours)
	cv2.imshow('color',res2)
	cv2.imshow('blurred',blur_temp)
	#cv2.imshow('after',med2)
	#cv2.imshow('KMeans Segmentation',res2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

