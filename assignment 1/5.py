import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

#reading the image
img = cv2.imread("Face.png",cv2.IMREAD_GRAYSCALE)
img1 = img

#Whitened Image
x, y = img.shape
mean = np.mean(img)
variance = np.var(img)
std = math.sqrt(variance)

temp = np.zeros(img.shape)
for i in range( x ):
	for j in range( y ):
		temp[i,j] = (img[i,j] - mean) / std

#Histogram Equilized
histEq = cv2.equalizeHist(img1)

#Showing the images
res = np.hstack((img1,temp,histEq)) #stacking images side-by-side
cv2.imwrite('./out/res.png',res)
cv2.imshow('white',temp)
cv2.waitKey(0)
plt.figure(1)
plt.subplot(131)
plt.imshow(img1,cmap = plt.cm.gray)
plt.title('Original')
plt.subplot(132)
plt.imshow(temp,cmap = plt.cm.gray)
plt.title('Whitened Image')
plt.subplot(133)
plt.imshow(histEq, cmap = plt.cm.gray)
plt.title('Histogram Equalized')
plt.show()
