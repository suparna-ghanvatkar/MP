import cv2
import numpy as np
from matplotlib import pyplot as plt

#reading the image
img = cv2.imread('res/Lenna.jpg',cv2.IMREAD_GRAYSCALE)

#Defining the kernel matrix for Roberts operator and them applying the Roberts operator
roberts_y = np.array(([0,0,0],[0,1,0],[0,0,-1]),dtype="int")
roberts_x = np.array(([0,0,0],[0,0,1],[0,-1,0]), dtype = "int")
rob_x = cv2.filter2D(img, -1, roberts_x)
rob_y = cv2.filter2D(img, -1, roberts_y)

# h-stacking the images and then displaying
#res = np.hstack((img,rob_x, rob_y))
#cv2.imwrite("out/image.png", res)
plt.figure(1)
plt.subplot(131)
plt.imshow(img,cmap = plt.cm.gray)
plt.title('Original')
plt.subplot(132)
plt.imshow(rob_x,cmap = plt.cm.gray)
plt.title('Roberts_X')
plt.subplot(133)
plt.imshow(rob_y, cmap = plt.cm.gray)
plt.title('Roberts_Y')
plt.show()
#cv2.waitKey(0)
