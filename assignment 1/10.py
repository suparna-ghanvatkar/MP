import cv2
import numpy as np
from matplotlib import pyplot as plt

#reading the image
img = cv2.imread('res/Lenna.jpg', cv2.IMREAD_GRAYSCALE)
im = img.copy()

#Taking the laplacian
lap = cv2.Laplacian(im, cv2.CV_64F)
#Defining the kernel matrix for Laplacian operator and them applying the laplacian operator
kernel = np.array(([0,-1,0],[-1,4,-1],[0,-1,0]),dtype = "int")
sharp = cv2.filter2D(img,-1, kernel)


#Sharpening of the image using laplacian edges
sharp = sharp + img
sharp = sharp - np.amin(sharp)
sharp = sharp * ( 255.0 / np.amax(sharp))
#res = np.hstack((img,lap, sharp))
#cv2.imwrite("out/Sharpened.png",res)
plt.figure(1)
plt.subplot(131)
plt.imshow(img,cmap = plt.cm.gray)
plt.title('Original')
plt.subplot(132)
plt.imshow(lap,cmap = plt.cm.gray)
plt.title('Laplacian')
plt.subplot(133)
plt.imshow(sharp, cmap = plt.cm.gray)
plt.title('Sharpened')
plt.show()
#cv2.waitKey(0)
#cv2.waitKey(0)
