import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

#defining the image size as 200*200
img = np.zeros((200,200))
#Taking random number of lines in range (10,25)
num_lines = random.randint(10,25)


#taking random width lines and drawing them in the image
i = 0
width = random.randint(0,5)
#width = 5
while i < num_lines:
    vert = random.randint(0,199)
    hori = random.randint(0,199)
    img[0:199, vert : vert + width] = 255
    img[hori : hori+width, 0:199] = 255
    i = i + 1
out = np.copy(img)


#applying sobel filter with kernel size = 3
sobelx = cv2.Sobel(img, cv2.CV_64F, 1,0, ksize = 3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 3)

#h-stacking the images
res = np.hstack((img,sobelx, sobely))

#Defining the kernel matrix for Prewitt operator and them applying the prewitt operator
prewitt_y = np.array(([-1,-1,-1],[0,0,0],[1,1,1]),dtype="int")
prewitt_x = np.array(([-1,0,1],[-1,0,1],[-1,0,1]), dtype = "int")
prew_x = np.zeros((200,200))
prew_y = np.zeros((200,200))
prew_x = cv2.filter2D(img, -1, prewitt_x)
prew_y = cv2.filter2D(img, -1, prewitt_y)

#Displaying all the result together
plt.figure(1)
plt.subplot(231)
plt.imshow(img, cmap = plt.cm.gray)
plt.title('Original')
plt.subplot(232)
plt.imshow(sobelx,cmap = plt.cm.gray)
plt.title('SobelX')
plt.subplot(233)
plt.imshow(sobely, cmap = plt.cm.gray)
plt.title('SobelY')
plt.subplot(234)
plt.imshow(img, cmap = plt.cm.gray)
plt.title('Original')
plt.subplot(235)
plt.imshow(prew_x, cmap = plt.cm.gray)
plt.title('Prewitt_X')
plt.subplot(236)
plt.imshow(prew_y, cmap = plt.cm.gray)
plt.title('Prewitt_y')
plt.show()

#result = np.vstack((res,np.hstack((img, prew_x,prew_y))))
#cv2.imshow('imagee',result)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
