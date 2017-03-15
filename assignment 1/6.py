import cv2
import numpy as np
from matplotlib import pyplot as plt


#reading the image and a copy
img = cv2.imread("res/noisy.png",cv2.IMREAD_GRAYSCALE)
img1 = img

#Gaussian blur for values range from 3 to 15 and showing them one by one
i = 3
while i <= 15:
	blur = cv2.GaussianBlur(img, (i,i), 0)	
	res = np.hstack((img,blur)) #stacking images side-by-side
	plt.figure(1)
	plt.subplot(121)
	plt.imshow(img,cmap = plt.cm.gray)
	plt.title('Original')
	plt.subplot(122)
	plt.imshow(blur,cmap = plt.cm.gray)
	plt.title("ksize=  " +  str(i))
	plt.show()
	#cv2.imwrite("out/" + str(i) + ".jpg" ,res)
	#cv2.waitKey(0)
	i = i + 2

