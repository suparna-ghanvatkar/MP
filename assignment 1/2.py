import cv2
import numpy as np
from matplotlib import pyplot as plt

#reading the image
img = cv2.imread("hsv1.png")

#converting from bgr to hsv
hs = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#split the image into HSV planes
h, s, v = cv2.split(hs)

#create a copy of each h,s,v numpy arrays
h1 = np.copy(h)
s1 = np.copy(s)
v1 = np.copy(v)

#fill the saturation and values copy to 255 for displaying hue in color format
s1.fill(255)
v1.fill(255)
h1 = cv2.cvtColor(cv2.merge([h,s1,v1]), cv2.COLOR_HSV2RGB)

#display the images in RGB form as matplotlib requires images in RGB
plt.figure(1)
plt.subplot(221)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title('original')
plt.subplot(222)
plt.imshow(h1)
plt.title('Hue')
plt.subplot(223)
plt.imshow(s, cmap=plt.cm.gray)
plt.title('Saturation')
plt.subplot(224)
plt.imshow(v, cmap = plt.cm.gray)
plt.title('Value')
plt.show()

#convert to HSL
hs = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
#split into H,S,L planes
h, l, s = cv2.split(hs)

#create a copy of S and L and fill them with fixed values and display the hue in color format
s1 = np.copy(s)
l1 = np.copy(l)
s1.fill(255)
l1.fill(128)
h1 = cv2.cvtColor(cv2.merge([h,l1,s1]), cv2.COLOR_HLS2RGB)

#display image using matplotlib
plt.figure(2)
plt.subplot(221)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title('original')
plt.subplot(222)
plt.imshow(h1)
plt.title('Hue')
plt.subplot(223)
plt.imshow(s,cmap=plt.cm.gray)
plt.title('Saturation')
plt.subplot(224)
plt.imshow(l, cmap=plt.cm.gray)
plt.title('Lightness')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
