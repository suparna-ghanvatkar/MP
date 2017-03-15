import cv2
from matplotlib import pyplot as plt

#reading the image
img = cv2.imread('res/Lenna.jpg')
#splitting the channels
b,g,r = cv2.split(img)

#plotting using matplotlib
plt.figure(1)
plt.subplot(221)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title('original')
plt.subplot(222)
plt.imshow(b,cmap=plt.cm.gray)
plt.title('blue')
plt.subplot(223)
plt.imshow(g, cmap=plt.cm.gray)
plt.title('green')
plt.subplot(224)
plt.imshow(r, cmap=plt.cm.gray)
plt.title('red')
plt.show()
