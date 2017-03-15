import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("flower.png")

#converting from bgr to L*a*b*
hs = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
#split the image into planes
L,a,b = cv2.split(hs)

#create a copy of each L,a,b numpy arrays
L1 = np.copy(L)
a1 = np.copy(a)
b1 = np.copy(b)

#fill the L,a,b copy values to constant for displaying in color format
L1.fill(127)
a1.fill(128)
b1.fill(128)
Lt = cv2.cvtColor(cv2.merge([L,a1,b1]), cv2.COLOR_Lab2RGB)

#display the images in RGB form as matplotlib requires images in RGB
plt.figure(1)
plt.subplot(221)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title('original')
plt.subplot(222)
plt.imshow(Lt)
plt.title('L*')
plt.subplot(223)
plt.imshow(cv2.cvtColor(cv2.merge([L1,a,b1]), cv2.COLOR_Lab2RGB))
plt.title('a*')
plt.subplot(224)
plt.imshow(cv2.cvtColor(cv2.merge([L1,a1,b]), cv2.COLOR_Lab2RGB))
plt.title('b*')
plt.show()
