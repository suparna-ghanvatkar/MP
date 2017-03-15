import cv2
import numpy as np
import sys

#read image from argument passed
img = cv2.imread(sys.argv[1])

#create copies for storing hough lines
imgh = img.copy()
imgh.fill(0)
imgp = img.copy()
imgp.fill(0)

#find contours using Canny
edges = cv2.Canny(img, 50,350)
thresh,contours = cv2.threshold(edges, 128, 255, cv2.THRESH_BINARY)
cv2.imshow('t',contours)

#draw probablistic hough lines on imgp image
lines = cv2.HoughLinesP(contours, 5, np.pi/180,180,60,10)
#print lines
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(imgp, (x1,y1), (x2,y2),(0,255,0),4)

#feedback loop for determining the threshold or houghVote
hvote = 200
while True:
    hlines = cv2.HoughLines(contours, 1, np.pi/180, hvote)
    if hlines==None:
        hvote = hvote-10
    else:
        break
if hlines.shape[0] >2 or hvote<1:
    hvote = 200
else:
    hvote = hvote+25
print hlines.shape[0]
while hlines.shape[0]<3 and hvote>0:
    hlines1 = cv2.HoughLines(contours,1,np.pi/180,hvote)
    if hlines1!=None:
        hlines = hlines1
    hvote = hvote-5
    print hlines.shape[0]
print 'out of loop' + str(hlines.shape[0])

#draw the lines detected from hough transform onto imgh
for i in xrange(hlines.shape[0]):
    for rho,theta in hlines[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(imgh,(x1,y1),(x2,y2),(0,255,0),2)
cv2.imshow('i',imgh)
cv2.imshow('h',imgp)

#bitwise and the lines obtained from both transform to eliminate lines at horizon etc.
imgh = cv2.bitwise_and(imgh,imgp)
img = cv2.bitwise_or(imgh,img)
cv2.imshow('houghlines3',img)

cv2.waitKey(0)
