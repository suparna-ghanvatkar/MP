import cv2
import numpy as np

def stitch(images,ratio=0.75, reprojThresh=4.0):
    (img2,img1)=images   #images should be in left to right order
    (kps1,features1)=findDescriptors(img1)            
    (kps2,features2)=findDescriptors(img2)
    
    m = matchDescriptors( kps1,kps2,features1,features2,ratio, reprojThresh)
    
	#panorama cannot be created, no matches
    if m is None:       
        return None
    #keypoint matches, homography matrix,list of indexes of keypoints verified using Ransac
    (matches,H,status) = m    
    
    result = cv2.warpPerspective(img1, H,(img1.shape[1] + img2.shape[1], img1.shape[0])) 
    
    result[0:img2.shape[0], 0:img2.shape[1]] = img2  
    
    #to visualize the keypoint matches
    vis = drawMatches(img1, img2, kps1, kps2, matches,status)
 
    return (result, vis) #return stitched image and keypoint matches                  
	

def findDescriptors(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect keypoints using Differnce of Gaussian detector
    detector = cv2.FeatureDetector_create("SIFT")
    kps = detector.detect(gray)
 
	# extract features using SIFT festure extractor
    extractor = cv2.DescriptorExtractor_create("SIFT")
    (kps, features) = extractor.compute(gray, kps)
	
	# convert keypoints from objects to NumPy arrays
    kps = np.float32([kp.pt for kp in kps])
	
    return (kps, features)   # return keypoints and features
	


def matchDescriptors(kps1, kps2, features1, features2, ratio, reprojThresh): 

	# compute the raw matches 
    matcher = cv2.DescriptorMatcher_create("BruteForce")   #construct the feature matcher
    
	
    rawMatches = matcher.knnMatch(features1, features2, 2)  
    matches = []    #to store only good matches 
 
	# loop over raw matches to apply Lowe's ratio test 
    for m in rawMatches:
		# compute good matches
	    if len(m) == 2 and m[0].distance < m[1].distance * ratio:
			matches.append((m[0].trainIdx, m[0].queryIdx))
			
	
	# computing a homography requires at least 4 matches
    if len(matches) > 4:
        pts1 = np.float32([kps1[i] for (_, i) in matches])
        pts2 = np.float32([kps2[i] for (i, _) in matches])
 
		# compute homography
        (H,status)= cv2.findHomography(pts1,pts2,cv2.RANSAC,reprojThresh)#rthresh is maximum pixel wiggle room allowed by RANSAC
 
        return (matches, H, status)
 
    return None    # if no homograpy could be computed
	
	

def drawMatches(img1, img2, kps1, kps2, matches, status):
	# initialize the output visualization image
    (h1, w1) = img1.shape[:2]
    (h2, w2) = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")    #images stacked together in result
    vis[0:h1, 0:w1] = img1
    vis[0:h2, w1:] = img2
 
	# loop over the matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):
		#process the match only if the keypoint was successfully matched
    	if s == 1:
    		pt1 = (int(kps1[queryIdx][0]), int(kps1[queryIdx][1]))  #queryIdx is index of the query descriptors(2nd i/p image)
    		pt2 = (int(kps2[trainIdx][0]) + w1, int(kps2[trainIdx][1])) #trainIdx is index of the train descriptors(1st i/p image)
    		cv2.line(vis, pt1, pt2, (0, 255, 0), 1)    #draw line between the matching points
 
    return vis
	
	
	
# load the two images
imgA = cv2.imread('dataset/img1.jpg',1)
imgB = cv2.imread('dataset/img2.jpg',1)
 
# stitch the images together to create a panorama
(result, vis) = stitch([imgA, imgB])
 
# show the images
cv2.imshow("Image A", imgA)
cv2.imshow("Image B", imgB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
