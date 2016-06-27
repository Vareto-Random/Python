import cv2 as cv 
import numpy as np 
import ransac
import matplotlib
import matplotlib.pyplot as plt

KP_THRESH = 0.7

def match(img1, des1, kp1, img2, des2, kp2):
	des1 = np.array(des1)
	des2 = np.array(des2)
	height1, width1, depth1 = img1.shape
	height2, width2, depth2 = img2.shape
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   # or pass empty dictionary

	flann = cv.FlannBasedMatcher(index_params,search_params)

	matches = flann.knnMatch(des1,des2,k=2)

	goodMatches = []
	# ratio test as per Lowe's paper
	for i,(m,n) in enumerate(matches):
	    if m.distance < 0.7*n.distance:
	        goodMatches.append((matches[i][0].queryIdx,matches[i][0].trainIdx))

	matches = goodMatches
	matches, model = ransac.ransac(matches, kp1, kp2)
	print len(matches)
	im1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
	im2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
	img3 = np.zeros((max(height1,height2),width1+width2), np.uint8)
	img3[:height1, :width1] = im1
	img3[:height2, width1:width1+width2] = im2
	img3 = cv.cvtColor(img3,cv.COLOR_GRAY2BGR)
	xScale = 0.4
	yScale = 0.4
	img3 = cv.resize(img3, (0,0), fx=xScale, fy=yScale) 
	for match in matches:
		ind1, ind2 = match
		imgId = str(ind1)+str(ind2)
		pt1 = kp1[ind1]
		pt2 = kp2[ind2]
		pt1 = pt1.pt
		pt2 = pt2.pt
		pt1 = (int(int(pt1[0])*xScale), int(int(pt1[1])*yScale))
		pt2 = (int((int(pt2[0]) + width1)*xScale), int(int(pt2[1])*yScale))
		cv.line(img3, pt1, pt2, 255)
	imgId = str(ind1)+str(ind2)
		
	print "... displaying matches ... "
	cv.imshow('img' + imgId, img3)
	# plt.imshow(img3, cmap = 'gray', interpolation = 'bicubic')
	# plt.xticks([]), plt.yticks([])
	# plt.show()
	return matches

def matchKeypoints(des1, des2, thresh):
	matches = []
	for i in range(len(des1)):
		best_val = float('Inf')
		best_ind = 0
		second_val = float('Inf')
		v1 = np.array(des1[i])
		for j in range(len(des2)):
			v2 = np.array(des2[j])
			d = np.linalg.norm(v1-v2)
			if d < best_val:
				last_best_val = best_val
				best_val = d
				best_ind = j
				if last_best_val < second_val:
					second_val = last_best_val
			elif d < second_val:
				second_val = d
		d_ratio = float(best_val)/second_val
		if d_ratio < thresh:
			matches.append((i, best_ind))
	return matches