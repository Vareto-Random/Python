import cv2 as cv 
import re
import numpy as np


PATH_TO_DATA = '../data/'

def getxy(event, x, y, flags, param):
	if event == cv.EVENT_LBUTTONDOWN:
		global correspondences
		if (len(correspondences[0]) > len(correspondences[1])):
			#image 2
			correspondences[1].append((x, y))
			print "Now click correspondence for image 1"
		else:
			#image 1 
			correspondences[0].append((x, y))
			print "Now click correspondence for image 2"

correspondences = [[], []]

img1Filename = raw_input('Image 1 file from /data/ : ')
img2Filename = raw_input('Image 2 file from /data/ : ')
img1 = cv.imread(PATH_TO_DATA+img1Filename)
img2 = cv.imread(PATH_TO_DATA+img2Filename)

cv.namedWindow('IMAGE 1')
cv.setMouseCallback('IMAGE 1', getxy)

cv.namedWindow('IMAGE 2')
cv.setMouseCallback('IMAGE 2', getxy)

cv.imshow('IMAGE 1', img1)
cv.imshow('IMAGE 2', img2)

print "Click correspondence for image 1"

cv.waitKey(0)
cv.destroyAllWindows()

gray= cv.cvtColor(img1,cv.COLOR_BGR2GRAY)

sift = cv.SIFT()
kp = sift.detect(gray,None)

img1=cv.drawKeypoints(gray,kp)

cv.imwrite('sift_keypoints1.jpg',img1)

gray= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

sift = cv.SIFT()
kp = sift.detect(gray,None)

img2=cv.drawKeypoints(gray,kp)

cv.imwrite('sift_keypoints2.jpg',img2)

print correspondences[0]
print correspondences[1]

folder = re.findall('(.*)/.*\.jpg', img1Filename)[0]
img1Name = re.findall('.*/(.*)\.jpg', img1Filename)[0]
img2Name = re.findall('.*/(.*)\.jpg', img2Filename)[0]
toWriteName = img1Name + "_to_" + img2Name
fileToWrite = open(PATH_TO_DATA + folder + "/" + toWriteName, 'w')
for i in range(min(len(correspondences[0]), len(correspondences[1]))):
	toWrite = str(correspondences[0][i][0]) + " " + str(correspondences[0][i][1]) + "\t" + str(correspondences[1][i][0]) + " " + str(correspondences[1][i][1]) + "\n"
	fileToWrite.write(toWrite)








