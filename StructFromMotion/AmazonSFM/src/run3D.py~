import cv2 as cv
import descriptors
import detection
import numpy as np
import ransac
#import scipy.linalg
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotStructure
import makeD

PATH_TO_DATA = '../data/'

IMAGE1 = 'vball_tiny/leftside.jpg'
IMAGE2 = 'vball_tiny/front.jpg'
IMAGE3 = 'vball_tiny/rightside.jpg'

img1 = cv.imread(PATH_TO_DATA + IMAGE1)
img2 = cv.imread(PATH_TO_DATA + IMAGE2)
img3 = cv.imread(PATH_TO_DATA + IMAGE3)
kp1, des1 = descriptors.surf(IMAGE1)
kp2, des2 = descriptors.surf(IMAGE2)
kp3, des3 = descriptors.surf(IMAGE3)
matches1 = detection.match(img1, des1, kp1, img2, des2, kp2)
matches2 = detection.match(img2, des2, kp2, img3, des3, kp3)
D1 = makeD.makeD(matches1,kp1,kp2)
D2 = makeD.makeD(matches2,kp2,kp3)
plotStructure1 = plotStructure.plotStructure(D1)
plotStructure2 = plotStructure.plotStructure(D2)



d1 = []
for match in matches1:
	d1.append(des2[match[1]])
	print des2[match[1]]
cv.waitKey()
d1 = np.array(d1)	
cv.waitKey()

d2 = []
for match in matches2:
	d2.append(des2[match[0]])
	print des2[match[0]]

#d1 = np.array(d1)
d2 = np.array(d2)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(d1,d2,k=2)

goodMatches = []
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        goodMatches.append((matches[i][0].queryIdx,matches[i][0].trainIdx))

matches = goodMatches

pts1 = []
pts2 = []


for match in matches:
	pts1.append((plotStructure1[match[0],0],plotStructure1[match[0],1],plotStructure1[match[0],2]))
	pts2.append((plotStructure2[match[1],0],plotStructure2[match[1],1],plotStructure2[match[1],2]))


inliers, model = ransac.ransac3D(matches,pts1,pts2)

homogenizedStruct = []

for index in xrange(len(plotStructure1[:,0])):
	homogenizedPoint = (plotStructure1[index,0],plotStructure1[index,1],plotStructure1[index,2],1.)
	homogenizedStruct.append(homogenizedPoint)

homogenizedTranspose = model*np.transpose(homogenizedStruct)

plotStructure3 = []

for index in xrange(len(homogenizedTranspose[0,:])):
	xCoord = homogenizedTranspose[0,index]
	yCoord = homogenizedTranspose[1,index]
	zCoord = homogenizedTranspose[2,index]
	homogenizedCoord = homogenizedTranspose[3,index]
	xCoord = xCoord/homogenizedCoord
	yCoord = yCoord/homogenizedCoord
	zCoord = zCoord/homogenizedCoord
	plotStructure3.append((xCoord,yCoord,zCoord))

plotStructure = np.vstack((plotStructure2,plotStructure3))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax._axis3don = False
xs = []
ys = []
zs = []
for index in xrange(len(plotStructure[:,0])):
	xs.append(plotStructure[index,0])
	ys.append(plotStructure[index,1])
	zs.append(plotStructure[index,2])
ax.scatter(xs,ys,zs, c='r', marker='o')
plt.show()

