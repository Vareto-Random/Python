import cv2 as cv 
import numpy as np
import scipy.linalg
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plotStructure(D):
	U, W, V = np.linalg.svd(D)
	V = np.transpose(V)
	U3 = U[:,0:3]
	W3 = W[0:3]
	V3 = V[:,0:3]
	W3 = np.diag(W3)
	Structure = scipy.linalg.sqrtm(W3)* np.transpose(V3)
	plotStructure = np.transpose(Structure)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	xs = []
	ys = []
	zs = []
	for index in xrange(len(plotStructure[:,1])):
		xs.append(plotStructure[index,0])
		ys.append(plotStructure[index,1])
		zs.append(plotStructure[index,2])
	ax.scatter(xs,ys,zs, c='r', marker='o')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	plt.show()
	return plotStructure
