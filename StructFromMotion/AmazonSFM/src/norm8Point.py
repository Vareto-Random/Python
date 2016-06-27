import numpy as np
import math
import re
import cv2 as cv 

PATH_TO_DATA = '../data/'

def normalize(points1, points2):
	# take averages
	x1_avg = float(sum(x[0] for x in points1))/(len(points1))
	y1_avg = float(sum(x[1] for x in points1))/(len(points1))
	x2_avg = float(sum(x[0] for x in points2))/(len(points2))
	y2_avg = float(sum(x[1] for x in points2))/(len(points2))
	# compute RMS 
	RMS1 = 0.0
	RMS2 = 0.0
	for x in points1:
		RMS1 += math.sqrt(math.pow(x[0]-x1_avg, 2) + math.pow(x[1]-y1_avg, 2))
	for x in points2:
		RMS2 += math.sqrt(math.pow(x[0]-x2_avg, 2) + math.pow(x[1]-y2_avg, 2))
	RMS1 = float(RMS1)/len(points1)
	RMS2 = float(RMS2)/len(points2)
	s1 = math.sqrt(2.)/RMS1
	s2 = math.sqrt(2.)/RMS2
	# make Ts
	T1 = np.matrix([[s1, 0, -s1*x1_avg],[0, s1, -s1*y1_avg],[0, 0, 1]])
	T2 = np.matrix([[s2, 0, -s2*x2_avg],[0, s2, -s2*y2_avg],[0, 0, 1]])
	homoPoints1 = []
	for x in points1:
		homoPoints1.append([x[0], x[1], 1])
	homoPoints2 = []
	for x in points2:
		homoPoints2.append([x[0], x[1], 1])
	homoPoints1 = np.matrix(homoPoints1)
	homoPoints2 = np.matrix(homoPoints2)
	# apply Ts
	newPoints1 = np.transpose(T1*np.transpose(homoPoints1))
	newPoints2 = np.transpose(T2*np.transpose(homoPoints2))
	return newPoints1, newPoints2, T1, T2

def constructW(A, B):
	W = []
	for i in range(len(A)):
		a = A[i]
		b = B[i]
		row = [a.item(0)*b.item(0), a.item(1)*b.item(0), b.item(0), a.item(0)*b.item(1), a.item(1)*b.item(1), b.item(1), a.item(0), a.item(1), 1.]
		W.append(row)
	W = np.matrix(W)
	return W 

def getImages(cFilename, points1, points2):
	folderName = re.findall('(\w+/)\w+', cFilename)[0]
	img1Name = re.findall('(\w+)_to_\w+', cFilename)[0]
	img2Name = re.findall('\w+_to_(\w+)', cFilename)[0]
	img1 = cv.imread(PATH_TO_DATA+folderName+img1Name+'.jpg')
	img2 = cv.imread(PATH_TO_DATA+folderName+img2Name+'.jpg')
	cv.namedWindow('IMAGE 1')
	cv.namedWindow('IMAGE 2')
	for point in points1:
		pt = (int(point[0]), int(point[1]))
		cv.circle(img1, pt, 2, (255, 255, 0), -1)
	for point in points2:
		pt = (int(point[0]), int(point[1]))
		cv.circle(img2, pt, 2, (255, 255, 0), -1)
	return img1, img2

def drawEpipolars(F, points1, points2, img):
	for index, point in enumerate(points1):
		p = np.array((point[0], point[1], 1.0))
		pPrime = np.array((points2[index][0], points2[index][1], 1.0))

		d = F.dot(p)
		dA = d.item(0)
		dB = d.item(1)
		dC = d.item(2)

		x = pPrime.item(0)
		y = pPrime.item(1)

		x1 = int(x-10)
		x2 = int(x+10)

		y1 = int(float(-dA*(x1)-dC)/dB)
		y2 = int(float(-dA*(x2)-dC)/dB)

		cv.line(img, (x1, y1), (x2, y2), (255, 255, 0), 4)

def run():
	# open file 
	cFilename = raw_input('Give the name of the correspondece file from /data/ : ')
	cFile = open(PATH_TO_DATA + cFilename, 'r')
	points1 = []
	points2 = []
	for line in cFile:
		x1, y1, x2, y2 = line.split()
		points1.append([float(x1), float(y1)])
		points2.append([float(x2), float(y2)])
	img1, img2 = getImages(cFilename, points1, points2)
	#normalize
	normPoints1, normPoints2, T1, T2 = normalize(points1, points2)
	#create W such that Wf = 0
	W = constructW(normPoints1, normPoints2)
	# take SVD of W to get f
	U, D, V = np.linalg.svd(W, full_matrices=True)
	V = np.transpose(V)
	f = V[:, 8]
	f = f.reshape(3,3)
	# SVD new f to make rank 2
	U, D, V = np.linalg.svd(f)
	D.itemset(2, 0.0)
	F = U*np.diag(D)*V
	# de normalize
	F = np.transpose(T2)*F*T1
	drawEpipolars(F, points1, points2, img2)
	drawEpipolars(np.transpose(F), points2, points1, img1)
	cv.imshow('IMAGE 1', img1)
	cv.imshow('IMAGE 2', img2)
	cv.waitKey(0)
	cv.destroyAllWindows()


