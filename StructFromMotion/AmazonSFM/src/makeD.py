import numpy as np 
PATH_TO_DATA = '../data/'

def makeD(matches, kp1, kp2):
	D = []
	x1s = []
	y1s = []
	x2s = []
	y2s = []
	for index in range(len(matches)):
		point1 = kp1[matches[index][0]].pt
		point2 = kp2[matches[index][1]].pt
		x1s.append(point1[0])
		y1s.append(point1[1])
		x2s.append(point2[0])
		y2s.append(point2[1])
	x1Avg = sum(x1s)/float(len(x1s))
	y1Avg = sum(y1s)/float(len(y1s))
	x2Avg = sum(x2s)/float(len(x2s))
	y2Avg = sum(y2s)/float(len(y2s))
	for index in xrange(len(x1s)):
		x1s[index] -= x1Avg
		y1s[index] -= y1Avg
		x2s[index] -= x2Avg
		y2s[index] -= y2Avg
	D.append(x1s)
	D.append(y1s)
	D.append(x2s)
	D.append(y2s)
	D = np.matrix(D)
	return D

def readD():
	cFilename = raw_input('Give the name of the correspondece file from /data/ : ')
	cFile = open(PATH_TO_DATA + cFilename, 'r')
	D = []
	x1s = []
	y1s = []
	x2s = []
	y2s = []
	for line in cFile:
		x1, y1, x2, y2 = line.split()
		x1s.append(float(x1))
		y1s.append(float(y1))
		x2s.append(float(x2))
		y2s.append(float(y2))
	x1Avg = sum(x1s)/float(len(x1s))
	y1Avg = sum(y1s)/float(len(y1s))
	x2Avg = sum(x2s)/float(len(x2s))
	y2Avg = sum(y2s)/float(len(y2s))
	for index in xrange(len(x1s)):
		x1s[index] -= x1Avg
		y1s[index] -= y1Avg
		x2s[index] -= x2Avg
		y2s[index] -= y2Avg
	D.append(x1s)
	D.append(x2s)
	D.append(y1s)
	D.append(y2s)
	D = np.matrix(D)
	return D
