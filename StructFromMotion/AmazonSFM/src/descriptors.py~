import cv2 as cv
import re
import numpy as np
import os.path

PATH_TO_DATA = '../data/'

def check_file_exists(filename):
	return os.path.isfile(filename)

def sift(filename):
	img = cv.imread(PATH_TO_DATA+filename)
	sift = cv.xfeatures2d.SIFT_create()
	gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	kp, des = sift.detectAndCompute(img, None)
	print "sift",len(kp), len(des)
	folder = re.findall('(.*)/.*\.jpg', filename)[0]
	imgName = re.findall('.*/(.*)\.jpg', filename)[0]
	toWriteName = imgName + '.sift'
	fileToWrite = open(PATH_TO_DATA + folder + "/" + toWriteName, 'w')
	for index, d in enumerate(des):
		kpi = kp[index]
		kpList = [str(kpi.pt[0]), str(kpi.pt[1]), str(kpi.angle), str(kpi.size)]
		toWrite = ", ".join(kpList) + ':'
		for dimension in d:
			toWrite += str(dimension) + ','
		toWrite += '\n'	
		fileToWrite.write(toWrite)
	return kp, des

def surf(filename):
	img = cv.imread(PATH_TO_DATA+filename)
	surf = cv.xfeatures2d.SURF_create()
	gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	kp, des = surf.detectAndCompute(img, None)
	print "surf",len(kp), len(des)
	folder = re.findall('(.*)/.*\.jpg', filename)[0]
	imgName = re.findall('.*/(.*)\.jpg', filename)[0]
	toWriteName = imgName + '.surf'
	fileToWrite = open(PATH_TO_DATA + folder + "/" + toWriteName, 'w')
	for index, d in enumerate(des):
		kpi = kp[index]
		kpList = [str(kpi.pt[0]), str(kpi.pt[1]), str(kpi.angle), str(kpi.size)]
		toWrite = ", ".join(kpList) + ':'
		for dimension in d:
			toWrite += str(dimension) + ','
		toWrite += '\n'	
		fileToWrite.write(toWrite)
	return kp, des

def orb(filename):
	img = cv.imread(PATH_TO_DATA+filename)
	orb = cv.xfeatures2d.ORB_create()
	gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	kp, des = orb.detectAndCompute(img, None)
	print "orb",len(kp), len(des)
	folder = re.findall('(.*)/.*\.jpg', filename)[0]
	imgName = re.findall('.*/(.*)\.jpg', filename)[0]
	toWriteName = imgName + '.orb'
	fileToWrite = open(PATH_TO_DATA + folder + "/" + toWriteName, 'w')
	for index, d in enumerate(des):
		kpi = kp[index]
		kpList = [str(kpi.pt[0]), str(kpi.pt[1]), str(kpi.angle), str(kpi.size)]
		toWrite = ", ".join(kpList) + ':'
		for dimension in d:
			toWrite += str(dimension) + ','
		toWrite += '\n'	
		fileToWrite.write(toWrite)
	return kp, des

