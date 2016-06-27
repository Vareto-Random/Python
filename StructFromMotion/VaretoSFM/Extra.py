import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import random
import scipy.linalg


# My Methods
# ======================================================================================================================

def sift(img):
    sift = cv.xfeatures2d.SIFT_create()
    key, desc = sift.detectAndCompute(img, None)
    print "sift", len(key), len(desc)
    return key, desc


def surf(img):
    surf = cv.xfeatures2d.SURF_create()
    key, desc = surf.detectAndCompute(img, None)
    print "surf", len(key), len(desc)
    return key, desc


def orb(img):
    surf = cv.xfeatures2d.ORB_create()
    key, desc = surf.detectAndCompute(img, None)
    print "surf", len(key), len(desc)
    return key, desc


# Internet Methods
# ======================================================================================================================

NUM_ITER = 1000
NUM_SAMPLES = 4
NUM_SAMPLES3D = 8
PIXEL_THRESH = 50


def ransac(matches, kp1, kp2):
    pts1 = []
    pts2 = []
    for match in matches:
        ind1, ind2 = match
        pt1 = kp1[ind1].pt
        pt2 = kp2[ind2].pt
        pts1.append((pt1[0], pt1[1], 1.))
        pts2.append((pt2[0], pt2[1], 1.))
    inliers = []
    numInliers = 0
    model = None
    for i in range(NUM_ITER):
        if i % 100 == 0:
            print i
        sampleInd = random.sample(range(len(matches)), NUM_SAMPLES)
        samples1 = np.zeros((3, NUM_SAMPLES))
        samples2 = np.zeros((3, NUM_SAMPLES))
        for j in range(NUM_SAMPLES):
            samples1[:, j] = pts1[sampleInd[j]]
            samples2[:, j] = pts2[sampleInd[j]]
        H = np.matrix(samples2) * np.matrix(samples1).I
        inlierTemp = []
        for j in range(len(pts1)):
            x = np.matrix(pts1[j]).T
            xPrime = H * x
            x2 = np.matrix(pts2[j]).T
            xPrime[0] = xPrime[0][0] / xPrime[2][0]
            xPrime[1] = xPrime[1][0] / xPrime[2][0]
            xPrime = xPrime[0:2]
            x2[0] = x2[0][0] / x2[2][0]
            x2[1] = x2[1][0] / x2[2][0]
            x2 = x2[0:2]
            diff = np.linalg.norm(x2 - xPrime)
            if diff < PIXEL_THRESH:
                inlierTemp.append(matches[j])
        if len(inlierTemp) > numInliers:
            numInliers = len(inlierTemp)
            inliers = inlierTemp
            model = H
    return inliers, model


def ransac3D(matches, p1, p2):
    pts1 = []
    pts2 = []
    for pt in p1:
        pts1.append((pt[0], pt[1], pt[2], 1.))
    for pt in p2:
        pts2.append((pt[0], pt[1], pt[2], 1.))
    inliers = []
    numInliers = 0
    model = None
    for i in range(NUM_ITER):
        if i % 100 == 0:
            print i
        sampleInd = random.sample(range(len(matches)), NUM_SAMPLES3D)
        samples1 = np.zeros((4, NUM_SAMPLES3D))
        samples2 = np.zeros((4, NUM_SAMPLES3D))
        for j in range(NUM_SAMPLES3D):
            samples1[:, j] = pts1[sampleInd[j]]
            samples2[:, j] = pts2[sampleInd[j]]
        H = np.matrix(samples2) * np.matrix(samples1).I
        inlierTemp = []
        for j in range(len(pts1)):
            x = np.matrix(pts1[j]).T
            xPrime = H * x
            x2 = np.matrix(pts2[j]).T
            xPrime[0] = xPrime[0][0] / xPrime[3][0]
            xPrime[1] = xPrime[1][0] / xPrime[3][0]
            xPrime[2] = xPrime[2][0] / xPrime[3][0]
            x2[0] = x2[0][0] / x2[3][0]
            x2[1] = x2[1][0] / x2[3][0]
            x2[2] = x2[2][0] / x2[3][0]
            diff = np.linalg.norm(x2 - xPrime)
            if diff < PIXEL_THRESH:
                inlierTemp.append(matches[j])
        if len(inlierTemp) > numInliers:
            numInliers = len(inlierTemp)
            inliers = inlierTemp
            model = H
    return inliers, model


def factorize(matches, kp1, kp2):
    # create measurement matrix
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
    x1avg = sum(x1s) / float(len(x1s))
    y1avg = sum(y1s) / float(len(y1s))
    x2avg = sum(x2s) / float(len(x2s))
    y2avg = sum(y2s) / float(len(y2s))
    for index in xrange(len(x1s)):
        x1s[index] -= x1avg
        y1s[index] -= y1avg
        x2s[index] -= x2avg
        y2s[index] -= y2avg
    D.append(x1s)
    D.append(y1s)
    D.append(x2s)
    D.append(y2s)
    D = np.matrix(D)

    # decompose matrix
    U, W, V = np.linalg.svd(D)
    V = np.transpose(V)
    U3 = U[:, 0:3]
    W3 = W[0:3]
    V3 = V[:, 0:3]
    W3 = np.diag(W3)
    structure = scipy.linalg.sqrtm(W3) * np.transpose(V3)
    factor = np.transpose(structure)
    return factor


def decomposeMatrix(D):
    U, W, V = np.linalg.svd(D)
    V = np.transpose(V)
    U3 = U[:, 0:3]
    W3 = W[0:3]
    V3 = V[:, 0:3]
    W3 = np.diag(W3)
    structure = scipy.linalg.sqrtm(W3) * np.transpose(V3)
    plotStructure = np.transpose(structure)
    return plotStructure
