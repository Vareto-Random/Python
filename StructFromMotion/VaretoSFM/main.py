# ======================================================================================================================
# Importing libraties
# ======================================================================================================================
from mpl_toolkits.mplot3d import Axes3D

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random

import Extra

# ======================================================================================================================
# Reading images
# ======================================================================================================================
MIN_MATCH_COUNT = 10
PATH_TO_IMAGES = 'vball_small/'

IMGA = 'leftside.jpg'
IMGB = 'front.jpg'
IMGC = 'rightside.jpg'

imageA = cv.imread(PATH_TO_IMAGES + IMGA)
imageB = cv.imread(PATH_TO_IMAGES + IMGB)
imageC = cv.imread(PATH_TO_IMAGES + IMGC)

# ======================================================================================================================
# Extracting SIFT features
# ======================================================================================================================
keypointsA, descriptorsA = Extra.sift(imageA)
keypointsB, descriptorsB = Extra.sift(imageB)
keypointsC, descriptorsC = Extra.sift(imageC)

# ======================================================================================================================
# Matching descriptors between images A-B and B-C
# ======================================================================================================================
matcher = cv.BFMatcher()
matchesAB = matcher.knnMatch(descriptorsA, descriptorsB, k=2)
matchesBC = matcher.knnMatch(descriptorsB, descriptorsC, k=2)

# ======================================================================================================================
# Filtering best point correspondence
# ======================================================================================================================
goodMatchesAB = []
for i, (a, b) in enumerate(matchesAB):
    if a.distance < 0.70 * b.distance:
        goodMatchesAB.append((matchesAB[i][0].queryIdx, matchesAB[i][0].trainIdx))
matchesAB, modelAB = Extra.ransac(goodMatchesAB, keypointsA, keypointsB)

goodMatchesBC = []
for i, (b, c) in enumerate(matchesBC):
    if b.distance < 0.70 * c.distance:
        goodMatchesBC.append((matchesBC[i][0].queryIdx, matchesBC[i][0].trainIdx))
matchesBC, modelBC = Extra.ransac(goodMatchesBC, keypointsB, keypointsC)

# ======================================================================================================================
# If minimum number of points satisfied
# ======================================================================================================================
if len(matchesAB) > MIN_MATCH_COUNT and len(matchesBC) > MIN_MATCH_COUNT:

    # ==================================================================================================================
    # Refactoration: generating measurement matrix and SVD
    # ==================================================================================================================
    factorAB = Extra.factorize(matchesAB, keypointsA, keypointsB)
    factorBC = Extra.factorize(matchesBC, keypointsB, keypointsC)

    # ==================================================================================================================
    # Choosing corresponding features in all three images
    # ==================================================================================================================
    descAB = []
    for match in matchesAB:
        descAB.append(descriptorsB[match[1]])

    descBC = []
    for match in matchesBC:
        descBC.append(descriptorsB[match[0]])

    descAB = np.array(descAB)
    descBC = np.array(descBC)


    # ==================================================================================================================
    # Filtering best point correspondence
    # ==================================================================================================================
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flannMatcher = cv.FlannBasedMatcher(index_params, search_params)
    matches = flannMatcher.knnMatch(descAB, descBC, k=2)

    goodMatches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            goodMatches.append((matches[i][0].queryIdx, matches[i][0].trainIdx))

    points1 = []
    points2 = []
    for match in goodMatches:
        points1.append((factorAB[match[0], 0], factorAB[match[0], 1], factorAB[match[0], 2]))
        points2.append((factorBC[match[1], 0], factorBC[match[1], 1], factorBC[match[1], 2]))
    inliers, model = Extra.ransac3D(goodMatches, points1, points2)
    homogenizedStruct = []

    for index in xrange(len(factorAB[:, 0])):
        homogenizedPoint = (factorAB[index, 0], factorAB[index, 1], factorAB[index, 2], 1.)
        homogenizedStruct.append(homogenizedPoint)
    homogenizedTranspose = model * np.transpose(homogenizedStruct)

    # ==================================================================================================================
    # Ploting points
    # ==================================================================================================================
    tempStructure = []
    for index in xrange(len(homogenizedTranspose[0, :])):
        xCoord = homogenizedTranspose[0, index]
        yCoord = homogenizedTranspose[1, index]
        zCoord = homogenizedTranspose[2, index]
        homogenizedCoord = homogenizedTranspose[3, index]
        xCoord = xCoord / homogenizedCoord
        yCoord = yCoord / homogenizedCoord
        zCoord = zCoord / homogenizedCoord
        tempStructure.append((xCoord, yCoord, zCoord))

    finalStructure = np.vstack((factorBC, tempStructure))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax._axis3don = False
    xs = []
    ys = []
    zs = []
    for index in xrange(len(finalStructure[:, 0])):
        xs.append(finalStructure[index, 0])
        ys.append(finalStructure[index, 1])
        zs.append(finalStructure[index, 2])
    ax.scatter(xs, ys, zs, c='r', marker='o')
    plt.show()
