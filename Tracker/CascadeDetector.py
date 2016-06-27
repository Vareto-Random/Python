# import the necessary packages
from __future__ import print_function
from imutils import paths
import argparse
import imutils
import os
import cv2

# https://github.com/Itseez/opencv_contrib/blob/master/modules/tracking/samples/tracker.py

def main():
    cv2.namedWindow("Detection + Tracking")

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True, help="Path to your frame directories")
    args = vars(ap.parse_args())

    # initialize descriptor/person detector
    faceCascade = cv2.CascadeClassifier()
    faceCascade.load('cv_data/haarcascades/haarcascade_frontalface_default.xml')
    bodyCascade = cv2.CascadeClassifier()
    bodyCascade.load('cv_data/haarcascades/haarcascade_upperbody.xml')

    # using list comprehension
    #b = list(paths.list_images(args['images']))
    #a = [b[i] for i in range(0,len(b), 4)]
    #print (a)
    #exit

    listOfPath = list(paths.list_images(args["images"]))
    tracker = cv2.Tracker_create("MIL")

    for index in range(0, len(listOfPath)):
        # load the image and resize it to (1) reduce detection time and (2) improve detection accuracy
        image = cv2.imread(listOfPath[index])
        image = imutils.resize(image, width = min(500, image.shape[1]))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect people in the image
        bbox = bodyCascade.detectMultiScale(gray, 1.1, 3)
        if len(bbox) == 1:
            status = True
            while status:
                status = tracker.init(image, bbox)
                print(bbox)

                index = index + 1
                image = cv2.imread(listOfPath[index])
                status, newbox = tracker.update(image)

                if len(newbox) == 1:
                    p1 = (int(newbox[0]), int(newbox[1]))
                    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                    cv2.rectangle(image, p1, p2, (200, 0, 0))

            cv2.imshow('img', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)

if __name__ == "__main__": main()