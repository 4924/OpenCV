import cv2 as cv
import numpy as np
import matplotlib.pyplot as mpl
import os
import math

cap = cv.VideoCapture("greentraining.avi")
fourcc = cv.cv.CV_FOURCC(*'XVID')
out = cv.VideoWriter('output.avi',fourcc, 15.0, (320,240))


def callback(x):
    pass

#cv.namedWindow('thresh')
#
## Track bars
#cv.createTrackbar('lH','thresh', 0, 180, callback)
#cv.createTrackbar('lS', 'thresh', 0, 255, callback)
#cv.createTrackbar('lV', 'thresh', 0, 255, callback)
#cv.createTrackbar('uH', 'thresh', 0, 180, callback)
#cv.createTrackbar('uS', 'thresh', 0, 255, callback)
#cv.createTrackbar('uV', 'thresh', 0, 255, callback)
#
## capture video
#
#
#while True:
#    # Getting values from track bars
#    lH = cv.getTrackbarPos('lH', 'thresh')
#    uH = cv.getTrackbarPos('uH', 'thresh')
#    lS = cv.getTrackbarPos('lS', 'thresh')
#    uS = cv.getTrackbarPos('uS', 'thresh')
#    lV = cv.getTrackbarPos('lV', 'thresh')
#    uV = cv.getTrackbarPos('uV', 'thresh')
#
#
#    lowerb = np.array([lH, lS, lV], np.uint8)
#    upperb = np.array([uH, uS, uV], np.uint8)
#
#    #print lowerb
#    #print upperb
#
#    frame = cv.inRange(im4, lowerb, upperb)
#    cv.imshow('thresh', frame)
#
#    if cv.waitKey(1) & 0xFF == ord('q'):
#        break
#
##cap.release()
#cv.destroyAllWindows()
#
bol, frame = cap.read()
while bol:

    im = frame
#############################################################################################################################################

    im2 = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', im2)

    im4 = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    im6 = cv.inRange(im4, np.array([47, 42, 40]), np.array([100, 255, 255]))
    #cv.imshow('inRange', im6)
    kernel = np.ones((9,2),np.uint8)
    im7 = cv.morphologyEx(im6, cv.MORPH_OPEN, kernel, iterations = 1)
    #cv.imshow('open', im7)
    kernel = np.ones((4,1),np.uint8)
    im7 = cv.morphologyEx(im7, cv.MORPH_OPEN, kernel, iterations = 2)
    #cv.imshow('double', im7)

    contours2, hierarchy2 = cv.findContours(im7,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    total = 0
    for c in contours2:
        total += cv.contourArea(c)
    total = math.sqrt(math.sqrt(total))

    if(total > 0):
        kerneld = np.ones(( int(total*3), int(total*3) ),np.uint8)
        im8 = cv.dilate(im7,kerneld,iterations = 1)
        #cv.imshow('big', im8)
    else:
        im8 = im7
        print("Error!")


    contours, hierarchy = cv.findContours(im8,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)


    thing = []
    for c in contours:
        M = cv.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        x,y,w,h = cv.boundingRect(c)
        aspect_ratio = float(w)/h
        if(aspect_ratio<.85):
            thing.append([cX,cY, c])

    def dist(first, second):
        return math.sqrt(abs(first[0]-second[0])**2 + abs(first[1]-second[1])**2)

    i = 0
    maxs = 9999
    maxarray = []
    while i < len(thing):
        for p in thing[i+1:]:
            if dist(thing[i], p) < maxs:
                maxs = dist(thing[i], p)
                maxarray = [thing[i][2], p[2]]
        i += 1


    im2 = cv.cvtColor(im2, cv.COLOR_GRAY2BGR)
    cv.drawContours(im2, maxarray, -1, (0,0,255), 1)
    if(len(maxarray) > 1):
        print( cv.contourArea(maxarray[0])/cv.contourArea(maxarray[1]))
    total = math.sqrt(math.sqrt(total))
    cv.imshow('title', im2)
    out.write(im2)

    bol, frame = cap.read()
#############################################################################################################################################
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
out.release()
cap.release()
cv.destroyAllWindows()
