# Standard imports
import cv2 as cv
import numpy as np
import os
import math
import pickle
import time

import sys
import time
from networktables import NetworkTables

# To see messages from networktables, you must setup logging
import logging
logging.basicConfig(level=logging.DEBUG)

if len(sys.argv) != 2:
    print("Error: specify an IP to connect to!")
    exit(0)

ip = sys.argv[1]

NetworkTables.initialize(server=ip)

sd = NetworkTables.getTable("SmartDashboard")


#424
#425
#495
#496
#66
#67
#68
#69
#70
#71
#72
#73
#74
#75
#76
#83
#91
#92
#for s in os.listdir("2017/Vision Images/LED Peg"):
#    im = cv.imread("2017/Vision Images/LED Peg/" + s, cv.IMREAD_COLOR)
##if True:
#    #im = cv.imread("2017/Vision Images/LED Peg/1ftH1ftD0Angle0Brightness.jpg", cv.IMREAD_COLOR)
#
#    #print s
#
#    cv.imshow('im', im)
#    im = cv.resize(im, (320,240))
#    #im = cv.bitwise_not(im)
#
#    im2 = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
#    cv.imshow('gray', im2)
#
#    im4 = cv.cvtColor(im, cv.COLOR_BGR2HSV)
#    #im6 = cv.inRange(im4, np.array([79, 162, 50]), np.array([98, 255, 255]))
#    im6 = cv.inRange(im4, np.array([47, 42, 40]), np.array([100, 255, 255]))
#    ##im6 = cv.cvtColor(im5, cv.COLOR_HSV2BGR)
#    #cv.imshow('color', im)
#    cv.imshow('inRange', im6)
#    kernel = np.ones((9,2),np.uint8)
#    im7 = cv.morphologyEx(im6, cv.MORPH_OPEN, kernel, iterations = 1)
#    cv.imshow('open', im7)
#    kernel = np.ones((4,1),np.uint8)
#    im7 = cv.morphologyEx(im7, cv.MORPH_OPEN, kernel, iterations = 2)
#    cv.imshow('double', im7)
#
#    contours2, hierarchy2 = cv.findContours(im7,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
#    total = 0
#    for c in contours2:
#        total += cv.contourArea(c)
#    total = math.sqrt(math.sqrt(total))
#
#    if(total > 0):
#        kerneld = np.ones(( int(total*3), int(total*3) ),np.uint8)
#        im8 = cv.dilate(im7,kerneld,iterations = 1)
#        cv.imshow('big', im8)
#    else:
#        im8 = im7
#        print("Error!")
#
#
#    contours, hierarchy = cv.findContours(im8,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
#
#
#    #lan = [0]*5
#    thing = []
#    for c in contours:
#        M = cv.moments(c)
#        cX = int(M["m10"] / M["m00"])
#        cY = int(M["m01"] / M["m00"])
#        thing.append([cX,cY, c])
#        #areaConvex = cv.contourArea(c)
#        #for i in range(0,3):
#        #    if( areaConvex>=lan[i]):
#        #        lan.insert(i, areaConvex)
#        #        thing.insert(i, c)
#        #        break
#
#    def dist(first, second):
#        return math.sqrt(abs(first[0]-second[0])**2 + abs(first[1]-second[1])**2)
#
#    i = 0
#    maxs = 9999
#    maxarray = []
#    while i < len(thing):
#        for p in thing[i+1:]:
#            if dist(thing[i], p) < maxs:
#                maxs = dist(thing[i], p)
#                maxarray = [thing[i][2], p[2]]
#        i += 1
#
##(x,y),(MA,ma),angle = cv.fitEllipse(thing)
##print angle
##for i in range(0, 3):
##    x,y,w,h = cv.boundingRect(thing[0])
#    crop_im = im[y:y+h, x:x+w]
#    cv.imwrite("lolz/"+str(i)+".jpg", crop_im)
#print "hello"
#cv.imshow('crop', crop_im)


#save = thing
#saveHandler = open('example.contour', 'w')
#pickle.dump(save, saveHandler)

#saveHandler = open('example.contour', 'r')
#savedThing = pickle.load(saveHandler)
#print cv.matchShapes(thing, savedThing, 1, 0.0)

#print cv.contourArea(cv.convexHull(thing))
#print cv.contourArea(cv.convexHull(thing))/cv.contourArea(thing)*50

#    im2 = cv.cvtColor(im2, cv.COLOR_GRAY2BGR)
#    cv.drawContours(im2, maxarray, -1, (0,0,255), 1)
#    if(len(maxarray) > 1):
#        print( cv.contourArea(maxarray[0])/cv.contourArea(maxarray[1]))
#    total = math.sqrt(math.sqrt(total))
    #cv.imwrite("filesas/"+s, im4)
#    cv.imshow('title', im2)
#    cv.waitKey(0)
#cv.destroyAllWindows()
cap = cv.VideoCapture("http://10.13.37.51/mjpg/video.mjpg")

#def callback(x):
#    pass
#
#cv.namedWindow('thresh')
#
## Track bars
#cv.createTrackbar('lH','thresh', 0, 180, callback)
#cv.createTrackbar('lS', 'thresh', 0, 255, callback)
#cv.createTrackbar('lV', 'thresh', 0, 255, callback)
#cv.createTrackbar('uH', 'thresh', 0, 180, callback)
#cv.createTrackbar('uS', 'thresh', 0, 255, callback)
#cv.createTrackbar('uV', 'thresh', 0, 255, callback)

# capture video


while True:
    # Getting values from track bars
    bol, frame = cap.read()
    #lH = cv.getTrackbarPos('lH', 'thresh')
    #uH = cv.getTrackbarPos('uH', 'thresh')
    #lS = cv.getTrackbarPos('lS', 'thresh')
    #uS = cv.getTrackbarPos('uS', 'thresh')
    #lV = cv.getTrackbarPos('lV', 'thresh')
    #uV = cv.getTrackbarPos('uV', 'thresh')


    #lowerb = np.array([lH, lS, lV], np.uint8)
    #upperb = np.array([uH, uS, uV], np.uint8)

    #print lowerb
    #print upperb
    #cv.imshow('before', frame)
    #im4 = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #im6 = cv.inRange(im4, lowerb, upperb)
    #cv.imshow('thresh', im6)

    im = frame

    #im = cv.resize(im, (320,240))
    cv.imshow('color', im)
    im2 = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', im2)

    im4 = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    #im6 = cv.inRange(im4, np.array([47, 42, 40]), np.array([100, 255, 255]))
    #im6 = cv.inRange(im4, np.array([75, 187, 181]), np.array([99, 226, 235]))
    im6 = cv.inRange(im4, np.array([54, 166, 144]), np.array([101, 220, 255]))
    #im6 = cv.inRange(im4, lowerb, upperb)
    cv.imshow('inRange', im6)

    #kernel = np.ones((4,1),np.uint8)
    #im7 = cv.morphologyEx(im7, cv.MORPH_OPEN, kernel, iterations = 2)
    #cv.imshow('double', im7)

    #imc, contours2, hierarchy2 = cv.findContours(im7,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    #total = 0
    #for c in contours2:
    #    total += cv.contourArea(c)
    #total = math.sqrt(math.sqrt(total))

    #if(total > 0):
    #kerneld = np.ones(( int(total*3), int(total*3) ),np.uint8)
    kerneld = np.ones(( 4, 4 ),np.uint8)
    im8 = cv.dilate(im6,kerneld,iterations = 1)
    cv.imshow('big', im8)
    kernel = np.ones((9,2),np.uint8)
    im8 = cv.morphologyEx(im8, cv.MORPH_OPEN, kernel, iterations = 1)
    cv.imshow('open', im8)
    #else:
        #im8 = im7
        #print("Error!")


    imc, contours, hierarchy = cv.findContours(im8,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)


    thing = []
    for c in contours:
        M = cv.moments(c)
        cX = int(M["m10"] / (M["m00"]+0.000000000001))
        cY = int(M["m01"] / (M["m00"]+0.000000000001))
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
    done = []
    for m in maxarray:
        M = cv.moments(m)
        done.append(int(M["m10"] / (M["m00"]+0.000000000001)))
    if len(maxarray) > 0:
        if math.fabs(((sum(done)/len(done))/320)-.5) < .2:
            print((((sum(done)/len(done))/320)-.5)*3)
            sd.putNumber('cameraX', -(((sum(done)/len(done))/320)-.5)*3)
        elif math.fabs(((sum(done)/len(done))/320)-.5) < .4:
            print((30+((sum(done)/len(done))/320)-.5)*1.5)
            sd.putNumber('cameraX', -(((sum(done)/len(done))/320)-.5)*1.5)
        else:
            print(50+((sum(done)/len(done))/320)-.5)
            sd.putNumber('cameraX', -(((sum(done)/len(done))/320)-.5))

    cv.imshow('title', im2)

    #im = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #frame = cv.inRange(im, lowerb, upperb)
    #cv.imshow('threshs', im)
    #time.sleep(.05);
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
